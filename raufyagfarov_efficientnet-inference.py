!pip install ../input/efficientnetpytorch/EfficientNet-PyTorch-master
import math

import heapq

import torch

import torch.nn as nn

from torch.cuda import amp

from efficientnet_pytorch import EfficientNet

from torch.nn import Parameter

import torch.nn.functional as F

from collections import OrderedDict

import pandas as pd

import numpy as np

from tqdm import tqdm

from sklearn.metrics.pairwise import cosine_similarity

import albumentations as A

from albumentations.pytorch import ToTensorV2

import os

import cv2

import glob

from shutil import copyfile

import multiprocessing

from torch.utils.data import Dataset, DataLoader



import time

import copy

import gc

import operator

import pathlib

import PIL

import pydegensac

from scipy import spatial

import tensorflow as tf
DEBUG = False



train_data_dir = '/kaggle/input/landmark-recognition-2020/train/'

test_data_dir = '/kaggle/input/landmark-recognition-2020/test/'

train_csv_path = '/kaggle/input/landmark-recognition-2020/train.csv'



image_size = 640

embeddings_size = 512

batch_size = 30

num_workers = multiprocessing.cpu_count()

n_chunks = 200 if not DEBUG else 20

n_top_global = 5

n_top_local = 5

calculate_local = True

reranking = True

train_encodings_from_file = False

N_rerank_check = 10 if DEBUG else 5000

N_rerank_select = 5 if DEBUG else 300

reranking_th = 25



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def generate_test_df(data_dir, img_ext='.jpg'):

    files = [f.split('/')[-1].replace(img_ext,'') for f in glob.glob(data_dir + f'**/*{img_ext}',recursive=True)]

    return pd.DataFrame(files, columns=['id'])



def my_collate(batch):

    all_images = []

    all_names = []

    

    total_n_transforms = len(batch[0][0])

    

    for i in range(total_n_transforms):

        curr_imgs_list = []

        curr_name_list = []

        for b in batch:

            curr_imgs_list.append(b[0][i])

            curr_name_list.append(b[1][i])

        curr_imgs_list = torch.stack(curr_imgs_list)

        all_images.append(curr_imgs_list)

        all_names.append(curr_name_list)

    return [all_images, all_names]





class GLRDataset(Dataset):

    def __init__(self, data_dir, transform, df_path=None, mapper=None, img_ext='.jpg'):

        super().__init__()

        self.df = pd.read_csv(df_path) if df_path is not None else generate_test_df(data_dir,img_ext=img_ext)

        self.data_dir = data_dir

        self.transform = transform

        self.img_ext = img_ext

        self.n_classes = self.df['landmark_id'].nunique() if 'landmark_id' in self.df else None

        self.n_samples = len(self.df)

        self.mapper = mapper



    def __len__(self):

        return len(self.df)

          

    def __getitem__(self, index):       

        img_name = self.df['id'].iloc[index]

        f1,f2,f3 = img_name[:3]

        img_path = os.path.join(self.data_dir, f1, f2, f3, img_name+self.img_ext)

        images = []

        img_names = []

        image = cv2.imread(img_path)  # BGR

        

        for trans in self.transform:

            augmented = trans(image=image)

            images.append(augmented['image'])

            img_names.append(img_name)

        return images, img_names





def get_dataloader(data_dir, transform=None, batch_size=8, df_path=None, 

                             mapper=None, num_workers=2, 

                             pin_memory=True, drop_last=True, collate_fn=None):

    dataset = GLRDataset(data_dir=data_dir, df_path=df_path, mapper=mapper, transform=transform)

    sampler = None

    data_loader = DataLoader(dataset=dataset, 

                             batch_size=batch_size, 

                             sampler=sampler, 

                             pin_memory=pin_memory,

                             drop_last=drop_last,

                             collate_fn=collate_fn,

                             num_workers=num_workers)

    return data_loader, dataset



def load_ckp(checkpoint_fpath, model, optimizer=None, margin=None, remove_module=False):

    checkpoint = torch.load(checkpoint_fpath)



    pretrained_dict = checkpoint['model']

    model_state_dict = model.state_dict()

    if remove_module:

        new_state_dict = OrderedDict()

        for k, v in pretrained_dict.items():

            name = k[7:] # remove 'module.' of DataParallel/DistributedDataParallel

            new_state_dict[name] = v

        pretrained_dict = new_state_dict

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_state_dict}

    model_state_dict.update(pretrained_dict)

 



    model.load_state_dict(pretrained_dict)



    try:

        optimizer.load_state_dict(checkpoint['optimizer'])

    except:

        print('Cannot load optimizer params')

        

    if margin is not None and 'margin' in checkpoint:

            margin.load_state_dict(checkpoint['margin'])



    epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0

    idx_to_class = checkpoint['idx_to_class'] if 'idx_to_class' in checkpoint else {} 

    return model, optimizer, margin, epoch, idx_to_class



class EmbeddigsNet(nn.Module):

    def __init__(self, model_name, embeddings_size):

        super(EmbeddigsNet, self).__init__()

        self.effnet = EfficientNet.from_name(model_name)

        # Unfreeze model weights

        for param in self.effnet.parameters():

            param.requires_grad = True

        num_ftrs = self.effnet._fc.in_features

        self.effnet._fc = nn.Linear(num_ftrs, embeddings_size)

    def forward(self, x):

        x = self.effnet(x)

        return x





def get_model(model_name, embeddings_size):

    model = EmbeddigsNet(model_name, embeddings_size)

    return model





def calculate_top_n(sim_matrix,best_top_n_vals,

                               best_top_n_idxs,

                               curr_zero_idx=0,

                               n=10):

    n_rows, n_cols = sim_matrix.shape

    total_matrix_vals = sim_matrix

    total_matrix_idxs = np.tile(np.arange(n_rows).reshape(n_rows,1), (1,n_cols)).astype(int) + curr_zero_idx

    if curr_zero_idx>0:

        total_matrix_vals = np.vstack((total_matrix_vals, best_top_n_vals))

        total_matrix_idxs = np.vstack((total_matrix_idxs, best_top_n_idxs))

    res = np.argpartition(total_matrix_vals, -n, axis=0)[-n:]

    res_vals = np.take_along_axis(total_matrix_vals, res, axis=0)

    res_idxs = np.take_along_axis(total_matrix_idxs, res, axis=0)



    del res, total_matrix_idxs, total_matrix_vals

    return res_vals, res_idxs





def cosine_similarity_chunks(X, Y, n_chunks=5, top_n=5):

    ch_sz = X.shape[0]//n_chunks

    best_top_n_vals = None

    best_top_n_idxs = None

    num_chunks = 2 if DEBUG else n_chunks

    for i in tqdm(range(num_chunks)):

        chunk = X[i*ch_sz:,:] if i==n_chunks-1 else X[i*ch_sz:(i+1)*ch_sz,:]

        cosine_sim_matrix_i = cosine_similarity(chunk, Y)

        best_top_n_vals, best_top_n_idxs = calculate_top_n(cosine_sim_matrix_i,

                                                           best_top_n_vals,

                                                            best_top_n_idxs,

                                                            curr_zero_idx=(i*ch_sz),

                                                            n=top_n)

    print(f'Best similar vals shape {best_top_n_vals.shape}')

    print(f'Best similar idxs shape {best_top_n_idxs.shape}')

    return best_top_n_vals, best_top_n_idxs





def get_embeddings(model, test_dataloader, device, l2_norm=True):

    generated_embeddings = {}

    model.eval()



    tqdm_test = tqdm(test_loader, total=int(len(test_loader)))

   

    with torch.no_grad():

        for batch_index, [data, images_ids] in enumerate(tqdm_test):

            if DEBUG and batch_index>10:

                break

            

            logits = None

            for dat in data:

                dat = dat.to(device)

                logits_i = model(dat)

                logits_i = logits_i.cpu().numpy()

                if logits is None:

                    logits = logits_i/len(data)

                else:

                    logits += logits_i/len(data)

                    

            image_ids = images_ids[0]

            for logits_i, image_id in zip(logits, image_ids):

               generated_embeddings[image_id] = logits_i / np.linalg.norm(logits_i) if l2_norm else logits_i

                

                

    return generated_embeddings



def get_max_conf_pred(curr_n_pred, curr_n_conf):

    dict_conf_list = {}

    dict_conf_sum = {}



    for p, c in zip(curr_n_pred, curr_n_conf):

        if p in dict_conf_list:

            dict_conf_list[p].append(c)

        else:

            dict_conf_list[p] = [c]



    dict_conf_sum = { k : sum(v) for k, v in dict_conf_list.items()}

    pred_i = max(dict_conf_sum, key=dict_conf_sum.get)

    conf_i = sum(dict_conf_list[pred_i])

    return pred_i, conf_i



def generate_submission_df(ids, predictions, confidences, th=0):

    ids, predictions, confidences = list(ids), list(predictions), list(confidences)

    pred_conf_strings = [f'{int(p)} {c}' if c>=th else '' for p,c in zip(predictions, confidences)]

    df = pd.DataFrame(list(zip(ids, pred_conf_strings)), columns =['id', 'landmarks'])

    return df

transform_no_aug = A.Compose([

            A.Resize(image_size, image_size),

            A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

            ToTensorV2()

        ])



transform_resize_d = A.Compose([

            A.Resize(int(0.75*image_size), int(0.75*image_size)),

            A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

            ToTensorV2()

        ])



transform_resize_u = A.Compose([

            A.Resize(int(1.25*image_size), int(1.25*image_size)),

            A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

            ToTensorV2()

        ])



transform_resize_u2 = A.Compose([

            A.Resize(int(2*image_size), int(2*image_size)),

            A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

            ToTensorV2()

        ])



transform_flip_lr = A.Compose([

            A.Resize(image_size, image_size),

            A.HorizontalFlip(always_apply=True, p=1),

            A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

            ToTensorV2()

        ])
# model_names = ['efficientnet-b3']

# weights_paths = ['/kaggle/input/efficientnetb3best/best_512.pt']



model_names = ['efficientnet-b3']

weights_paths = ['/kaggle/input/e3-weights/last.pt']



# model_names = ['efficientnet-b3',

#                'efficientnet-b7']

# weights_paths = ['/kaggle/input/efficientnetb3best/best_512.pt',

#                  '/kaggle/input/efficientnetb7best/best_epoch_45_wce.pt']



# model_names = ['efficientnet-b3',

#                'efficientnet-b7']

# weights_paths = ['/kaggle/input/efficientnetb3best/best_512.pt',

#                  '/kaggle/input/efficientnetb7best/best_epoch_45_wce.pt']

# model_names = ['efficientnet-b3']

# weights_paths = ['/kaggle/input/efficientnetb3best/best_512.pt']



# model_names = ['efficientnet-b3',

#                'efficientnet-b5',

#                'efficientnet-b7']

# weights_paths = ['/kaggle/input/efficientnetb3best/best_512.pt',

#                  '/kaggle/input/efficientnetb5best/last.pt',

#                  '/kaggle/input/efficientnetb7best/best_epoch_45_wce.pt']



# model_names = ['efficientnet-b5',

#                'efficientnet-b7']

# weights_paths = ['/kaggle/input/efficientnetb5best/last.pt',

#                  '/kaggle/input/efficientnetb7best/best_epoch_45_wce.pt']

# model_names = ['efficientnet-b5']

# weights_paths = ['/kaggle/input/efficientnetb5best/last.pt']

# transforms_all = [transform_no_aug]

transforms_all = [transform_no_aug, transform_resize_u]

# transforms_all = [transform_no_aug, transform_resize_u, transform_resize_d]



max_n_samples = 10000000 if DEBUG else 1000000
df_train = pd.read_csv(train_csv_path)

n_files = len(df_train)

print(f'Number of training samples {n_files}')
if n_files>max_n_samples:

    copyfile('/kaggle/input/landmark-recognition-2020/sample_submission.csv', 'submission.csv')

else:

    models = []

    for model_name, weights_path in zip(model_names, weights_paths):

        print(model_name)

        model = get_model(model_name, embeddings_size=embeddings_size)

        model = model.to(device)

        model, _, _, _, _ = load_ckp(weights_path, model, remove_module=True)

        models.append(model)

        

    embeddings_all = []

    

    train_embeddings_final = {}

    test_embeddings_final = {}

    

    embeddings_transform = []

    if train_encodings_from_file:

        data_dirs = [test_data_dir]

        df = pd.read_csv('/kaggle/input/landmark-recognition-2020/train.csv')

        encd = np.load('/kaggle/input/encodings/encodings.npy')

        df_nms = pd.read_csv('/kaggle/input/encodings/names.csv')

        cl_nms = df_nms['id'].tolist()

        tr_nms = df['id'].tolist()

        enc_dict = {}

        

        for v_idx , (cl_nm, enc) in tqdm(enumerate(zip(cl_nms, encd)), total=len(cl_nms)):

            if DEBUG:

                if v_idx>100:

                    break

            if cl_nm in tr_nms:

                enc_dict[cl_nm] = enc

        embeddings_transform.append(enc_dict)

        del tr_nms, cl_nms, df_nms, encd, df, enc_dict

    else:

        data_dirs = [train_data_dir, test_data_dir]

    

    for data_dir in data_dirs:

        test_loader, test_data = get_dataloader(data_dir=data_dir,  

                                                transform=transforms_all , 

                                                batch_size=batch_size,

                                                num_workers=num_workers,

                                                collate_fn=my_collate,

                                                drop_last=False)

        embeddings = {}

        for model in models:

            embeddings_i = get_embeddings(model, test_loader, device)

            if not embeddings:

                embeddings = embeddings_i

            else:

                for k, v in embeddings_i.items():

                    embeddings[k] = np.concatenate([embeddings[k], v])

        embeddings_transform.append(embeddings)



    train_embeddings, test_embeddings = embeddings_transform



    df_train['encodings'] = df_train['id'].map(train_embeddings)

    df_train = df_train.dropna()

    df_test = pd.DataFrame(list(test_embeddings.items()), columns=['id', 'encodings'])



    encodings_train = np.array(df_train['encodings'].tolist(), dtype=np.float32)

    encodings_valid = np.array(df_test['encodings'].tolist(), dtype=np.float32)

    print(encodings_train.shape)
if calculate_local:

    # Dataset parameters:

    INPUT_DIR = os.path.join('..', 'input')



    DATASET_DIR = os.path.join(INPUT_DIR, 'landmark-recognition-2020')

    TEST_IMAGE_DIR = os.path.join(DATASET_DIR, 'test')

    TRAIN_IMAGE_DIR = os.path.join(DATASET_DIR, 'train')

    TRAIN_LABELMAP_PATH = os.path.join(DATASET_DIR, 'train.csv')



    # RANSAC parameters:

    MAX_INLIER_SCORE = 35

    MAX_REPROJECTION_ERROR = 6.0

    MAX_RANSAC_ITERATIONS = 5_000_000

    HOMOGRAPHY_CONFIDENCE = 0.99



    # DELG model:

    SAVED_MODEL_DIR = '../input/delg-saved-models/local_and_global'

    DELG_MODEL = tf.saved_model.load(SAVED_MODEL_DIR)

    DELG_IMAGE_SCALES_TENSOR = tf.convert_to_tensor([0.70710677, 1.0, 1.4142135])

    DELG_SCORE_THRESHOLD_TENSOR = tf.constant(175.)

    DELG_INPUT_TENSOR_NAMES = ['input_image:0', 'input_scales:0', 'input_abs_thres:0']





    # Local feature extraction:

    LOCAL_FEATURE_NUM_TENSOR = tf.constant(1000)

    LOCAL_FEATURE_EXTRACTION_FN = DELG_MODEL.prune(DELG_INPUT_TENSOR_NAMES + ['input_max_feature_num:0'],['boxes:0', 'features:0'])



    def get_image_path(subset, name):

        return os.path.join(DATASET_DIR, subset, name[0], name[1], name[2],f'{name}.jpg')





    def load_image_tensor(image_path):

        return tf.convert_to_tensor(np.array(PIL.Image.open(image_path).convert('RGB')))





    def extract_local_features(image_path):

      """Extracts local features for the given `image_path`."""



      image_tensor = load_image_tensor(image_path)



      features = LOCAL_FEATURE_EXTRACTION_FN(image_tensor, DELG_IMAGE_SCALES_TENSOR,

                                             DELG_SCORE_THRESHOLD_TENSOR,

                                             LOCAL_FEATURE_NUM_TENSOR)



      # Shape: (N, 2)

      keypoints = tf.divide(

          tf.add(

              tf.gather(features[0], [0, 1], axis=1),

              tf.gather(features[0], [2, 3], axis=1)), 2.0).numpy()



      # Shape: (N, 128)

      descriptors = tf.nn.l2_normalize(features[1], axis=1, name='l2_normalization').numpy()

      return keypoints, descriptors

    

    def get_total_score(num_inliers, global_score):

      local_score = min(num_inliers, MAX_INLIER_SCORE) / MAX_INLIER_SCORE

      return local_score + global_score



    def get_putative_matching_keypoints(test_keypoints,

                                        test_descriptors,

                                        train_keypoints,

                                        train_descriptors,

                                        max_distance=0.9):

      """Finds matches from `test_descriptors` to KD-tree of `train_descriptors`."""



      train_descriptor_tree = spatial.cKDTree(train_descriptors)

      _, matches = train_descriptor_tree.query(test_descriptors, distance_upper_bound=max_distance)



      test_kp_count = test_keypoints.shape[0]

      train_kp_count = train_keypoints.shape[0]



      test_matching_keypoints = np.array([test_keypoints[i,]

                                          for i in range(test_kp_count)

                                          if matches[i] != train_kp_count])

      train_matching_keypoints = np.array([train_keypoints[matches[i],]

                                           for i in range(test_kp_count)

                                           if matches[i] != train_kp_count])



      return test_matching_keypoints, train_matching_keypoints





    def get_num_inliers(test_keypoints, test_descriptors, train_keypoints, train_descriptors):

      """Returns the number of RANSAC inliers."""



      test_match_kp, train_match_kp = get_putative_matching_keypoints(

          test_keypoints, test_descriptors, train_keypoints, train_descriptors)



      if test_match_kp.shape[0] <= 4:  # Min keypoints supported by `pydegensac.findHomography()`

        return 0



      try:

        _, mask = pydegensac.findHomography(test_match_kp, train_match_kp,

                                            MAX_REPROJECTION_ERROR,

                                            HOMOGRAPHY_CONFIDENCE,

                                            MAX_RANSAC_ITERATIONS)

      except np.linalg.LinAlgError:  # When det(H)=0, can't invert matrix.

        return 0



      return int(copy.deepcopy(mask).astype(np.float32).sum())
if n_files<=max_n_samples:

    best_top_n_vals, best_top_n_idxs = cosine_similarity_chunks(encodings_train, encodings_valid, n_chunks=n_chunks, top_n=n_top_global)

    p = np.argsort(-best_top_n_vals, axis=0)

    best_top_n_vals = np.take_along_axis(best_top_n_vals, p, axis=0)

    best_top_n_idxs = np.take_along_axis(best_top_n_idxs, p, axis=0)

    predictions_n = np.zeros_like(best_top_n_idxs)

    confidences_n = best_top_n_vals

    filenames_n = []

    

    for row_idx, row in enumerate(best_top_n_idxs):

        df_prediction = df_train.iloc[row, :]

        pred = np.array(df_prediction['landmark_id'].tolist())

        fnames = np.array(df_prediction['id'].tolist(), dtype=str)

        predictions_n[row_idx, :] = pred

        filenames_n.append(fnames)

    filenames_n = np.array(filenames_n)

    test_images_names = df_test['id'].tolist()

    

    predictions_final = np.zeros(confidences_n.shape[1])

    confidences_final = np.zeros(confidences_n.shape[1])

    

    # Extract local features for test images

    if calculate_local:

        test_kp_ds = []

        for test_img_idx, test_img_name in enumerate(tqdm(test_images_names)):

            test_img_path = get_image_path('test', test_img_name)

            test_keypoints, test_descriptors = extract_local_features(test_img_path)

            curr_n_conf = []

            if DEBUG and test_img_idx>20:

                break

            for train_img_idx, train_img_name in enumerate(filenames_n[:n_top_local, test_img_idx]):

#                 start_time = time.time()

                train_img_path = get_image_path('train', train_img_name)

                train_keypoints, train_descriptors = extract_local_features(train_img_path)

                n_matching = get_num_inliers(test_keypoints, test_descriptors, train_keypoints, train_descriptors)

                curr_n_conf.append(n_matching)

             

            curr_n_conf = np.array(curr_n_conf)

            curr_n_pred = predictions_n[:, test_img_idx]

            

            pred_i, conf_i = get_max_conf_pred(curr_n_pred, curr_n_conf)

            if reranking:

                test_kp_ds.append((test_img_name, test_keypoints, test_descriptors, pred_i, conf_i))

            predictions_final[test_img_idx] = pred_i

            confidences_final[test_img_idx] = conf_i

#             print(f'predisctions size without ranking {predictions_final.shape}')

#                 print(f'One image in { time.time() - start_time} s')

        if reranking:

            print('Start reranking')

            reranking_eps = 0.01

            N_rerank_check = min(N_rerank_check, len(test_kp_ds))

            test_kp_ds.sort(key=lambda data: data[4], reverse=True)

            test_kp_ds_to_rerank = test_kp_ds[:N_rerank_check]

            reranked_vals = []

            reranked_vals_names = []

            

            for r_i in tqdm(range(N_rerank_select)):

                im_i, tk_i, td_i, pr_i, c_i = test_kp_ds_to_rerank[r_i]

                if im_i in reranked_vals_names:

                    continue

                curr_rerank = []

                reranked_vals.append((im_i, pr_i, c_i))

                reranked_vals_names.append(im_i)



                for r_j in range(r_i+1, N_rerank_check):

                    im_j, tk_j, td_j, pr_j, c_j = test_kp_ds_to_rerank[r_j]

                    if pr_i != pr_j:

#                         pr_j = pr_i

                        continue

                    if im_j in reranked_vals_names:

                        continue

                    n_matching = get_num_inliers(tk_i, td_i, tk_j, td_j)

                    if n_matching < reranking_th:

                        continue

                    curr_rerank.append((im_j, pr_j, n_matching))



                curr_rerank.sort(key=lambda data: data[2], reverse=True)

                for j_j, curr_val in enumerate(curr_rerank):

                    im_j_j, pr_j_j, c_j_j = curr_val

                    new_score = c_i - (reranking_eps*j_j)

                    reranked_vals_names.append(im_j_j)

                    reranked_vals.append((im_j_j, pr_j_j, new_score))

            for t_kp_ds in test_kp_ds:

                im_i, tk_i, td_i, pr_i, c_i = t_kp_ds

                if im_i in reranked_vals_names:

                    continue

                reranked_vals.append((im_i, pr_i, c_i))

            test_images_names, predictions_final, confidences_final = zip(*reranked_vals)

            print('Reranking completed')

    else:

        for j in range(confidences_n.shape[1]):

            curr_n_pred = predictions_n[:, j]

            curr_n_conf = confidences_n[:, j]

            pred_i, conf_i = get_max_conf_pred(curr_n_pred, curr_n_conf)

            predictions_final[j] = pred_i

            confidences_final[j] = conf_i

        

    sub_csv = generate_submission_df(test_images_names, predictions_final, confidences_final)

    sub_csv.to_csv('submission.csv', index=False)

    

    print(len(sub_csv))

    sub_csv.head()