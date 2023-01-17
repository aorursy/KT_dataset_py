import os, glob

import random

from sklearn.model_selection import train_test_split

import cv2

import numpy as np

import pandas as pd

import multiprocessing

from copy import deepcopy

from sklearn.metrics import precision_recall_curve, auc

import tensorflow.keras as keras

from tensorflow.keras.models import load_model

from tensorflow.keras.utils import Sequence

from tensorflow.keras import backend as K

import matplotlib.pyplot as plt

from IPython.display import Image

from tqdm import tqdm_notebook as tqdm

from tensorflow.python.framework import ops

from numpy.random import seed

seed(10)

from tensorflow import set_random_seed

import tensorflow as tf

set_random_seed(10)

import gc

%matplotlib inline
test_imgs_folder = '../input/understanding_cloud_organization/test_images/'

train_imgs_folder = '../input/understanding_cloud_organization/train_images/'

image_width, image_height = 224, 224

num_cores = multiprocessing.cpu_count()

batch_size = 64

model_class_names =  ['Fish', 'Flower', 'Sugar', 'Gravel']
train_df_orig = pd.read_csv('../input/understanding_cloud_organization/train.csv')

train_df = pd.read_csv('../input/understanding_cloud_organization/train.csv')
train_df = train_df[~train_df['EncodedPixels'].isnull()]

train_df['Image'] = train_df['Image_Label'].map(lambda x: x.split('_')[0])

train_df['Class'] = train_df['Image_Label'].map(lambda x: x.split('_')[1])

classes = train_df['Class'].unique()

train_df = train_df.groupby('Image')['Class'].agg(set).reset_index()

for class_name in classes:

    train_df[class_name] = train_df['Class'].map(lambda x: 1 if class_name in x else 0)
# dictionary for fast access to ohe vectors

img_2_ohe_vector = {img:vec for img, vec in zip(train_df['Image'], train_df.iloc[:, 2:].values)}
train_imgs, val_imgs = train_test_split(train_df['Image'].values, 

                                        test_size=0.2, 

                                        stratify=train_df['Class'].map(lambda x: str(sorted(list(x)))), # sorting present classes in lexicographical order, just to be sure

                                        random_state=10)
val_imgs_np = np.empty((len(val_imgs), image_height, image_width, 3))

for img_i, img_name in enumerate(tqdm(val_imgs)):

    img_path = os.path.join(train_imgs_folder, img_name)

    val_imgs_np[img_i, :, :, :] = cv2.resize(cv2.imread(img_path), (image_height, image_width)).astype(np.float32)/255.0
# helper functions

# credits: https://www.kaggle.com/artgor/segmentation-in-pytorch-using-convenient-tools?scriptVersionId=20202006

def rle_decode(mask_rle: str = '', shape: tuple = (1400, 2100)):

    '''

    Decode rle encoded mask.

    

    :param mask_rle: run-length as string formatted (start length)

    :param shape: (height, width) of array to return 

    Returns numpy array, 1 - mask, 0 - background

    '''

    s = mask_rle.split()

    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    starts -= 1

    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):

        img[lo:hi] = 1

    return img.reshape(shape, order='F')



def mask2rle(img):

    '''

    Convert mask to rle.

    img: numpy array, 1 - mask, 0 - background

    Returns run length as string formated

    '''

    pixels= img.T.flatten()

    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)





def make_mask(df, image_label, shape: tuple = (1400, 2100)):

    """

    Create mask based on df, image name and shape.

    """

    df = df.set_index('Image_Label')

    encoded_mask = df.loc[image_label, 'EncodedPixels']

    mask = np.zeros((shape[0], shape[1]), dtype=np.float32)

    if encoded_mask is not np.nan:

        mask = rle_decode(encoded_mask)

            

    return cv2.resize(mask, (image_height, image_width))
val_masks_np = np.empty((len(model_class_names), len(val_imgs), image_height, image_width))

for class_i, class_name in enumerate(tqdm(model_class_names)):

    for img_i, img_name in enumerate(val_imgs):

        mask = make_mask(train_df_orig, img_name + '_' + class_name)

        val_masks_np[class_i][img_i] = mask
model = load_model('../input/clouds-classifier-files/classifier_densenet169_epoch_21_val_pr_auc_0.8365921057512743.h5')
# gradcam functions source/inspiration: https://github.com/eclique/keras-gradcam/blob/master/grad_cam.py

layer_name='conv5_block16_concat'



# for batch gradcam

gradient_fns = []

for class_i in range(len(model_class_names)):

    class_predictions = tf.slice(model.output, [0, class_i], [-1, 1])

    conv_layer_output = model.get_layer(layer_name).output

    grads = K.gradients(class_predictions, conv_layer_output)[0]

    gradient_fns.append(K.function([model.input, K.learning_phase()], [conv_layer_output, grads]))



def grad_cam_batch(images, class_i):

    """GradCAM method for visualizing input saliency.

    Same as grad_cam but processes multiple images in one run."""

    conv_output, grads_val = gradient_fns[class_i]([images, 0])    

    weights = np.mean(grads_val, axis=(1, 2))

    cams = np.einsum('ijkl,il->ijk', conv_output, weights)

    

    # Process CAMs

    new_cams = np.empty((images.shape[0], image_height, image_width))

    for i in range(images.shape[0]):

        cam_i = cams[i] - cams[i].mean()

        cam_i = (cam_i + 1e-10) / (np.linalg.norm(cam_i, 2) + 1e-10)

        new_cams[i] = cv2.resize(cam_i, (image_height, image_width), cv2.INTER_LINEAR)

        new_cams[i] = np.maximum(new_cams[i], 0)

        new_cams[i] = new_cams[i] / new_cams[i].max()    

    return new_cams
# for a given class probability finds corresponding precision

tabular_precisions =  np.concatenate((np.arange(0, 0.35, 0.1), np.arange(0.35, 0.6, 0.05), np.arange(0.6, 1, 0.025)))

def get_tabular_precisions(class_i, class_probs, class_probability_2_precision):

    prob_2_precision = class_probability_2_precision[class_i]

    return np.array([prob_2_precision[min(prob_2_precision.keys(), 

                                          key=lambda x: abs(x - class_prob))] 

                     for class_prob in class_probs])



# credits https://www.kaggle.com/artgor/segmentation-in-pytorch-using-convenient-tools?scriptVersionId=20202006

def remove_small_mask_components(masks, min_size):

    cleaned_masks = np.zeros_like(masks, np.float32)

    for mask_i in range(masks.shape[0]):

        mask = masks[mask_i]

        num_component, component = cv2.connectedComponents(mask.astype(np.uint8))

        for c in range(1, num_component):

            p = (component == c)

            if p.sum() > min_size:

                cleaned_masks[mask_i][p] = 1

    return cleaned_masks

    

def generate_gradcam_masks(precision_2_binarization_threshold,

                        min_size,

                           imgs_np,

                        class_probability_2_precision,

                        model=model, model_class_names=model_class_names

                       ):

    gradcam_masks_np = np.empty((len(model_class_names),) + imgs_np.shape[:3], np.float32)

    zero_mask = np.zeros((image_height, image_width), np.float32)

    print(f'Generating masks using Grad-CAM')

    num_batches = imgs_np.shape[0]//batch_size + 1

    for batch_i in tqdm(range(num_batches)):

        imgs_batch = imgs_np[batch_i*batch_size: (batch_i + 1)*batch_size]

        predictions = model.predict(imgs_batch)

        for class_i, class_name in enumerate(model_class_names):

            gradcams_batch = grad_cam_batch(imgs_batch, class_i)

            class_probs = predictions[:, class_i]            

            class_precisions = get_tabular_precisions(class_i, class_probs, class_probability_2_precision)

            binarization_thresholds = np.array([precision_2_binarization_threshold[class_precision]

                                                for class_precision in class_precisions])

            # to perform elementwise binarization on all images at once

            binarization_thresholds_batch = np.repeat(binarization_thresholds, 

                                                      image_width*image_height).reshape(imgs_batch.shape[0], image_height, image_width)

            binarized_gradcams_batch = gradcams_batch.copy()

            binarized_gradcams_batch = np.where(binarized_gradcams_batch > binarization_thresholds_batch, 1.0, 0.0)

            binarized_gradcams_batch[imgs_batch[:,:,:,0]==0] = 0

            binarized_gradcams_batch = remove_small_mask_components(binarized_gradcams_batch, min_size)

            gradcam_masks_np[class_i][batch_i*batch_size: (batch_i + 1)*batch_size] = binarized_gradcams_batch

    return gradcam_masks_np



def visualize_img_gradcam_mask(img_idx, imgs_np, gradcam_masks_np, masks_val_np, names=None):

    img = imgs_np[img_idx]

    fig, axes = plt.subplots(2, 2, figsize=(10,10))

    for class_i, class_name in enumerate(model_class_names):

        ax = axes[int(class_i>1), class_i%2]

        ax.set_title(f'GradCAM: {class_name}')

        ax.axis('off')

        ax.imshow(img)

        ax.imshow(gradcam_masks_np[class_i][img_idx], cmap='jet', alpha=0.7)

        ax.imshow(masks_val_np[class_i][img_idx], alpha=0.35)

    if not names is None:

        plt.suptitle(f'Image {names[img_idx]}', fontsize=15)
class DataGenenerator(Sequence):

    def __init__(self, images_list=None, folder_imgs=train_imgs_folder, 

                 batch_size=32, shuffle=True, augmentation=None,

                 resized_height=224, resized_width=224, num_channels=3):

        self.batch_size = batch_size

        self.shuffle = shuffle

        self.augmentation = augmentation

        if images_list is None:

            self.images_list = os.listdir(folder_imgs)

        else:

            self.images_list = deepcopy(images_list)

        self.folder_imgs = folder_imgs

        self.len = len(self.images_list) // self.batch_size

        self.resized_height = resized_height

        self.resized_width = resized_width

        self.num_channels = num_channels

        self.num_classes = 4

        self.is_test = not 'train' in folder_imgs

        if not shuffle and not self.is_test:

            self.labels = [img_2_ohe_vector[img] for img in self.images_list[:self.len*self.batch_size]]



    def __len__(self):

        return self.len

    

    def on_epoch_start(self):

        if self.shuffle:

            random.shuffle(self.images_list)



    def __getitem__(self, idx):

        current_batch = self.images_list[idx * self.batch_size: (idx + 1) * self.batch_size]

        X = np.empty((self.batch_size, self.resized_height, self.resized_width, self.num_channels))

        y = np.empty((self.batch_size, self.num_classes))



        for i, image_name in enumerate(current_batch):

            path = os.path.join(self.folder_imgs, image_name)

            img = cv2.resize(cv2.imread(path), (self.resized_height, self.resized_width)).astype(np.float32)

            if not self.augmentation is None:

                augmented = self.augmentation(image=img)

                img = augmented['image']

            X[i, :, :, :] = img/255.0

            if not self.is_test:

                y[i, :] = img_2_ohe_vector[image_name]

        return X, y



    def get_labels(self):

        if self.shuffle:

            images_current = self.images_list[:self.len*self.batch_size]

            labels = [img_2_ohe_vector[img] for img in images_current]

        else:

            labels = self.labels

        return np.array(labels)
data_generator_val = DataGenenerator(val_imgs, shuffle=False)

y_pred = model.predict_generator(data_generator_val, workers=num_cores)

y_true = data_generator_val.get_labels()



def get_probability_for_precision_threshold(y_true, y_pred, class_i, precision_threshold):

    precision, recall, thresholds = precision_recall_curve(y_true[:, class_i], y_pred[:, class_i])

    # consice, even though unnecessary passing through all the values

    probability_threshold = [thres for prec, thres in zip(precision, thresholds) if prec >= precision_threshold][0]

    return probability_threshold



class_probability_2_precision = [dict() for _ in range(len(model_class_names))]

for class_i in tqdm(range(len(model_class_names))):

    for precition_thres in tabular_precisions:

        class_prob = get_probability_for_precision_threshold(y_true, y_pred, class_i, precition_thres)

        class_probability_2_precision[class_i][class_prob] = precition_thres
precision_2_binarization_threshold = {prec: 1.0 if prec < 0.85 else 0.2 if prec < 0.9 else 0.01 for prec in tabular_precisions}

min_size = 2000

gradcam_masks_np = generate_gradcam_masks(precision_2_binarization_threshold, min_size, val_imgs_np, class_probability_2_precision)
for img_i in random.sample(list(range(len(val_imgs))), 15):

    visualize_img_gradcam_mask(img_i, val_imgs_np, gradcam_masks_np, val_masks_np, names=val_imgs)
# credits: https://www.kaggle.com/artgor/segmentation-in-pytorch-using-convenient-tools?scriptVersionId=20202006

def dice(img1, img2):

    img1 = np.asarray(img1).astype(np.bool)

    img2 = np.asarray(img2).astype(np.bool)



    intersection = np.logical_and(img1, img2)



    return 2. * intersection.sum() / (img1.sum() + img2.sum())
def estimate_dice(gradcam_masks_np, val_masks_np=val_masks_np):

    assert(gradcam_masks_np.shape==val_masks_np.shape)

    scores_all = []

    scores_per_class = [[] for _ in range(len(model_class_names))]

    for class_i, class_name in enumerate(model_class_names):        

        for img_idx in range(gradcam_masks_np.shape[1]):

            if gradcam_masks_np[class_i][img_idx].sum() != 0:

                score = dice(gradcam_masks_np[class_i][img_idx], val_masks_np[class_i][img_idx])

                scores_all.append(score)

                scores_per_class[class_i].append(score)

            else:

                scores_all.append(val_masks_np[class_i][img_idx].sum() == 0)

                scores_per_class[class_i].append(val_masks_np[class_i][img_idx].sum() == 0)

    return np.mean(scores_all), [np.mean(scores) for scores in scores_per_class]  



mean_dice, mean_dice_per_cls = estimate_dice(gradcam_masks_np)

print(f"""Val Mean Dice overall: {mean_dice: .3f}



{chr(10).join([chr(9) + 'Val Mean Dice for class ' + 

class_name + ' is {:.3f}'.format(mean_dice_per_cls[class_i]) for class_i, class_name in enumerate(model_class_names)])}""")
del val_imgs_np

gc.collect()



test_imgs = os.listdir(test_imgs_folder)

test_imgs_np = np.empty((len(test_imgs), image_height, image_width, 3))

for img_i, img_name in enumerate(tqdm(test_imgs)):

    img_path = os.path.join(test_imgs_folder, img_name)

    test_imgs_np[img_i, :, :, :] = cv2.resize(cv2.imread(img_path), (image_height, image_width)).astype(np.float32)/255.0
gradcam_masks_np = generate_gradcam_masks(precision_2_binarization_threshold, 2000, test_imgs_np, class_probability_2_precision)
img_label_list = []

enc_pixels_list = []

for test_img_i, test_img in enumerate(tqdm(test_imgs)):

    for class_i, class_name in enumerate(model_class_names):

        img_label_list.append(f'{test_img}_{class_name}')

        mask = gradcam_masks_np[class_i][test_img_i]

        if mask.sum() == 0:

            enc_pixels_list.append(np.nan)

        else:

            mask = cv2.resize(mask, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)

            mask = np.where(mask > 0.5, 1.0, 0.0)

            enc_pixels_list.append(mask2rle(mask))

submission_df = pd.DataFrame({'Image_Label': img_label_list, 'EncodedPixels': enc_pixels_list})

submission_df.to_csv('sub_gradcam.csv', index=None)