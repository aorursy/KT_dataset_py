!pip install efficientnet_pytorch >> /dev/null
import os, random, re, math, time

random.seed(a=42)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np

import pandas as pd

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score



from glob import glob

from tqdm import tqdm



import tensorflow as tf

import tensorflow.keras.backend as K



from tensorflow.keras.mixed_precision import experimental as mixed_precision



gpus = tf.config.experimental.list_physical_devices('GPU')

try:

    for gpu in gpus:

        tf.config.experimental.set_memory_growth(gpu, True)

except RuntimeError as e:

    print(e)



CFG = dict(

    batch_size        =  32,

    read_size         = 256, 

    crop_size         = 235, 

    net_size          = 224, 

    

    LR                =   1e-4,

    epochs            =  30,

    

    rot               = 180.0,

    shr               =   2.0,

    hzoom             =   8.0,

    wzoom             =   8.0,

    hshift            =   8.0,

    wshift            =   8.0,

    tta_steps = 15,

    

    es_patience = 4,

    

)





GCS_PATH = f'../input//melanoma-{CFG["read_size"]}x{CFG["read_size"]}'

files_train = np.sort(np.array(tf.io.gfile.glob(GCS_PATH + '/train*.tfrec')))

files_test  = np.sort(np.array(tf.io.gfile.glob(GCS_PATH + '/test*.tfrec')))



malig_files = sorted(glob(f'../input/malignant-v2-{CFG["read_size"]}x{CFG["read_size"]}/*.tfrec'))

malig_files = malig_files[15:]



files_train = np.concatenate([files_train, malig_files])

    
def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):

    # returns 3x3 transformmatrix which transforms indicies

        

    # CONVERT DEGREES TO RADIANS

    rotation = math.pi * rotation / 180.

    shear    = math.pi * shear    / 180.



    def get_3x3_mat(lst):

        return tf.reshape(tf.concat([lst],axis=0), [3,3])

    

    # ROTATION MATRIX

    c1   = tf.math.cos(rotation)

    s1   = tf.math.sin(rotation)

    one  = tf.constant([1],dtype='float32')

    zero = tf.constant([0],dtype='float32')

    

    rotation_matrix = get_3x3_mat([c1,   s1,   zero, 

                                   -s1,  c1,   zero, 

                                   zero, zero, one])    

    # SHEAR MATRIX

    c2 = tf.math.cos(shear)

    s2 = tf.math.sin(shear)    

    

    shear_matrix = get_3x3_mat([one,  s2,   zero, 

                                zero, c2,   zero, 

                                zero, zero, one])        

    # ZOOM MATRIX

    zoom_matrix = get_3x3_mat([one/height_zoom, zero,           zero, 

                               zero,            one/width_zoom, zero, 

                               zero,            zero,           one])    

    # SHIFT MATRIX

    shift_matrix = get_3x3_mat([one,  zero, height_shift, 

                                zero, one,  width_shift, 

                                zero, zero, one])

    

    return K.dot(K.dot(rotation_matrix, shear_matrix), 

                 K.dot(zoom_matrix,     shift_matrix))





def transform(image, cfg):

    DIM = cfg["read_size"]

    XDIM = DIM%2 #fix for size 331

    

    rot = cfg['rot'] * tf.random.normal([1], dtype='float32')

    shr = cfg['shr'] * tf.random.normal([1], dtype='float32') 

    h_zoom = 1.0 + tf.random.normal([1], dtype='float32') / cfg['hzoom']

    w_zoom = 1.0 + tf.random.normal([1], dtype='float32') / cfg['wzoom']

    h_shift = cfg['hshift'] * tf.random.normal([1], dtype='float32') 

    w_shift = cfg['wshift'] * tf.random.normal([1], dtype='float32') 



    # GET TRANSFORMATION MATRIX

    m = get_mat(rot,shr,h_zoom,w_zoom,h_shift,w_shift) 



    # LIST DESTINATION PIXEL INDICES

    x   = tf.repeat(tf.range(DIM//2, -DIM//2,-1), DIM)

    y   = tf.tile(tf.range(-DIM//2, DIM//2), [DIM])

    z   = tf.ones([DIM*DIM], dtype='int32')

    idx = tf.stack( [x,y,z] )

    

    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS

    idx2 = K.dot(m, tf.cast(idx, dtype='float32'))

    idx2 = K.cast(idx2, dtype='int32')

    idx2 = K.clip(idx2, -DIM//2+XDIM+1, DIM//2)

    

    # FIND ORIGIN PIXEL VALUES           

    idx3 = tf.stack([DIM//2-idx2[0,], DIM//2-1+idx2[1,]])

    d    = tf.gather_nd(image, tf.transpose(idx3))

        

    return tf.reshape(d,[DIM, DIM,3])



def read_labeled_tfrecord(example):

    tfrec_format = {

        'image'                        : tf.io.FixedLenFeature([], tf.string),

        'image_name'                   : tf.io.FixedLenFeature([], tf.string),

        'patient_id'                   : tf.io.FixedLenFeature([], tf.int64),

        'sex'                          : tf.io.FixedLenFeature([], tf.int64),

        'age_approx'                   : tf.io.FixedLenFeature([], tf.int64),

        'anatom_site_general_challenge': tf.io.FixedLenFeature([], tf.int64),

        'diagnosis'                    : tf.io.FixedLenFeature([], tf.int64),

        'target'                       : tf.io.FixedLenFeature([], tf.int64)

    }           

    example = tf.io.parse_single_example(example, tfrec_format)

    return example['image'], example['target'], example['image_name']





def read_unlabeled_tfrecord(example):

    tfrec_format = {

        'image'                        : tf.io.FixedLenFeature([], tf.string),

        'image_name'                   : tf.io.FixedLenFeature([], tf.string),

    }

    example = tf.io.parse_single_example(example, tfrec_format)

    return example['image'], 0, example['image_name']



 

def prepare_image(img, cfg=None, augment=True):    

    img = tf.image.decode_jpeg(img, channels=3)

    img = tf.image.resize(img, [cfg['read_size'], cfg['read_size']])

    img = tf.cast(img, tf.float32) / 255.0

    

    if augment:

        img = transform(img, cfg)

        img = tf.image.random_crop(img, [cfg['crop_size'], cfg['crop_size'], 3])

        img = tf.image.random_flip_left_right(img)

        img = tf.image.random_hue(img, 0.01)

        img = tf.image.random_saturation(img, 0.7, 1.3)

        img = tf.image.random_contrast(img, 0.8, 1.2)

        img = tf.image.random_brightness(img, 0.1)



    else:

        img = tf.image.central_crop(img, cfg['crop_size'] / cfg['read_size'])

                                   

    img = tf.image.resize(img, [cfg['net_size'], cfg['net_size']])

    img = tf.reshape(img, [cfg['net_size'], cfg['net_size'], 3])

    return img



def get_dataset(files, cfg, augment = False, shuffle = False, 

                labeled=True):

    AUTO     = tf.data.experimental.AUTOTUNE

    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO)

    ds = ds.cache()

    if shuffle: 

        ds = ds.shuffle(1024*8)

        opt = tf.data.Options()

        opt.experimental_deterministic = False

        ds = ds.with_options(opt)

        

    if labeled: 

        ds = ds.map(read_labeled_tfrecord, num_parallel_calls=AUTO)

    else:

        ds = ds.map(lambda example: read_unlabeled_tfrecord(example), 

                    num_parallel_calls=AUTO)      

    

    ds = ds.map(lambda img, label, fn: (prepare_image(img, augment=augment, cfg=cfg), 

                                               label, fn), 

                num_parallel_calls=AUTO)

    

    ds = ds.batch(cfg['batch_size'])

    ds = ds.prefetch(AUTO)

    return ds
#pytorch

from efficientnet_pytorch import EfficientNet

import torch



def get_model():

    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=1)

    model.to(device)

    return model



def get_opt_loss_fn(model):

    optimizer = torch.optim.Adam(model.parameters(), lr=CFG["LR"])

    loss_object = torch.nn.BCEWithLogitsLoss().to(device)

    return optimizer, loss_object



def get_train_fn():

    def train_step(images, labels, model, optimizer, loss_object):

        #sacrilegious conversion

        images = torch.from_numpy(images.numpy()).permute(0,3,1,2).to(device)

        labels = torch.from_numpy(labels.numpy()).to(device).float()

        

        if not model.training:

            model.train()

        predictions = model.forward(images)

        loss = loss_object(predictions[:,0], labels)    

        loss.backward()

        optimizer.step()

        optimizer.zero_grad() 

        return loss.cpu().detach()

    

    return train_step





def test_step(images, labels):

    #sacrilegious conversion

    images = torch.from_numpy(images.numpy()).permute(0,3,1,2).to(device)

    labels = torch.from_numpy(labels.numpy()).to(device).float()

    

    if model.training:

        model.eval()

    with torch.no_grad():

        predictions = model.forward(images)

    loss = loss_object(predictions[:,0], labels)

    return loss.cpu().detach()





def pred_step(images):

    #sacrilegious conversion

    images = torch.from_numpy(images.numpy()).permute(0,3,1,2).to(device)

    

    if model.training:

        model.eval()

    with torch.no_grad():

        return torch.sigmoid(model(images)).cpu().numpy()





device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
fold_cv_scores = []

submission_scores = []



folds = KFold(n_splits=5, shuffle = True, random_state = 42)

fold_num = 0

for tr_idx, va_idx in folds.split(files_train):

    print(f"Starting fold: {fold_num}")

    no_imp = 0

    CFG['batch_size'] = 32

    checkpoint_filepath = f"checkpoint_{fold_num}.h5"

    

    files_train_tr = files_train[tr_idx]

    files_train_va = files_train[va_idx]  

    

    ds_train     = get_dataset(files_train_tr, CFG, augment=True, shuffle=True)

    ds_val     = get_dataset(files_train_va, CFG, augment=False, shuffle=False)

    

    model = get_model()

    optimizer, loss_object = get_opt_loss_fn(model)

    train_fn = get_train_fn()

    torch.cuda.empty_cache()

    

    bestLoss = float("inf")

    for e in range(CFG["epochs"]):

        trainLoss = 0

        tk0 = tqdm(ds_train)

        for idx, (x,y,_) in enumerate(tk0):

            loss = train_fn(x, y, model, optimizer, loss_object)

            trainLoss += loss.numpy()

            tk0.set_postfix(loss=trainLoss/(idx+1))

            

        testLoss = 0

        tk0 = tqdm(ds_val)

        for idx, (x,y,_) in enumerate(tk0):

            loss = test_step(x,y)

            testLoss += loss.numpy()

            tk0.set_postfix(loss=testLoss/(idx+1))

            

        testLoss /= idx

        if testLoss < bestLoss:

            no_imp = 0

            bestLoss = testLoss

            torch.save(model.state_dict(), checkpoint_filepath)

        else:

            no_imp += 1

            if no_imp > CFG["es_patience"]:

                print("Early stopping..")

                break

            



    model.load_state_dict(torch.load(checkpoint_filepath))

    

    ##############################################################

    CFG['batch_size'] = 256

    ds_valAug = get_dataset(files_train_va, CFG, augment=True)

    ds_testAug = get_dataset(files_test, CFG, augment=True, labeled=False)

    for t in range(CFG['tta_steps']):

        ####EVAL FOLD

        for idx, (x,y,fn) in enumerate(ds_valAug):

            predictions = pred_step(x)

            for j in range(predictions.shape[0]):

                fold_cv_scores.append([fold_num, 

                                       fn[j].numpy().decode("utf-8"),

                                        predictions[j,0].item(),

                                        y[j].numpy()])

        

        ####INFER ON TEST

        for idx, (x,y,fn) in enumerate(ds_testAug):

            predictions = pred_step(x)

            for j in range(predictions.shape[0]):

                submission_scores.append([fold_num, 

                                       fn[j].numpy().decode("utf-8"),

                                        predictions[j,0].item()])

        

    

    fold_num += 1

df_fold = pd.DataFrame(fold_cv_scores, columns=["Fold", "Filename", "Pred", "Label"])

df_sub = pd.DataFrame(submission_scores, columns=["Fold", "Filename", "Pred"])
df_fold = df_fold.groupby(["Filename"]).mean().reset_index()

print("CV ROCAUC: ")

print(roc_auc_score(df_fold["Label"], df_fold["Pred"]))
df_sub = df_sub.groupby(["Filename"]).mean().reset_index()

df_sub = df_sub[["Filename", "Pred"]]

df_sub.columns=["image_name", "target"]

df_sub.to_csv("submission.csv", index=False)