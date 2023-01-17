import albumentations as A
import cv2
import gc
import numpy as np
import os
import pandas as pd
import random
import sys
from math import floor
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
!pip install '../input/glrec2020/Keras_Applications-1.0.8-py3-none-any.whl'
!pip install '../input/glrec2020/efficientnet-1.1.0-py3-none-any.whl'
import efficientnet.tfkeras as efn
os.chdir("../input/keras-vgg16-places365")
from vgg16_places_365 import VGG16_Places365
os.chdir("/kaggle/working/")
from keras.callbacks import ReduceLROnPlateau
SEED = 4249
np.random.seed(SEED)
tf.random.set_seed(SEED)
TEST_SIZE = 0.1
SIZE = 256
BATCH_SIZE = 64
LR = 0.0001
CHANNELS = 3
SHAPE = (SIZE, SIZE, CHANNELS)
TRAIN_ROOT_PATH = '../input/landmark-recognition-2020/train'
TEST_ROOT_PATH = '../input/landmark-recognition-2020/test'
submission = pd.read_csv('../input/landmark-recognition-2020/sample_submission.csv')
train_data = pd.read_csv('../input/glrec2020/train.csv')
INFERENCE = True
PLACES = True
def read_image(path, im_size, normalize_image = False):
    img = cv2.imread(path, cv2.IMREAD_COLOR)  
    img = cv2.resize(img, (im_size, im_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    if normalize_image:
        img /= 255.0 
    return img
def get_train_transforms():
    return A.Compose([
            A.HorizontalFlip(p = 0.15),
            A.VerticalFlip(p = 0.25),
            A.RandomRotate90(p = 0.3),
            A.Transpose(p = 0.1),
            A.ShiftScaleRotate(shift_limit = 0.05, scale_limit = 0.1, rotate_limit = 25, interpolation = 1, border_mode = 4, p = 0.15)
        ], p = 0.9)

class TrainDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_ids, labels, classes, batch_size = 16, augmentation = False, *args, **kwargs):
        self.image_ids = image_ids
        self.labels = labels
        self.classes = classes
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.indices = range(len(self.image_ids))
        self.indices = np.arange(len(self.image_ids))

    def __len__(self):
        return int(floor(len(self.image_ids) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        if self.augmentation:
            augmentor = get_train_transforms()
        X = np.empty((self.batch_size, *(SIZE, SIZE, CHANNELS)))
        Y = np.empty((self.batch_size, self.classes))
        for i, index in enumerate(indices):
            image_id, label = self.image_ids[index], self.labels[index]
            image = read_image(f"{TRAIN_ROOT_PATH}/{image_id[0]}/{image_id[1]}/{image_id[2]}/{image_id}.jpg", SIZE, normalize_image = True)
            if self.augmentation:
                data = {"image": image}
                augmented = augmentor(**data)
                image = augmented["image"]
            X[i,] = image
            Y[i,] = label.toarray()
        return X, Y 
print(train_data.head())
print(train_data.shape)
train_value_counts_regular = train_data['landmark_id'].value_counts()
plt.figure(figsize=(12, 8))
sns.distplot(train_value_counts_regular, hist = False, rug = False, label = "Train Data normal value count distribution")
plt.xlabel("Value Counts")
plt.show()
def custom_sampler(df, min_samples, max_samples):
    landmark_id_counts = df['landmark_id'].value_counts()
    df1 = df[df['landmark_id'].isin(landmark_id_counts[(landmark_id_counts >= min_samples) & (landmark_id_counts <= max_samples)].index)]

    for id in landmark_id_counts[landmark_id_counts > max_samples].index:
        temp_df = df[df.landmark_id == id].sample(max_samples, random_state = SEED)
        df1 = pd.concat([df1, temp_df], axis = 0)
    return df1
train_data = custom_sampler(train_data, min_samples = 12, max_samples = 140)
train_value_counts_custom = train_data['landmark_id'].value_counts()
plt.figure(figsize=(12, 8))
sns.distplot(train_value_counts_custom, hist = False, rug = False, label = "Train Data custom value count distribution")
plt.xlabel("Value Counts")
plt.show() 
all_landmark_ids = train_data.landmark_id.unique().tolist()
print(f'Total number of classes sampled: {len(all_landmark_ids)}')
ALL_LABELS = np.sort(np.unique(all_landmark_ids))
lb = LabelBinarizer(sparse_output = True)
lb.fit(ALL_LABELS)
def batch_gap(y_t, y_p):
    pred_cat = tf.argmax(y_p, axis=-1)    
    y_t_cat = tf.argmax(y_t, axis=-1) * tf.cast(
        tf.reduce_sum(y_t, axis=-1), tf.int64)
    n_pred = tf.shape(pred_cat)[0]
    is_c = tf.cast(tf.equal(pred_cat, y_t_cat), tf.float32)
    GAP = tf.reduce_mean(
          tf.cumsum(is_c) * is_c / tf.cast(
              tf.range(1, n_pred + 1), 
              dtype=tf.float32))
    return GAP
def ModelCheckpoint():
    return tf.keras.callbacks.ModelCheckpoint(
                            'Model_epoch{epoch:02d}_vl{val_loss:.4f}_va{val_acc:.4f}_vbg{val_batch_gap:.4f}.h5', 
                            monitor = 'val_loss', 
                            verbose = 1, 
                            save_best_only = False, 
                            save_weights_only = True, 
                            mode = 'min', 
                            save_freq = 'epoch')
def generalized_mean_pool_2d(X):
    gm_exp = 3.0
    pool = (tf.reduce_mean(tf.abs(X**(gm_exp)), 
                        axis = [1, 2], 
                        keepdims = False) + 1.e-7)**(1./gm_exp)
    return pool
def create_model_gem(WEIGHTS, CLASSES): 
    input = tf.keras.layers.Input(shape = SHAPE)
    effnet_model = efn.EfficientNetB2(weights = WEIGHTS, include_top = False, input_tensor = input, pooling = None , classes = None)
    X = tf.keras.layers.Lambda(generalized_mean_pool_2d, name = 'gem')(effnet_model.output)
    X = tf.keras.layers.Dropout(0.25)(X)
    X = tf.keras.layers.Dense(1024, activation = 'relu')(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Dropout(0.25)(X)
    preds = tf.keras.layers.Dense(CLASSES, activation = 'softmax')(X)
    model = tf.keras.Model(inputs = effnet_model.input, outputs = preds)
    for layer in model.layers:
        layer.trainable = True
    return model
def create_model(WEIGHTS, CLASSES): 
    input = tf.keras.layers.Input(shape = SHAPE)
    effnet_model = efn.EfficientNetB2(weights = WEIGHTS, include_top = False, input_tensor = input, pooling = 'avg', classes = None)
    X = tf.keras.layers.Dropout(0.25)(effnet_model.output)
    X = tf.keras.layers.Dense(1024, activation = 'relu')(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Dropout(0.25)(X)
    preds = tf.keras.layers.Dense(CLASSES, activation = 'softmax')(X)
    model = tf.keras.Model(inputs = effnet_model.input, outputs = preds)
    for layer in model.layers:
        layer.trainable = True
    return model
print(len(all_landmark_ids))
def create_image_tta(im_res):
        im_res_lr = np.fliplr(im_res)
        return np.stack((im_res, im_res_lr))  
model = create_model(None, len(all_landmark_ids))
model.load_weights('../input/glrec2020/Model_epoch06_vl3.2417_va0.6774_vbg0.4803.h5')
if PLACES:
        model_places = VGG16_Places365(weights = '../input/keras-vgg16-places365/vgg16-places365_weights_tf_dim_ordering_tf_kernels.h5')
        predictions_to_return = 6
        places_classes = pd.read_csv('../input/keras-vgg16-places365/categories_places365_extended_v1.csv')
print('Start Creating Submission...')
prediction = []
for index, row in tqdm(submission.iterrows(), total = submission.shape[0]):
        image_id = row['id']
        file_name = f"{TEST_ROOT_PATH}/{image_id[0]}/{image_id[1]}/{image_id[2]}/{image_id}.jpg"
        image = read_image(file_name, SIZE, normalize_image = True)
        if PLACES:
            image_places = read_image(file_name, 224, normalize_image = False)
            places_preds = model_places.predict(create_image_tta(image_places))
            places_pred = np.mean(places_preds, axis = 0)
            places_top_preds = np.argsort(places_pred)[::-1][0:predictions_to_return]
            counter = 0 
            if (places_classes.loc[places_classes['class'] == places_top_preds[0]].io == 1).bool():
                counter +=1
            if (places_classes.loc[places_classes['class'] == places_top_preds[1]].io == 1).bool():
                counter +=1
            if (places_classes.loc[places_classes['class'] == places_top_preds[2]].io == 1).bool():
                counter +=1
            if (places_classes.loc[places_classes['class'] == places_top_preds[3]].io == 1).bool():
                counter +=1
            if counter >= 3:
                prediction.append(' ')                    
            else:
                pred = model.predict(create_image_tta(image))
                max_value = np.max(np.mean(pred, axis = 0))
                max_index = np.argmax(np.mean(pred, axis = 0))
                prediction.append(str(ALL_LABELS[max_index]) + ' ' + str(max_value))
        else:
            pred = model.predict(create_image_tta(image))
            max_value = np.max(np.mean(pred, axis = 0))
            max_index = np.argmax(np.mean(pred, axis = 0))
            prediction.append(str(ALL_LABELS[max_index]) + ' ' + str(max_value))
submission['landmarks'] = np.array(prediction)
submission.to_csv('submission.csv', index = False)
print(submission.head(25))