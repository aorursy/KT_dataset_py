# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import multiprocessing

num_cores = multiprocessing.cpu_count()
num_cores
train_df = pd.read_csv("/kaggle/input/understanding_cloud_organization/train.csv")

train_df.head()
image_and_label = train_df["Image_Label"].str.split("_", expand = True)
train_df["Image"] = image_and_label[0]

train_df["Label"] = image_and_label[1]
train_df.head()
img_dir = "/kaggle/input/understanding_cloud_organization/train_images/"
# Hot one encoding cho moi hinh

# Data augmentation??? => What types???

# Architecture of the model

# Training loop
train_df.head()
corr_df = pd.get_dummies(train_df, columns = ['Label'])

# fill null values with '-1'

corr_df = corr_df.fillna('-1')



# define a helper function to fill dummy columns

def get_dummy_value(row, cloud_type):

    ''' Get value for dummy column '''

    if cloud_type == 'fish':

        return row['Label_Fish'] * (row['EncodedPixels'] != '-1')

    if cloud_type == 'flower':

        return row['Label_Flower'] * (row['EncodedPixels'] != '-1')

    if cloud_type == 'gravel':

        return row['Label_Gravel'] * (row['EncodedPixels'] != '-1')

    if cloud_type == 'sugar':

        return row['Label_Sugar'] * (row['EncodedPixels'] != '-1')

    

# fill dummy columns

corr_df['Label_Fish'] = corr_df.apply(lambda row: get_dummy_value(row, 'fish'), axis=1)

corr_df['Label_Flower'] = corr_df.apply(lambda row: get_dummy_value(row, 'flower'), axis=1)

corr_df['Label_Gravel'] = corr_df.apply(lambda row: get_dummy_value(row, 'gravel'), axis=1)

corr_df['Label_Sugar'] = corr_df.apply(lambda row: get_dummy_value(row, 'sugar'), axis=1)



# check the result

corr_df.head()
df = corr_df.groupby("Image")['Label_Fish', 'Label_Flower', 'Label_Gravel', 'Label_Sugar'].max()
df = df.reset_index()
df
df["Image"]
df
df.iloc[:, 1:].values
# dictionary for fast access to ohe vectors

img_2_ohe_vector = {img:np.array(vec) for img, vec in zip(df['Image'], df.iloc[:, 1:].values)}
img_2_ohe_vector
import pandas as pd, numpy as np, os

from PIL import Image 

import cv2, keras, gc

import keras.backend as K

from keras import layers

from keras.models import Model

from keras.models import load_model

from keras.callbacks import LearningRateScheduler

from sklearn.model_selection import KFold

import matplotlib.pyplot as plt, time

from sklearn.metrics import roc_auc_score, accuracy_score

import random

from sklearn.model_selection import train_test_split

import multiprocessing

from copy import deepcopy

from sklearn.metrics import precision_recall_curve, auc

import keras

import keras.backend as K

from keras.optimizers import Adam

from keras.callbacks import Callback

from keras.applications.densenet import DenseNet201

from keras.layers import Dense, Flatten

from keras.models import Model, load_model

from keras.utils import Sequence

from albumentations import Compose, VerticalFlip, HorizontalFlip, Rotate, GridDistortion

import matplotlib.pyplot as plt

from IPython.display import Image

from tqdm import tqdm_notebook as tqdm

from numpy.random import seed

import tensorflow as tf

import glob

seed(10)

tf.random.set_seed(10)

%matplotlib inline
import copy

from albumentations import Compose, VerticalFlip, HorizontalFlip, Rotate, GridDistortion

class DataGenenerator(keras.utils.Sequence):

    def __init__(self, images_list=None, folder_imgs=img_dir, 

                 batch_size=32, shuffle=True, augmentation=None,

                 resized_height=260, resized_width=260, num_channels=3):

        self.batch_size = batch_size

        self.shuffle = shuffle

        self.augmentation = augmentation

        if images_list is None:

            self.images_list = os.listdir(folder_imgs)

        else:

            self.images_list = copy.deepcopy(np.array(images_list))

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

        print(X, y)

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
from albumentations import Normalize

albumentations_train = Compose([

    VerticalFlip(), HorizontalFlip(), Rotate(limit=20), GridDistortion(), Normalize()

], p=1)
!pip install -U git+https://github.com/qubvel/efficientnet
import efficientnet.keras as efn 

from keras.layers import Dense

def get_model():

    K.clear_session()

    base_model =  efn.EfficientNetB2(weights='imagenet', include_top=False, pooling='avg', input_shape=(1400, 2100, 3))

    for idx, layer in enumerate(base_model.layers):

        if idx != len(base_model.layers) - 1:

            layer.trainable = False

    x = base_model.output

    y_pred = Dense(4, activation='sigmoid')(x)

    return Model(inputs=base_model.input, outputs=y_pred)



model = get_model()
model.summary()
import keras

import keras.backend as K

from keras.optimizers import Adam

from keras.callbacks import Callback

from keras.applications.densenet import DenseNet201

from keras.layers import Dense, Flatten

from keras.models import Model, load_model

from keras.utils import Sequence

class PrAucCallback(Callback):

    def __init__(self, data_generator, num_workers=-1, 

                 early_stopping_patience=5, 

                 plateau_patience=3, reduction_rate=0.5,

                 stage='train', checkpoints_path='checkpoints/'):

        super(Callback, self).__init__()

        self.data_generator = data_generator

        self.num_workers = num_workers

        self.class_names = ['Fish', 'Flower', 'Sugar', 'Gravel']

        self.history = [[] for _ in range(len(self.class_names) + 1)] # to store per each class and also mean PR AUC

        self.early_stopping_patience = early_stopping_patience

        self.plateau_patience = plateau_patience

        self.reduction_rate = reduction_rate

        self.stage = stage

        self.best_pr_auc = -float('inf')

        if not os.path.exists(checkpoints_path):

            os.makedirs(checkpoints_path)

        self.checkpoints_path = checkpoints_path

        

    def compute_pr_auc(self, y_true, y_pred):

        pr_auc_mean = 0

        print(f"\n{'#'*30}\n")

        for class_i in range(len(self.class_names)):

            precision, recall, _ = precision_recall_curve(y_true[:, class_i], y_pred[:, class_i])

            pr_auc = auc(recall, precision)

            pr_auc_mean += pr_auc/len(self.class_names)

            print(f"PR AUC {self.class_names[class_i]}, {self.stage}: {pr_auc:.3f}\n")

            self.history[class_i].append(pr_auc)        

        print(f"\n{'#'*20}\n PR AUC mean, {self.stage}: {pr_auc_mean:.3f}\n{'#'*20}\n")

        self.history[-1].append(pr_auc_mean)

        return pr_auc_mean

              

    def is_patience_lost(self, patience):

        if len(self.history[-1]) > patience:

            best_performance = max(self.history[-1][-(patience + 1):-1])

            return best_performance == self.history[-1][-(patience + 1)] and best_performance >= self.history[-1][-1]    

              

    def early_stopping_check(self, pr_auc_mean):

        if self.is_patience_lost(self.early_stopping_patience):

            self.model.stop_training = True    

              

    def model_checkpoint(self, pr_auc_mean, epoch):

        if pr_auc_mean > self.best_pr_auc:

            # remove previous checkpoints to save space

            for checkpoint in glob.glob(os.path.join(self.checkpoints_path, 'classifier_densenet169_epoch_*')):

                os.remove(checkpoint)

            self.best_pr_auc = pr_auc_mean

            self.model.save(os.path.join(self.checkpoints_path, f'classifier_densenet169_epoch_{epoch}_val_pr_auc_{pr_auc_mean}.h5'))              

            print(f"\n{'#'*20}\nSaved new checkpoint\n{'#'*20}\n")

              

    def reduce_lr_on_plateau(self):

        if self.is_patience_lost(self.plateau_patience):

            new_lr = float(keras.backend.get_value(self.model.optimizer.lr)) * self.reduction_rate

            keras.backend.set_value(self.model.optimizer.lr, new_lr)

            print(f"\n{'#'*20}\nReduced learning rate to {new_lr}.\n{'#'*20}\n")

        

    def on_epoch_end(self, epoch, logs={}):

        y_pred = self.model.predict_generator(self.data_generator, workers=self.num_workers)

        y_true = self.data_generator.get_labels()

        # estimate AUC under precision recall curve for each class

        pr_auc_mean = self.compute_pr_auc(y_true, y_pred)

              

        if self.stage == 'val':

            # early stop after early_stopping_patience=4 epochs of no improvement in mean PR AUC

            self.early_stopping_check(pr_auc_mean)



            # save a model with the best PR AUC in validation

            self.model_checkpoint(pr_auc_mean, epoch)



            # reduce learning rate on PR AUC plateau

            self.reduce_lr_on_plateau()            

        

    def get_pr_auc_history(self):

        return self.history
df
VAL = 0.2

train_img_files = list(df["Image"][:int((1-VAL) * len(df))])

val_img_files = list(df["Image"][int((1-VAL) * len(df)):])
train_img_files

data_generator_train = DataGenenerator(train_img_files, img_dir, augmentation = albumentations_train)

data_generator_train_eval = DataGenenerator(train_img_files, shuffle=False)

data_generator_val = DataGenenerator(val_img_files, img_dir, augmentation = albumentations_train)
train_metric_callback = PrAucCallback(data_generator_train_eval)

val_callback = PrAucCallback(data_generator_val, stage='val')
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
model.compile(optimizer= keras.optimizers.Adam(amsgrad=False,name="Adam"),  loss='categorical_crossentropy', metrics=['accuracy'])



history_0 = model.fit(x=data_generator_train,

                              epochs=20,

                              callbacks=[train_metric_callback, val_callback],

                              workers=num_cores,

                              verbose=1)
history_0
model.save("/kaggle/output/model")