!pip install tf-nightly
import pandas as pd

import numpy as np

import tensorflow as tf

from tensorflow.keras.layers import Dropout, Dense, Conv2D, MaxPool2D, Flatten

from tensorflow.keras.callbacks import TensorBoard

import matplotlib.pyplot as plt

import seaborn as sns

import random

import time

print(tf.__version__)



import os

import csv

import sys



from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.preprocessing.image import img_to_array

from tensorflow.keras.preprocessing.image import load_img

import cv2

from PIL import Image

from skimage.transform import resize



from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import Sequence,to_categorical



from tensorflow.keras.models import Sequential

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard

from tensorflow.keras import optimizers, losses, activations, models

from tensorflow.keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Concatenate

from tensorflow.keras import applications



%matplotlib inline
! ls /kaggle/input/inception-weights-v1/inception
IMGS_PER_CLASS = 13333

ENFORCE_COUNT_UNTOUCHED = 7000 # set to None to deactivate 



M1 = '/kaggle/input/ft-tl-inception/checkpoints/15-epochs-64-0.3-sgd-8e-05.h5'

M2 = '/kaggle/input/inception-weights-v1/inception/2.h5'

M3 = '/kaggle/input/inception-weights-v1/inception/3.h5'

Mf1 = '/kaggle/input/inception-weights-v1/inception/f1-15-epochs-64-0.3-rmsprop-0.0001.h5'

Mf2 = '/kaggle/input/inception-weights-v1/inception/f2-7-epochs-64-0.4-adam-0.0006.h5'

M_EN = '/kaggle/input/good-efficientnet-b3-adam-0-001-14-ep/checkpoints/15-epochs-32-0.2-adam-0.001.h5'



NCLASS = 3

BATCH_SIZE = 32

SEED = 123

random.seed(SEED)
# root_path = "/kaggle/input/selfie-classification-wiki"

# image_names_csv = os.path.join(root_path, "merged_big.csv")

# images_folder = os.path.join(root_path, "merged_big")



# p = pd.read_csv(image_names_csv)

# #print(p.head(5),'\n_____________________________')



# ppl_df = p[p['class']== 1]

# slf_df = p[p['class']== 0]

# rnd_df = p[p['class']== 2]



# ppl_len = IMGS_PER_CLASS #ppl_df.shape[0]

# ppl_len = min(IMGS_PER_CLASS, min(slf_df.shape[0], rnd_df.shape[0], ppl_df.shape[0]))

# if ppl_len != IMGS_PER_CLASS:

#     print('WARNING: adjuested IMGS_PER_CLASS to the max={}'.format(ppl_len))



# slf_full_df = slf_df

# slf_df = slf_df.sample(n = ppl_len, random_state = SEED)

# rnd_df = rnd_df.sample(n = ppl_len, random_state = SEED)

# ppl_df = ppl_df.sample(n = ppl_len, random_state = SEED)
# slf_test_df = slf_full_df[~slf_full_df.apply(tuple,1).isin(slf_df.apply(tuple,1))]

# slf_test_df.reset_index(inplace=True, drop=True)

# if ENFORCE_COUNT_UNTOUCHED: slf_test_df = slf_test_df.sample(n=ENFORCE_COUNT_UNTOUCHED, random_state=SEED)

# # slf_test_df
# frames = [slf_test_df]

# p = pd.concat(frames)



# p = p[(p['filename'] != 'BGRibeseya.jpg') & (p['filename'] != '.DS_Store') & (p['filename'] != '.cache')]



# p['filename'] = p['filename'].apply(lambda x: os.path.join(images_folder,x))



# p = p.sample(frac=1, axis=0,random_state = SEED)

# p.reset_index(drop=True, inplace = True)



# p
# X = np.array(p['filename'])

# y = np.array(p['class'])

# labels = { p['filename'][i] : p['class'][i] for i in range(p.shape[0])}
# ! ls '/kaggle/input/portraits-with-demography-dataset/data'
bad_paths = [

"/kaggle/input/portraits-with-demography-dataset/data/ad60fd8f0c651d0b0d6db29c562006c9.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/e9660d78a34fc8173d339ab7f8360e58.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/fc4310a49d6bee0b1bc7337963cf53ae.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/3c5189fb065e9a7cb016e2da109644f5.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/c33d9fb2124166c3e0acd3f3288d2261.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/01d34d3444f7f7d65ff5bea005a94f55.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/67a0ad40ba8545e2e1d3505db26f3e83.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/317fe9c3591817b78b7493bdac560e46.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/67d2a626d82c21b4fc2abe61b45ce3d7.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/f51ad5bc75d9d1625b2633746c11e89b.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/ab31a56c0afdfc419845f7f625adf749.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/f303a9fcdf5551a19599923295d35336.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/c00a28d511400d9ceeeb2d193d7dcd45.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/cce3e968c6a7d8cfe1752481f832a3e4.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/a2aaefcd61c8ede2aaea34ffacb01f79.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/5ebbeb0258b87199550d8c89fece7f95.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/49cb99e84145cecebc5d251b59ed25b0.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/8c4b4509ad7c5ead02a2a39f9646567f.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/7bbb6873a1b496a30f1e326a5363daf5.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/35762e9f9a824c591c46044f914fb824.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/f968ab7a9db29ed4e9d969b327a7b674.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/bb1254adfecea28ddf38068fe52b1230.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/92ca6cc6e5c1f66fa716f0fa26256e95.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/5d192b5b1a74b0f45c0e5fa2578f3486.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/5fdd32ffa2ed2a440d377b07aee286db.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/4170eb6ed9331bb01e1f4ee5d53b5f31.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/7bab8e2952ea2b055cb100ea3490a777.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/7bbb6873a1b496a30f1e326a5363daf5.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/2fc542888302f620054818fa7c3d47f0.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/e50e1150d4b98fd1e8603959256956dd.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/8a5d9e49da42ef0710643ce2088277d2.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/bba4f9398e896288f2249e24724cac91.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/99edafe3d87966efe65203c91fa4c082.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/b4430b6ebb99fbe5253947caa3e03ed8.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/9bcf9d55097b2b5bf89e3630375903b2.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/6b914a1382608b39ee766d97a048656a.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/f816ed7d960c0b8467f18e8c729db06d.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/0fb43489a985c57259a18da1e47c7131.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/075799efad3db7d83160b5b7edb541b2.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/b18f07e20a8d027d94e7625f949c9dfd.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/3c5189fb065e9a7cb016e2da109644f5.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/c8146ca312a31e8d9766968e4d0a26cc.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/ff48aaa5836f351414812123b6cd59d4.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/ad63c319049dbca259f53174c0a412c2.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/451789ea6f8508f775ac62b90c06880c.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/5b9b05ab1bb56de8d122cda00ef6bda1.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/2208b106eb22a7be1eb66fd53dacd0f7.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/03f1fd0ffd99041f26a8932c8505f148.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/85c5d45c69a44cbf042aee38cce02442.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/2208b106eb22a7be1eb66fd53dacd0f7.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/d423c62229f0b3c57515012d5e28a96f.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/a2aaefcd61c8ede2aaea34ffacb01f79.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/98aae8f1dc10ba7de84fc0fba91c3fd6.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/95b65fb69bb6075fb2f82a94fe76e1ec.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/c5aad77941347601bfb58c669312e9b1.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/89ef21dba367584534edc53a2cf88002.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/04b2f080fe1219e0b9ceaf68a1a45907.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/d2ce613cd8ebd6d00d111ee2baee5f7e.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/4de022a61ea980977a06784cab507254.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/7bbb6873a1b496a30f1e326a5363daf5.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/c1b8e33980b97514baf4875db1d2bb0e.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/0e534bbf80f0c20638d8f588bc459717.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/5e00b1d2bd1c0333affe506a22f8c071.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/f2908d12eb93f9b2edab9867167baee8.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/3c5189fb065e9a7cb016e2da109644f5.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/9804da5e32cfcf44e2867b3e7c05c9f8.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/9804da5e32cfcf44e2867b3e7c05c9f8.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/3e912e4b3fe2a03f2e7c6a66b6b54965.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/56040dfca039460c106a11bd2ba3a38f.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/5ebbeb0258b87199550d8c89fece7f95.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/c0db22029cc5087c5876832257c705c4.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/63c6e1f02fcf3c5c68db0dfc286d58bb.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/70bbe5649fe7d174a7de3d3dcabae597.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/9133271c36552f8fff41a5dcd44e34b7.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/23d9120f1d24741717c9b47e6cde1951.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/e1c515044185c383e5e59aa64ad91c8e.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/567a8217a70b286acb96cc699ea09d0b.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/fe9c46b3a76efe1aeb460888685c039b.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/55a19b9061aba5edbbf6e69ca5c3c898.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/635104c9f54815eaa33aaacbe3fa0f7f.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/4262450b2bd698ce91a1b5d5b161b13c.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/42fbe356285cb224d9081fb677d7df3b.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/ea296090404a7a78d65a2998cc384519.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/4cae38bb08eb074e34a1bebf92ff18bd.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/66c09a29d5ceeb49cb8f9c69ace5a162.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/54e8d3d737fdafac218174d1533c5830.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/b68e1d9dc979ace02fc8bed3d4b3aed6.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/ea296090404a7a78d65a2998cc384519.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/c40534e69e96dcca960166bc05d536f9.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/f2908d12eb93f9b2edab9867167baee8.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/c9bb4d835a2f22b71c0543c8045c05b0.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/54e8d3d737fdafac218174d1533c5830.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/74f339ab4bcceaeb365b33653d05ce22.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/fee951aea90e212fef059c1adc0c0962.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/87fd832501bfd8daa8ee533cfc068706.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/d8cf2965d958b8ff41e86d55ed221972.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/c98217a225f8ea9b41dfbfb6497621e5.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/7b6753dfa8e61db3dcdba13f4ed9c0a9.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/f7a78dfe46bea90ceffbf8c0f16d9d3e.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/c99041a608b6a87e584040359521e9df.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/ff48aaa5836f351414812123b6cd59d4.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/da963c69ed7b83b4fafd2d793a82dfe8.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/27ced020c11ddbbd8d3c72fc523230c0.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/7a5ea41f83e36cbb7508301409f35558.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/15cae4d5d36dc77c5b9848b0cd3d0fd3.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/5ebbeb0258b87199550d8c89fece7f95.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/1d88929c5a1abc9528d773ff1405b64b.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/1c703bdb7a37f06b76f85928caacd523.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/99a476097f7a3f9a8a94ecb2b977fc5d.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/ed0aba81629ba5261e133afd9ad55b21.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/c56be9f162f26e83e517dd212660d01d.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/00eec877b4fd685e910bf3183a4379e9.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/f968ab7a9db29ed4e9d969b327a7b674.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/5e2b23b7ad482482750d80024f163c58.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/755c3d665a64db2954841239d1abefeb.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/412d55f723bc68f901c9252a1dad7357.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/bc9a65fdd6c56d9f8dc0caa6d4731586.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/f8adabce022dde0c03ab2641b47a88cf.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/1ebe4af3b7495d95e83d0a4e1507fa69.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/ecced7e084139b34b2b5af6d02a26225.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/597a09f1c23e2c0b63ddcfd9a3b60444.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/e78365175669e91bda703b1b2a3c2175.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/58ce13c63c4e99f1353d656c212ec4dc.jpg",

"/kaggle/input/portraits-with-demography-dataset/data/a8299811920c723bbcb7b0b8f0d894dc.jpg",

]



len(bad_paths)
images_folder = '/kaggle/input/portraits-with-demography-dataset/data'

l = pd.read_csv('/kaggle/input/portraits-with-demography-dataset/bias_filename.tsv', sep='\t')



gend_ethn_df = l[["filename","genderLabel", "ethnicityLabel"]]

gend_ethn_df = gend_ethn_df.sample(n = 29226, random_state = SEED)



# gend_ethn_df = gend_ethn_df[(gend_ethn_df['filename'] != 'a8299811920c723bbcb7b0b8f0d894dc.jpg')]



gend_ethn_df['filename'] = gend_ethn_df['filename'].apply(lambda x: os.path.join(images_folder,x))

gend_ethn_df = gend_ethn_df[[x not in bad_paths for x in gend_ethn_df['filename']]]



gend_ethn_df.reset_index(inplace=True, drop=True)

gend_ethn_df
X = np.array(gend_ethn_df['filename'])

y = np.array([1] * len(X))

labels = { x : 1 for x in X}
#m = gend_ethn_df["ethnicityLabel"].unique().tolist()



j = gend_ethn_df["ethnicityLabel"].value_counts()

print(j.head(55))



    

# ethnicityLabel 1086 1089

# dobLabel 19961

# genderLabel 6 

# 55 over 50

# 32 over 100
def load_img_as_arr(img_path):

    img = cv2.imread(img_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return cv2.resize(img, (299, 299), interpolation = cv2.INTER_AREA)



def preprocess_img(img_path):

    try:

        img = load_img_as_arr(img_path)

    except:

        print(img_path)

        return None

    img = tf.cast(img, tf.float32)

    img = (img / 255.)

    return img
class DataGenerator(Sequence):

    'Generates data for Keras'

    def __init__(self, list_IDs, labels, batch_size=32, dim=(299,299), n_channels=3,

                 n_classes=3, shuffle=True):

        'Initialization'

        self.dim = dim

        self.batch_size = batch_size

        self.labels = labels

        self.list_IDs = list_IDs

        self.n_channels = n_channels

        self.n_classes = n_classes

        self.shuffle = shuffle

        self.on_epoch_end()



    def __len__(self):

        'Denotes the number of batches per epoch'

        return int(np.floor(len(self.list_IDs) / self.batch_size))



    def __getitem__(self, index):

        'Generate one batch of data'

        # Generate indexes of the batch

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]



        # Find list of IDs

        list_IDs_temp = [self.list_IDs[k] for k in indexes]



        # Generate data

        X, y = self.__data_generation(list_IDs_temp)



        return X, y

    

    def get_filenames(self):

        n = len(self) * self.batch_size

        return [self.list_IDs[k] for k in self.indexes[:n]]



    def on_epoch_end(self):

        'Updates indexes after each epoch'

        self.indexes = np.arange(len(self.list_IDs))

        if self.shuffle == True:

            np.random.shuffle(self.indexes)

            

    def get_list_IDs(self):

        return [self.list_IDs[i] for i in self.indexes]



    def __data_generation(self, list_IDs_temp):

        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        # Initialization

        # X = np.empty((self.batch_size, *self.dim, self.n_channels))

        X = np.zeros((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)

        y = np.empty((self.batch_size), dtype=int)

        s = 0



        for i, ID in enumerate(list_IDs_temp):

            img_resize = preprocess_img(ID)

            if img_resize == None:

                s += 1

                continue

            if img_resize.shape != (299,299,3):

                continue

            # X[i,] = np.load('data/' + ID + '.npy')

            X[i - s] = np.dstack([img_resize])

            y[i - s] = self.labels[ID]

        return  X, to_categorical(y, num_classes=self.n_classes)
pre_trained_model = applications.InceptionV3(

    input_shape = (299, 299, 3), # Shape of our images

    include_top = False, # Leave out the last fully connected layer

    weights = 'imagenet'

)



not_trained_model = applications.InceptionV3(

    input_shape = (299, 299, 3), # Shape of our images

    include_top = False, # Leave out the last fully connected layer

    weights = None

)





efnet_pre_trained_model = applications.EfficientNetB3(

    input_shape = (299, 299, 3), # Shape of our images

    include_top = False, # Leave out the last fully connected layer

    weights = None

)
def get_exp_pred_vals(model, test_generator):

    print("=============================================================================================")

    y_true = []

    y_pred = []

    x = []



    y_pred_ = []

    y_p = []

    count = 0

    for X, y in test_generator:

        count += 1

        pred = model.predict(X)



#         x += list(X)

        y_true += list(np.argmax(y, axis=1))

        y_pred += list(np.argmax(pred, axis=1))

        y_p = list(np.argmax(pred, axis=1))

        y_pred_ += [pred[i][y_p[i]] for i in range(len(y_p))]

    

    x = test_generator.get_filenames()

    print(len(x), len(y_true), len(y_pred), len(y_pred_))

    d = {"x": x, "y_true": y_true, "y_pred": y_pred, "y_pred_": y_pred_}

    df = pd.DataFrame(d, columns=["x", "y_true", "y_pred", "y_pred_"])

    df = df.sort_values('y_pred_', ascending = True)

    

    return list(df["x"]), list(df.y_true), list(df.y_pred)
from tensorboard.plugins.hparams import api as hp



%load_ext tensorboard

# Clear any logs from previous runs

!rm -rf ./logs/ 



    

HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([64]))

HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([0.001])) #0.0001,0.001,

HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.2])) #0.3, 0.4, 0.35, 0.5

HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam']))#, 'rmsprop','adam', 'adadelta']))



METRIC_ACCURACY = 'accuracy'
def create_model(hparams, base_model = pre_trained_model, weights = None):

    nclass = NCLASS

    add_model = Sequential()

    add_model.add(base_model)

    add_model.add(GlobalAveragePooling2D())

    if weights == (M1 or M3):

        add_model.add(Dense(1024, activation='relu'))

    add_model.add(Dropout(hparams[HP_DROPOUT]))

    add_model.add(Dense(nclass, activation='softmax'))

    model = add_model

    

    optimizer = None

    if hparams[HP_OPTIMIZER] == 'adam':

        optimizer = optimizers.Adam(learning_rate=hparams[HP_LEARNING_RATE])

    elif hparams[HP_OPTIMIZER] == 'sgd':

        optimizer = optimizers.SGD(learning_rate=hparams[HP_LEARNING_RATE], momentum=0.9)

    elif hparams[HP_OPTIMIZER] == 'rmsprop':

        optimizer = optimizers.RMSprop(learning_rate=hparams[HP_LEARNING_RATE])

    elif hparams[HP_OPTIMIZER] == 'adadelta':

        optimizer = optimizers.Adadelta(learning_rate=hparams[HP_LEARNING_RATE])

    else:

        raise Exception("Unknown HP_OPTIMIZER value =", hparams[HP_OPTIMIZER])

    

    if weights != None:

        #model = tf.keras.models.load_model(weights)

        if weights == (M1):

            train_from_layer = 249

            model.layers[0].trainable = True

            for layer in model.layers[0].layers[:train_from_layer]:

                layer.trainable =  False

            model.load_weights(weights)

        elif weights == Mf1 or weights == Mf2 or weights == M_EN:

            #model.layers[0].trainable = True

            model.load_weights(weights)

        else:

            model = tf.keras.models.load_model(weights)

    

    model.compile(

        loss='categorical_crossentropy', 

        optimizer=optimizer,

        metrics=['accuracy']



    )

    

    return model
def results(fine_tuned_model, base_model):

    hparams = {

            HP_BATCH_SIZE: 64,

            HP_DROPOUT: 0.4,

            HP_OPTIMIZER: 'adam',

            HP_LEARNING_RATE: 0.0001}

    params = {

            'dim': (299, 299),

            'batch_size': hparams[HP_BATCH_SIZE],

            'n_classes': 3,

            'n_channels': 3,

        }

    model = create_model(hparams,base_model, fine_tuned_model)

    test_generator = DataGenerator(

        X, labels, **params, shuffle = False

    )





#     loss, acc = model.evaluate_generator(test_generator)

#     print('Loaded model loss: ', loss)

#     print('Loaded model accuracy: ', str(acc))



    x, y_true, y_pred = get_exp_pred_vals(model, test_generator)

    

    np.set_printoptions(precision=3)

#     cm = confusion_matrix(y_true, y_pred, normalize='true')

#     print("\nConfusion matrix of class accuracies\n", cm)

    return np.array(x), np.array(y_true), np.array(y_pred)
# def create_model(hparams, base_model=pre_trained_model, weights=None):

#     nclass = NCLASS

#     add_model = Sequential()

#     add_model.add(base_model)

#     add_model.add(GlobalAveragePooling2D())

#     if weights == (M1 or M3):

#         add_model.add(Dense(1024, activation='relu'))

#     add_model.add(Dropout(0.4))

#     add_model.add(Dense(nclass, activation='softmax'))

#     model = add_model

        

#     if weights != None:

#         if weights == (M1):

# #             train_from_layer = 249

# #             model.layers[0].trainable = True

# #             for layer in model.layers[0].layers[:train_from_layer]:

# #                 layer.trainable =  False

#             model.load_weights(weights)

#         elif weights == Mf1 or weights == Mf2 or weights == M_EN:

#             model.load_weights(weights)

#         else:

#             model = tf.keras.models.load_model(weights)

    

# #     model.compile(

# #         loss='categorical_crossentropy', 

# # #         optimizer=optimizers.Adam(learning_rate=0.0001),

# #         metrics=['accuracy']



# #     )

# #     

#     return model
# def get_exp_pred_vals(model, test_generator):

#     y_true = []

#     y_pred = []

#     count = 0

#     for X, y in test_generator:

#         if count % 10 == 0: print('get_exp_pred_vals iteration', count)

#         count += 1

#         pred = model.predict(X)



#         y_true += list(np.argmax(y, axis=1))

#         y_pred += list(np.argmax(pred, axis=1))

#     print("Number of generations: ", count)

#     return np.array(y_true), np.array(y_pred)

    
def get_attr(filename, pred, meta_df):

    name = os.path.splitext(filename)[0]

    row = meta_df[meta_df['img_name'] == name]

    return pred(row)



def get_attr_persons(filename, pred, meta_df):

    row = meta_df[meta_df['filename'] == filename]

#     print(type(row))

    #print("--------------get_attr_persons: ",row)

    return pred(row)





def get_stats(idx, y_true, y_pred):

    percent = lambda x, total: '{:.2%}'.format(x/total)

    if len(idx) == 0: return [0] + ['N/A'] * 3

    

    CLASS_IDX = 0

#     print(y_pred[idx])

    cm = confusion_matrix(y_true[idx], y_pred[idx])

#     print(cm)

    cm = cm[CLASS_IDX]

    

    total = cm.sum()

    correct = cm[CLASS_IDX]

    false_person = cm[1] if len(cm) > 1 else 0

    false_random = cm[2] if len(cm) > 2 else 0

    return [total, percent(correct, total), percent(false_person, total), percent(false_random, total)]



def get_stats_person(idx, y_true, y_pred):

    percent = lambda x, total: '{:.2%}'.format(x/total)

    if len(idx) == 0: return [0] + ['N/A'] * 3



    cm = confusion_matrix(y_true[idx], y_pred[idx])

#     print(cm)

    if cm.shape[0] == 3:

        cm = cm[1]

        false_selfie, correct, false_random = cm

    elif cm.shape[0] == 2:

        cm = cm[0]

        correct, false_random = cm

        false_selfie = 0

    else:

        cm = cm[0]

        correct = cm[0]

        false_selfie, false_random = 0,0

        

    total = cm.sum()

    return [total, percent(correct, total), percent(false_selfie, total), percent(false_random, total)]
def person_bias_analysis(x_filenames, y_true, y_pred, attrs, meta_df):

#     x_filenames = [os.path.basename(f) for f in x_filenames]

    #x_filenames = [f for f in x_filenames]

    bias_df = pd.DataFrame(columns=['attribute', 'total', 'correct', 'false_selfie', 'false_random'])

    #print(len(x_filenames))



    print('Evaluating model biases...')

    for i, (name, pred) in enumerate(attrs):

        print(name, pred, i)

        idx = [k for k,f in enumerate(x_filenames) if get_attr_persons(f, pred, meta_df)]

#         print(idx)

        

        print("idx", len(idx))

        bias_df.loc[i] = [name] + get_stats_person(idx, y_true, y_pred)

    print(bias_df)



    return bias_df
'TEST SUBSET SIZE = {}'.format(X.shape[0])
attrs = [

    ('African Americans', lambda row: row['ethnicityLabel'].values[0] == 'African Americans'),

    ('Armenians', lambda row: row['ethnicityLabel'].values[0] == 'Armenians' or row['ethnicityLabel'].values[0] == 'Armenian American'),

    ('Greeks', lambda row: row['ethnicityLabel'].values[0] == 'Greeks'),

    ('Czechs', lambda row: row['ethnicityLabel'].values[0] == 'Czechs'),

    ('Jews', lambda row: row['ethnicityLabel'].values[0] == 'Jewish people' or row['ethnicityLabel'].values[0] == 'American Jews'),

    ('Serbs', lambda row: row['ethnicityLabel'].values[0] == 'Serbs'),

    ('Bulgarians', lambda row: row['ethnicityLabel'].values[0] == 'Bulgarians'),

    ('Ukrainians', lambda row: row['ethnicityLabel'].values[0] == 'Ukrainians'),

    ('Swedish-speaking population of Finland', lambda row: row['ethnicityLabel'].values[0] == 'Swedish-speaking population of Finland'),

    ('Albanians', lambda row: row['ethnicityLabel'].values[0] == 'Albanians'),

    ('Germans', lambda row: row['ethnicityLabel'].values[0] == 'Germans' or row['ethnicityLabel'].values[0] == 'German Americans'),

    ('French people', lambda row: row['ethnicityLabel'].values[0] == 'French people'),

    ('English people', lambda row: row['ethnicityLabel'].values[0] == 'English people' or row['ethnicityLabel'].values[0] == 'English American'),

    ('Poles', lambda row: row['ethnicityLabel'].values[0] == 'Poles'),

    ('Yoruba people', lambda row: row['ethnicityLabel'].values[0] == 'Yoruba people'),

    ('Russians', lambda row: row['ethnicityLabel'].values[0] == 'Russians'),

    ('Italians', lambda row: row['ethnicityLabel'].values[0] == 'Italians' or  row['ethnicityLabel'].values[0] == 'Italian American'),

    #('American Jews', lambda row: row['ethnicityLabel'].values[0] == 'American Jews'),

    ('Japanese people', lambda row: row['ethnicityLabel'].values[0] == 'Japanese people'),

    ('Americans', lambda row: row['ethnicityLabel'].values[0] == 'Americans of the United States' or  row['ethnicityLabel'].values[0] == 'White American'),

    ('Han Chinese people', lambda row: row['ethnicityLabel'].values[0] == 'Han Chinese people'),

    ('Tibetan people', lambda row: row['ethnicityLabel'].values[0] == 'Tibetan people'),

    ('Sinhala people', lambda row: row['ethnicityLabel'].values[0] == 'Sinhala people'),

    ('British people', lambda row: row['ethnicityLabel'].values[0] == 'British people'),

    ('Arabs', lambda row: row['ethnicityLabel'].values[0] == 'Arabs'),

    ('Irish people', lambda row: row['ethnicityLabel'].values[0] == 'Irish people'),

    ('Swedes', lambda row: row['ethnicityLabel'].values[0] == 'Swedish American' or  row['ethnicityLabel'].values[0] == 'Swedes'),

    #('White American', lambda row: row['ethnicityLabel'].values[0] == 'White American'),

    #('Swedes', lambda row: row['ethnicityLabel'].values[0] == 'Swedes'),

    ('Georgians', lambda row: row['ethnicityLabel'].values[0] == 'Georgians'),

    ('Hungarians', lambda row: row['ethnicityLabel'].values[0] == 'Hungarians'),

    ('other', lambda row: row['ethnicityLabel'].values[0] not in ['African Americans', 'Armenians', 'Greeks', 'Czechs', 'Jewish people', 'Serbs', 'Bulgarians', 'Ukrainians', 'Swedish-speaking population of Finland', 'Albanians', 'Germans', 'French people', 'English people', 'Poles', 'Yoruba people', 'Russians', 'American Jews', 'Italians', 'Armenian American', 'Japanese people', 'Americans of the United States', 'Han Chinese people', 'Tibetan people', 'Sinhala people', 'British people', 'Arabs', 'Irish people', 'Swedish American', 'White American', 'Swedes', 'Georgians', 'Hungarians']),



#     ('Ashkenazi Jews', lambda row: row['ethnicityLabel'].values[0] == 'Ashkenazi Jews'),

#     ('Scottish people', lambda row: row['ethnicityLabel'].values[0] == 'Scottish people'),

#     #('Italian American', lambda row: row['ethnicityLabel'].values[0] == 'Italian American'),

#     ('Irish Americans', lambda row: row['ethnicityLabel'].values[0] == 'Irish Americans'),

#     ('Croats', lambda row: row['ethnicityLabel'].values[0] == 'Croats'),

#     ('Norwegians', lambda row: row['ethnicityLabel'].values[0] == 'Norwegians'),

#     ('Spaniards', lambda row: row['ethnicityLabel'].values[0] == 'Spaniards'),

#     ('Manchu', lambda row: row['ethnicityLabel'].values[0] == 'Manchu'),

#     ('Belarusians', lambda row: row['ethnicityLabel'].values[0] == 'Belarusians'),

#     ('Koreans', lambda row: row['ethnicityLabel'].values[0] == 'Koreans'),    

#     ('Romanians', lambda row: row['ethnicityLabel'].values[0] == 'Romanians'),

#     ('Bengali people', lambda row: row['ethnicityLabel'].values[0] == 'Bengali people'),

#     ('Vietnamese people', lambda row: row['ethnicityLabel'].values[0] == 'Vietnamese people'),

#     ('Dutch people', lambda row: row['ethnicityLabel'].values[0] == 'Dutch people'),

#     #('German Americans', lambda row: row['ethnicityLabel'].values[0] == 'German Americans'),

#     ('Austrians', lambda row: row['ethnicityLabel'].values[0] == 'Austrians'),

#     #('English American', lambda row: row['ethnicityLabel'].values[0] == 'English American'),

#     ('Kurds', lambda row: row['ethnicityLabel'].values[0] == 'Kurds'),

#     ('Indian people', lambda row: row['ethnicityLabel'].values[0] == 'Indian people'),

#     ('Macedonians', lambda row: row['ethnicityLabel'].values[0] == 'Macedonians'),

#     ('Ottoman Greeks', lambda row: row['ethnicityLabel'].values[0] == 'Ottoman Greeks'),

#     ('Mongols', lambda row: row['ethnicityLabel'].values[0] == 'Mongols'),

#     ('Slovaks', lambda row: row['ethnicityLabel'].values[0] == 'Slovaks')

]

len(attrs)
attrs_gen = [

    ('male', lambda row: row['genderLabel'].values[0] == 'male'),

    ('female', lambda row: row['genderLabel'].values[0] == 'female'),

    ('other', lambda row: row['genderLabel'].values[0] != 'male' and row['genderLabel'].values[0] != 'female'),

]

len(attrs_gen)
%%time

x, y_true, y_pred = results(M1, pre_trained_model)
%%time

df1 = person_bias_analysis(x, y_true, y_pred, attrs, gend_ethn_df)

df1
%%time

df1 = person_bias_analysis(x, y_true, y_pred, attrs_gen, gend_ethn_df)

df1
cm = confusion_matrix(y_true, y_pred)

cm[1]
'accuracy = {}'.format(round(cm[1][1] / sum(cm[1]), 3))
%%time

x, y_true, y_pred = results(M_EN, efnet_pre_trained_model)

df1 = person_bias_analysis(x, y_true, y_pred, attrs, gend_ethn_df)

df1
df1 = person_bias_analysis(x, y_true, y_pred, attrs_gen, gend_ethn_df)

df1
cm = confusion_matrix(y_true, y_pred)

cm[1]
'accuracy = {}'.format(round(cm[1][1] / sum(cm[1]), 3))
%%time

x, y_true, y_pred = results(Mf2, not_trained_model)
%%time

df1 = person_bias_analysis(x, y_true, y_pred, attrs, gend_ethn_df)

df1
df1 = person_bias_analysis(x, y_true, y_pred, attrs_gen, gend_ethn_df)

df1
cm = confusion_matrix(y_true, y_pred)

cm[1]
'accuracy = {}'.format(round(cm[1][1] / sum(cm[1]), 3))
def selfie_bias_analysis(x_filenames, y_true, y_pred, attrs):

#     params = {

#         'dim': (299, 299),

#         'batch_size': BATCH_SIZE,

#         'n_classes': NCLASS,

#         'n_channels': 3,

#     }



#     test_generator = DataGenerator(X, labels, **params, shuffle=False)

    

#     print('Evaluating model performance...')

#     loss, acc = model.evaluate_generator(test_generator)

#     print('Loaded model loss: ', loss)

#     print('Loaded model accuracy: ', str(acc))

    

#     x_filenames = test_generator.get_list_IDs()

    x_filenames = [os.path.basename(f) for f in x_filenames]

    

#     print('Evaluating model predictions...')

#     y_true, y_pred = get_exp_pred_vals(model, test_generator)

    

    columns = "img_name popularity_score partial_faces is_female baby child teenager youth middle_age senior white black asian oval_face round_face heart_face smiling mouth_open frowning wearing_glasses wearing_sunglasses wearing_lipstick tongue_out duck_face black_hair blond_hair brown_hair red_hair curly_hair straight_hair braid_hair showing_cellphone using_earphone using_mirror braces wearing_hat harsh_lighting dim_lighting"

    meta_df =  pd.read_csv('/kaggle/input/selfie-dataset-metadata/selfie_dataset.txt', names=columns.split(), sep=' ')

    

    bias_df = pd.DataFrame(columns=['attribute', 'total', 'correct', 'false_person', 'false_random'])



    print('Evaluating model biases...')

    selfie_idx = [i for i,y in enumerate(y_true) if y == 0]

    for i, (name, pred) in enumerate(attrs):

        idx = [k for k,f in enumerate(x_filenames) if k in selfie_idx and get_attr(f, pred, meta_df)]

        bias_df.loc[i] = [name] + get_stats(idx, y_true, y_pred)

        print(i)



    return bias_df
'TEST SUBSET SIZE = {}'.format(X.shape[0])
attrs = [

    ('female', lambda row: (row['is_female'] == 1).values[0]),

    ('male', lambda row: (row['is_female'] == -1).values[0]),

    ('white', lambda row: (row['white'] == 1).values[0]),

    ('black', lambda row: (row['black'] == 1).values[0]),

    ('asian', lambda row: (row['asian'] == 1).values[0]),

    ('other_races', lambda row: ((row['white'] == -1) & (row['black'] == -1) & (row['asian'] == -1)).values[0]),

    ('baby', lambda row: (row['baby'] == 1).values[0]),

    ('child', lambda row: (row['child'] == 1).values[0]),

    ('teenager', lambda row: (row['teenager'] == 1).values[0]),

    ('youth', lambda row: (row['youth'] == 1).values[0]),

    ('middle_age', lambda row: (row['middle_age'] == 1).values[0]),

    ('senior', lambda row: (row['senior'] == 1).values[0]),

    ('other_ages', lambda row: ((row['baby'] == -1) & (row['child'] == -1) & (row['teenager'] == -1) & (row['youth'] == -1) & (row['middle_age'] == -1) & (row['senior'] == -1)).values[0]),

]
# %%time

# x, y_true, y_pred = results(M1, pre_trained_model)

# df1 = selfie_bias_analysis(x, y_true, y_pred, attrs)

# df1
# %%time

# x, y_true, y_pred = results(M_EN, efnet_pre_trained_model)

# df2 = selfie_bias_analysis(x, y_true, y_pred, attrs)

# df2
# %%time

# x, y_true, y_pred = results(Mf2, not_trained_model)

# df3 = selfie_bias_analysis(x, y_true, y_pred, attrs)

# df3
# # intialise data of lists. 

# data = {'Name':['Tom', 'nick', 'krish', 'jack'], 'Age':[20, 21, 19, 18]} 

  

# # Create DataFrame 

# df = pd.DataFrame(data) 

# df
# import matplotlib.pyplot as plt

# import pandas as pd

# from pandas.plotting import table # EDIT: see deprecation warnings below



# ax = plt.subplot(111, frame_on=False) # no visible frame

# ax.xaxis.set_visible(False)  # hide the x axis

# ax.yaxis.set_visible(False)  # hide the y axis



# table(ax, df)  # where df is your data frame



# plt.savefig('mytable.png')
# %%time



# model = create_model(M_EN, efnet_pre_trained_model)

# selfie_bias_analysis(model, attrs)
# %%time

# model = create_model(M1, pre_trained_model)

# selfie_bias_analysis(model, attrs)
# %%time

# model = create_model(M2, pre_trained_model)

# selfie_bias_analysis(model, attrs)
# %%time

# model = create_model(M3, pre_trained_model)

# selfie_bias_analysis(model, attrs)
# %%time

# model = create_model(Mf1, not_trained_model)

# selfie_bias_analysis(model, attrs)
# %%time

# model = create_model(Mf2,not_trained_model)

# selfie_bias_analysis(model, attrs)
# model = tf.keras.models.load_model(M1)



# params = {

#     'dim': (299, 299),

#     'batch_size': 64,

#     'n_classes': NCLASS,

#     'n_channels': 3,

# }



# test_generator = DataGenerator(

#     partition['test'], labels, **params, shuffle = False

# )



# loss, acc = model.evaluate_generator(test_generator)

# print('Loaded model loss: ', loss)

# print('Loaded model accuracy: ', str(acc))
# x_filenames = test_generator.get_list_IDs()

# x_filenames = [os.path.basename(f) for f in x_filenames]
# x, y_true, y_pred = get_exp_pred_vals(model, test_generator)
# columns = "img_name popularity_score partial_faces is_female baby child teenager youth middle_age senior white black asian oval_face round_face heart_face smiling mouth_open frowning wearing_glasses wearing_sunglasses wearing_lipstick tongue_out duck_face black_hair blond_hair brown_hair red_hair curly_hair straight_hair braid_hair showing_cellphone using_earphone using_mirror braces wearing_hat harsh_lighting dim_lighting"

# meta_df =  pd.read_csv('/kaggle/input/selfie-dataset-metadata/selfie_dataset.txt', names=columns.split(), sep=' ')

# meta_df
# def get_attr(filename, pred, meta_df):

#     name = os.path.splitext(filename)[0]

#     row = meta_df[meta_df['img_name'] == name]

#     return pred(row)



# def get_stats(idx, y_true, y_pred):

#     percent = lambda x, total: '{:.2%}'.format(x/total)

#     if len(idx) == 0: return [0] + ['N/A'] * 3

    

#     CLASS_IDX = 0

#     cm = confusion_matrix(y_true[idx], y_pred[idx])[CLASS_IDX]

    

#     total = cm.sum()

#     correct = cm[CLASS_IDX]

#     false_person = cm[1] if len(cm) > 1 else 0

#     false_random = cm[2] if len(cm) > 2 else 0

#     return [total, percent(correct, total), percent(false_person, total), percent(false_random, total)]
# attrs = [

#     ('female', lambda row: (row['is_female'] == 1).values[0]),

#     ('male', lambda row: (row['is_female'] == -1).values[0]),

#     ('white', lambda row: (row['white'] == 1).values[0]),

#     ('black', lambda row: (row['black'] == 1).values[0]),

#     ('asian', lambda row: (row['asian'] == 1).values[0]),

#     ('other_races', lambda row: ((row['white'] == -1) & (row['black'] == -1) & (row['asian'] == -1)).values[0]),

#     ('baby', lambda row: (row['baby'] == 1).values[0]),

#     ('child', lambda row: (row['child'] == 1).values[0]),

#     ('teenager', lambda row: (row['teenager'] == 1).values[0]),

#     ('youth', lambda row: (row['teenager'] == 1).values[0]),

#     ('middle_age', lambda row: (row['teenager'] == 1).values[0]),

#     ('senior', lambda row: (row['teenager'] == 1).values[0]),

#     ('other_ages', lambda row: ((row['baby'] == -1) & (row['child'] == -1) & (row['teenager'] == -1) & (row['youth'] == -1) & (row['middle_age'] == -1) & (row['senior'] == -1)).values[0]),

# ]
# bias_df = pd.DataFrame(columns=['attribute', 'total', 'correct', 'false_person', 'false_random'])



# selfie_idx = [i for i,y in enumerate(y_true) if y == 0]

# for i, (name, pred) in enumerate(attrs):

#     idx = [k for k,f in enumerate(x_filenames) if k in selfie_idx and get_attr(f, pred, meta_df)]

#     bias_df.loc[i] = [name] + get_stats(idx, y_true, y_pred)



# bias_df