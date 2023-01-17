# general packages

import warnings

import json

import os

from PIL import Image

from glob import glob

from zipfile import ZipFile

import pandas as pd

import numpy as np



#sklearns 

from sklearn.metrics import accuracy_score

from sklearn.utils import class_weight

from sklearn.model_selection import train_test_split 



import random

import cv2

import gc

import math

import matplotlib.pyplot as plt

import seaborn as sns





# keras modules 

from keras.optimizers import Adam, Nadam, SGD

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model, load_model, Sequential

from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Conv2D, GlobalMaxPooling2D

from keras.layers import (MaxPooling2D, Input, Average, Activation, MaxPool2D,

                          Flatten, LeakyReLU, BatchNormalization, concatenate)

from keras import models

from keras import layers

from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201

from keras.applications.inception_v3 import InceptionV3

from keras.applications.vgg16 import VGG16

from keras.applications.inception_resnet_v2 import InceptionResNetV2

from keras_applications.resnet50 import ResNet50

from keras_applications.resnet_v2 import ResNet50V2

from keras.applications.xception import Xception

from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

from keras.callbacks import (ModelCheckpoint, LearningRateScheduler,

                             EarlyStopping, ReduceLROnPlateau, TensorBoard, CSVLogger)

from sklearn.metrics import accuracy_score, recall_score

from keras.callbacks import Callback





from keras.utils import Sequence

from keras import utils as np_utils

# from tensorflow.keras_radam import RAdam

from keras.callbacks import (Callback, ModelCheckpoint,

                                        LearningRateScheduler,EarlyStopping, 

                                        ReduceLROnPlateau,CSVLogger)



import albumentations

from PIL import Image, ImageOps, ImageEnhance

from albumentations.core.transforms_interface import ImageOnlyTransform

from albumentations.augmentations import functional as F

from albumentations import (ShiftScaleRotate,IAAAffine,IAAPerspective,

    RandomRotate90, IAAAdditiveGaussianNoise, GaussNoise

)



import tensorflow as tf

warnings.simplefilter('ignore')

sns.set_style('whitegrid')
SEED = 2020

batch_size = 64

FACTOR = 0.6

stats = (0.0692, 0.2051)



HEIGHT = 137 

WIDTH = 236



dim = (int(HEIGHT * FACTOR), int(WIDTH * FACTOR))

resize_wid = int(WIDTH * FACTOR)

resize_hit = int(HEIGHT * FACTOR)





def seed_all(seed):

    random.seed(seed)

    np.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)



seed_all(SEED)



# load files

im_path = '../input/137x236/137x236/'

train = pd.read_csv('/kaggle/input/train-new/fold_trian')

test = pd.read_csv('../input/bengaliai-cv19/test.csv')



# top 5 samples

train.head()
class GraphemeGenerator(Sequence):

    def __init__(self, data, batch_size, dim, kfold = (1,), shuffle=False, transform = None):

        

        data = data[["image_id", "grapheme_root", "vowel_diacritic",

                     "consonant_diacritic", "fold"]]

        data = data[data.fold.isin(kfold)].reset_index(drop=True)

        self._data = data

        

        self._label_1 = pd.get_dummies(self._data['grapheme_root'], 

                                       columns = ['grapheme_root'])

        self._label_2 = pd.get_dummies(self._data['vowel_diacritic'], 

                                       columns = ['vowel_diacritic'])

        self._label_3 = pd.get_dummies(self._data['consonant_diacritic'], 

                                       columns = ['consonant_diacritic'])

        self._list_idx = data.index.values

        self._batch_size = batch_size

        self._dim = dim

        self._shuffle = shuffle

        self._transform = transform

        self._kfold = kfold

        self.on_epoch_end()  

        

    def __len__(self):

        return int(np.floor(len(self._data)/self._batch_size))

    

    def __getitem__(self, index):

        batch_idx = self._indices[index*self._batch_size:(index+1)*self._batch_size]

        _idx = [self._list_idx[k] for k in batch_idx]



        Data     = np.empty((self._batch_size, *self._dim, 1))



        Target_1 = np.empty((self._batch_size, 168), dtype = int)

        Target_2 = np.empty((self._batch_size,  11), dtype = int)

        Target_3 = np.empty((self._batch_size,   7), dtype = int)

        

        for i, k in enumerate(_idx):

            image = cv2.imread(im_path + self._data['image_id'][k] + '.png') 

            image = cv2.resize(image, (resize_wid, resize_hit)) 



            if len(self._kfold) != 1:

                if self._transform is not None:

                    res =  self._transform(image=image)['image']

                    

            gray = lambda rgb : np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114]) 

            image = gray(image) 

            

            image = (image.astype(np.float32)/255.0 - stats[0])/stats[1]

            image = image[:, :, np.newaxis]

            Data[i,:, :, :] =  image

        

            Target_1[i,:] = self._label_1.loc[k, :].values

            Target_2[i,:] = self._label_2.loc[k, :].values

            Target_3[i,:] = self._label_3.loc[k, :].values

            

        return Data, [Target_1, Target_2, Target_3]

    

    

    def on_epoch_end(self):

        self._indices = np.arange(len(self._list_idx))

        if self._shuffle:

            np.random.shuffle(self._indices)
train_transform = albumentations.Compose([

                albumentations.OneOf([

                    ShiftScaleRotate(scale_limit=.15, rotate_limit=15, 

                                     border_mode=cv2.BORDER_CONSTANT),

                    IAAAffine(shear=20, mode='constant'),

                    IAAPerspective(),

                ])

            ])
train_generator = GraphemeGenerator(train, batch_size, dim , 

                                    shuffle = True,  

                                    kfold = (0, 1, 2, 3), 

                                    transform = train_transform)



val_generator = GraphemeGenerator(train, batch_size, dim, kfold = (4,),

                              shuffle = False)
from pylab import rcParams



# helper function to plot sample 

def plot_imgs(dataset_show):

    '''

    code: <plot_imgs> method from - https://www.kaggle.com/haqishen/gridmask

    '''

    rcParams['figure.figsize'] = 20,10

    for i in range(2):

        f, ax = plt.subplots(1,5)

        for p in range(5):

            idx = np.random.randint(0, len(dataset_show))

            img, label = dataset_show[idx]

            ax[p].grid(False)

            ax[p].imshow(img[0][:,:,0], cmap=plt.get_cmap('gray'))

            ax[p].set_title(idx)



plot_imgs(train_generator) 

plot_imgs(val_generator)
!pip install ../input/efficientnet-keras-source-code/repository/qubvel-efficientnet-c993591
import efficientnet.keras as efn 



def create_model(input_dim, output_dim, base_model):

    

    input_tensor = Input(input_dim)

    

    x = Conv2D(3, (3, 3), padding='same',  kernel_initializer='he_uniform', 

               bias_initializer='zeros')(input_tensor)

    curr_output = base_model(x)



    curr_output = GlobalAveragePooling2D()(curr_output)

    curr_output = Dense(784, activation='relu')(curr_output)

    curr_output = Dropout(0.5)(curr_output)



    oputput1 = Dense(168,  activation='softmax', name='gra') (curr_output)

    oputput2 = Dense(11,  activation='softmax', name='vow') (curr_output)

    oputput3 = Dense(7,  activation='softmax', name='cons') (curr_output)

    output_tensor = [oputput1, oputput2, oputput3]



    model = Model(input_tensor, output_tensor)

    

    return model



wg = '../input/efficientnet-keras-noisystudent-weights-b0b7/efficientnet-b0_noisy-student_notop.h5'

efnet = efn.EfficientNetB0(weights=wg,

                      include_top = False, input_shape=(*dim, 3))
def macro_recall(y_true, y_pred):

    return recall_score(y_true, y_pred, average='macro')



class CustomCallback(tf.keras.callbacks.Callback):

    def __init__(self, val_data, batch_size = 32):

        super().__init__()

        self.valid_data = val_data

        self.batch_size = batch_size

    

    def on_epoch_begin(self,epoch, logs={}):

        self.recall_scores = []

        self.avg_recall = []

        

    def on_epoch_end(self, epoch, logs={}):

        batches = len(self.valid_data)

        total = batches * self.batch_size

        self.val_recalls = {0: [], 1:[], 2:[]}

        

        for batch in range(batches):

            xVal, yVal = self.valid_data.__getitem__(batch)

            val_preds = self.model.predict(xVal)

            

            for i in range(3):

                preds = np.argmax(val_preds[i], axis=1)

                true = np.argmax(yVal[i], axis=1)

                self.val_recalls[i].append(macro_recall(true, preds))

        

        for i in range(3):

            self.recall_scores.append(np.average(self.val_recalls[i]))



        avg_result = np.average(self.recall_scores, weights=[2, 1, 1])

        self.avg_recall.append(avg_result)    



        if avg_result >= max(self.avg_recall):

            print("Avg. Recall Improved. Saving model.")

            print(f"Avg. Recall: {round(avg_result, 4)}")

            self.model.save_weights('best_avg_recall.h5')

        return
def Call_Back():

    # model check point

    checkpoint = ModelCheckpoint('Fold4.h5', 

                                 monitor = 'val_gra_loss', 

                                 verbose = 0, save_best_only=True, 

                                 mode = 'min',

                                 save_weights_only = True)

    

    csv_logger = CSVLogger('Fold4.csv')



    reduceLROnPlat = ReduceLROnPlateau(monitor='val_gra_loss',

                                   factor=0.3, patience=3,

                                   verbose=1, mode='auto',

                                   epsilon=0.0001, cooldown=1, min_lr=0.000001)

    

    custom_callback = CustomCallback(val_generator)



    return [checkpoint, csv_logger, reduceLROnPlat, custom_callback]





# epoch size 

epochs = 2



# calling all callbacks 

callbacks = Call_Back()



training = False



if training:

    # acatual training (fitting)

    train_history = model.fit_generator(

        train_generator,

        steps_per_epoch=batch_size, # batch_size

        validation_data=val_generator,

        validation_steps = batch_size,

        epochs=epochs,

        callbacks=callbacks

    )
model0 = create_model(input_dim=(*dim, 1), 

                     output_dim=(168,11,7), base_model = efnet)



model1 = create_model(input_dim=(*dim, 1), 

                     output_dim=(168,11,7), base_model = efnet)



model2 = create_model(input_dim=(*dim, 1), 

                     output_dim=(168,11,7), base_model = efnet)



model3 = create_model(input_dim=(*dim, 1), 

                     output_dim=(168,11,7), base_model = efnet)



model4 = create_model(input_dim=(*dim, 1), 

                     output_dim=(168,11,7), base_model = efnet)
model0.load_weights('../input/foldparts/Fold0.h5')

model1.load_weights('../input/foldparts/Fold1.h5')

model2.load_weights('../input/foldparts/Fold2.h5')

model3.load_weights('../input/foldparts/Fold3.h5')

model4.load_weights('../input/foldparts/Fold4.h5')
stats = (0.0692, 0.2051)

from tqdm import tqdm



# Image Prep

def resize_image(img, WIDTH_NEW, HEIGHT_NEW):

    # Reshape

    img = img.reshape(HEIGHT, WIDTH)

    image_resized = cv2.resize(img, (HEIGHT_NEW, WIDTH_NEW),

                               interpolation = cv2.INTER_AREA)



    return image_resized  
from tqdm import tqdm

def test_batch_generator(df, batch_size):

    num_imgs = len(df)



    for batch_start in range(0, num_imgs, batch_size):

        curr_batch_size = min(num_imgs, batch_start + batch_size) - batch_start

        idx = np.arange(batch_start, batch_start + curr_batch_size)



        names_batch = df.iloc[idx, 0].values

        imgs_batch = 255 - df.iloc[idx, 1:].values.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)

        X_batch = np.zeros((curr_batch_size, resize_hit, resize_wid, 1))

        

        for j in range(curr_batch_size):

            img = (imgs_batch[j,]*(255.0/imgs_batch[j,].max())).astype(np.uint8)

            img = resize_image(img, resize_hit, resize_wid)

            img = (img.astype(np.float32)/255.0 - stats[0])/stats[1]

            img = img[:, :, np.newaxis]

            X_batch[j,] = img



        yield X_batch, names_batch





# load the parquet files 

TEST = [

    "../input/bengaliai-cv19/test_image_data_0.parquet",

    "../input/bengaliai-cv19/test_image_data_1.parquet",

    "../input/bengaliai-cv19/test_image_data_2.parquet",

    "../input/bengaliai-cv19/test_image_data_3.parquet",

]



# placeholders 

row_id = []

target = []



# iterative over the test sets

for fname in tqdm(TEST):

    test_ = pd.read_parquet(fname)

    test_gen = test_batch_generator(test_, batch_size=batch_size)



    for batch_x, batch_name in test_gen:

        # prediction

        batch_predict0 = model0.predict(batch_x, batch_size = 128)

        batch_predict1 = model1.predict(batch_x, batch_size = 128)

        batch_predict2 = model2.predict(batch_x, batch_size = 128)

        batch_predict3 = model3.predict(batch_x, batch_size = 128)

        batch_predict4 = model4.predict(batch_x, batch_size = 128)

 

        for idx, name in enumerate(batch_name):

            row_id += [

                f"{name}_consonant_diacritic",

                f"{name}_grapheme_root",

                f"{name}_vowel_diacritic",

            ]

            target += [

                np.argmax((batch_predict0[2] + batch_predict1[2] + 

                           batch_predict2[2] + batch_predict3[2] + 

                           batch_predict4[2])/5, axis=1)[idx],

                

                np.argmax((batch_predict0[0] + batch_predict1[0] + 

                           batch_predict2[0] + batch_predict3[0] + 

                           batch_predict4[0])/5, axis=1)[idx],

                

                np.argmax((batch_predict0[1] + batch_predict1[1] + 

                           batch_predict2[1] + batch_predict3[1] + 

                           batch_predict4[1])/5, axis=1)[idx],

            ]



    del test_

    gc.collect()

    

    

df_sample = pd.DataFrame(

    {

        'row_id': row_id,

        'target':target

    },

    columns = ['row_id','target'] 

)



df_sample.to_csv('submission.csv',index=False)

gc.collect()