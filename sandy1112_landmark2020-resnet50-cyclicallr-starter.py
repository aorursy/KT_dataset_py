# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import tensorflow as tf

import cv2

import skimage.io

from skimage.transform import resize

from imgaug import augmenters as iaa

from sklearn import preprocessing

from sklearn.preprocessing import LabelBinarizer,LabelEncoder

from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras import layers

from tensorflow.keras.layers import Input, Add, Dense, Dropout, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,GlobalAveragePooling2D,Concatenate, ReLU, LeakyReLU,Reshape, Lambda

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

from tensorflow.keras.optimizers import Adam,SGD

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential, load_model, Model

from tensorflow.keras.callbacks import LearningRateScheduler

from tensorflow.keras.applications.resnet50 import ResNet50 

from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications.vgg16 import preprocess_input

from tensorflow.keras.applications.vgg16 import decode_predictions

from tensorflow.keras.utils import to_categorical

from tensorflow.keras import metrics

from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications.imagenet_utils import preprocess_input

from tqdm import tqdm

import imgaug as ia

from imgaug import augmenters as iaa

from PIL import Image

import keras.backend as K

K.set_image_data_format('channels_last')

K.set_learning_phase(1)





# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("../input/landmark-retrieval-2020/train.csv")

def get_paths(sub):

    index = ["0","1","2","3","4","5","6","7","8","9","a","b","c","d","e","f"]



    paths = []



    for a in index:

        for b in index:

            for c in index:

                try:

                    paths.extend([f"../input/landmark-retrieval-2020/{sub}/{a}/{b}/{c}/" + x for x in os.listdir(f"../input/landmark-retrieval-2020/{sub}/{a}/{b}/{c}")])

                except:

                    pass



    return paths
train_path = train



rows = []

for i in tqdm(range(len(train))):

    row = train.iloc[i]

    path  = list(row["id"])[:3]

    temp = row["id"]

    row["id"] = f"../input/landmark-retrieval-2020/train/{path[0]}/{path[1]}/{path[2]}/{temp}.jpg"

    rows.append(row["id"])

    

rows = pd.DataFrame(rows)

train_path["id"] = rows
batch_size = 128

seed = 42

shape = (224, 224, 3) ##desired shape of the image for resizing purposes

val_sample = 0.1 # 10 % as validation sample

train_labels = pd.read_csv('../input/landmark-retrieval-2020/train.csv')

train_labels.head()
k =train[['id','landmark_id']].groupby(['landmark_id']).agg({'id':'count'})

k.rename(columns={'id':'Count_class'}, inplace=True)

k.reset_index(level=(0), inplace=True)

freq_ct_df = pd.DataFrame(k)

freq_ct_df.head()
train_labels = pd.merge(train,freq_ct_df, on = ['landmark_id'], how='left')

train_labels.head()
freq_ct_df.sort_values(by=['Count_class'],ascending=False,inplace=True)

freq_ct_df.head()
freq_ct_df_top100 = freq_ct_df.iloc[:100]

top100_class = freq_ct_df_top100['landmark_id'].tolist()
top100class_train = train_path[train_path['landmark_id'].isin (top100_class) ]

top100class_train.shape
def getTrainParams():

    data = top100class_train.copy()

    le = preprocessing.LabelEncoder()

    data['label'] = le.fit_transform(data['landmark_id'])

    lbls = top100class_train['landmark_id'].tolist()

    lb = LabelBinarizer()

    labels = lb.fit_transform(lbls)

    

    return np.array(top100class_train['id'].tolist()),np.array(labels),le
class Landmark2020_DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, paths, labels, batch_size, shape, shuffle = False, use_cache = False, augment = False):

        self.paths, self.labels = paths, labels

        self.batch_size = batch_size

        self.shape = shape

        self.shuffle = shuffle

        self.use_cache = use_cache

        self.augment = augment

        if use_cache == True:

            self.cache = np.zeros((paths.shape[0], shape[0], shape[1], shape[2]), dtype=np.float16)

            self.is_cached = np.zeros((paths.shape[0]))

        self.on_epoch_end()

    

    def __len__(self):

        return int(np.ceil(len(self.paths) / float(self.batch_size)))

    

    def __getitem__(self, idx):

        indexes = self.indexes[idx * self.batch_size : (idx+1) * self.batch_size]



        paths = self.paths[indexes]

        X = np.zeros((paths.shape[0], self.shape[0], self.shape[1], self.shape[2]))

        # Generate data

        if self.use_cache == True:

            X = self.cache[indexes]

            for i, path in enumerate(paths[np.where(self.is_cached[indexes] == 0)]):

                image = self.__load_image(path)

                self.is_cached[indexes[i]] = 1

                self.cache[indexes[i]] = image

                X[i] = image

        else:

            for i, path in enumerate(paths):

                X[i] = self.__load_image(path)



        y = self.labels[indexes]

                

        if self.augment == True:

            seq = iaa.Sequential([

                iaa.OneOf([

                    iaa.Fliplr(0.5), # horizontal flips

                    

                    iaa.ContrastNormalization((0.75, 1.5)),

                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),

                    iaa.Multiply((0.8, 1.2), per_channel=0.2),

                    

                    iaa.Affine(rotate=0),

                    #iaa.Affine(rotate=90),

                    #iaa.Affine(rotate=180),

                    #iaa.Affine(rotate=270),

                    iaa.Fliplr(0.5),

                    #iaa.Flipud(0.5),

                ])], random_order=True)



            X = np.concatenate((X, seq.augment_images(X), seq.augment_images(X), seq.augment_images(X)), 0)

            y = np.concatenate((y, y, y, y), 0)

        

        return X, y

    

    def on_epoch_end(self):

        

        # Updates indexes after each epoch

        self.indexes = np.arange(len(self.paths))

        if self.shuffle == True:

            np.random.shuffle(self.indexes)



    def __iter__(self):

        """Create a generator that iterate over the Sequence."""

        for item in (self[i] for i in range(len(self))):

            yield item

            

    def __load_image(self, path):

        image_norm = skimage.io.imread(path)/255.0

        



        im = resize(image_norm, (shape[0], shape[1],shape[2]), mode='reflect')

        return im
from tensorflow.keras.callbacks import *



class CyclicLR(Callback):

    """This callback implements a cyclical learning rate policy (CLR).

    The method cycles the learning rate between two boundaries with

    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).

    The amplitude of the cycle can be scaled on a per-iteration or 

    per-cycle basis.

    This class has three built-in policies, as put forth in the paper.

    "triangular":

        A basic triangular cycle w/ no amplitude scaling.

    "triangular2":

        A basic triangular cycle that scales initial amplitude by half each cycle.

    "exp_range":

        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 

        cycle iteration.

    For more detail, please see paper.

    

    # Example

        ```python

            clr = CyclicLR(base_lr=0.001, max_lr=0.006,

                                step_size=2000., mode='triangular')

            model.fit(X_train, Y_train, callbacks=[clr])

        ```

    

    Class also supports custom scaling functions:

        ```python

            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))

            clr = CyclicLR(base_lr=0.001, max_lr=0.006,

                                step_size=2000., scale_fn=clr_fn,

                                scale_mode='cycle')

            model.fit(X_train, Y_train, callbacks=[clr])

        ```    

    # Arguments

        base_lr: initial learning rate which is the

            lower boundary in the cycle.

        max_lr: upper boundary in the cycle. Functionally,

            it defines the cycle amplitude (max_lr - base_lr).

            The lr at any cycle is the sum of base_lr

            and some scaling of the amplitude; therefore 

            max_lr may not actually be reached depending on

            scaling function.

        step_size: number of training iterations per

            half cycle. Authors suggest setting step_size

            2-8 x training iterations in epoch.

        mode: one of {triangular, triangular2, exp_range}.

            Default 'triangular'.

            Values correspond to policies detailed above.

            If scale_fn is not None, this argument is ignored.

        gamma: constant in 'exp_range' scaling function:

            gamma**(cycle iterations)

        scale_fn: Custom scaling policy defined by a single

            argument lambda function, where 

            0 <= scale_fn(x) <= 1 for all x >= 0.

            mode paramater is ignored 

        scale_mode: {'cycle', 'iterations'}.

            Defines whether scale_fn is evaluated on 

            cycle number or cycle iterations (training

            iterations since start of cycle). Default is 'cycle'.

    """



    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',

                 gamma=1., scale_fn=None, scale_mode='cycle'):

        super(CyclicLR, self).__init__()



        self.base_lr = base_lr

        self.max_lr = max_lr

        self.step_size = step_size

        self.mode = mode

        self.gamma = gamma

        if scale_fn == None:

            if self.mode == 'triangular':

                self.scale_fn = lambda x: 1.

                self.scale_mode = 'cycle'

            elif self.mode == 'triangular2':

                self.scale_fn = lambda x: 1/(2.**(x-1))

                self.scale_mode = 'cycle'

            elif self.mode == 'exp_range':

                self.scale_fn = lambda x: gamma**(x)

                self.scale_mode = 'iterations'

        else:

            self.scale_fn = scale_fn

            self.scale_mode = scale_mode

        self.clr_iterations = 0.

        self.trn_iterations = 0.

        self.history = {}



        self._reset()



    def _reset(self, new_base_lr=None, new_max_lr=None,

               new_step_size=None):

        """Resets cycle iterations.

        Optional boundary/step size adjustment.

        """

        if new_base_lr != None:

            self.base_lr = new_base_lr

        if new_max_lr != None:

            self.max_lr = new_max_lr

        if new_step_size != None:

            self.step_size = new_step_size

        self.clr_iterations = 0.

        

    def clr(self):

        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))

        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)

        if self.scale_mode == 'cycle':

            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)

        else:

            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)

        

    def on_train_begin(self, logs={}):

        logs = logs or {}



        if self.clr_iterations == 0:

            K.set_value(self.model.optimizer.lr, self.base_lr)

        else:

            K.set_value(self.model.optimizer.lr, self.clr())        

            

    def on_batch_end(self, epoch, logs=None):

        

        logs = logs or {}

        self.trn_iterations += 1

        self.clr_iterations += 1



        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))

        self.history.setdefault('iterations', []).append(self.trn_iterations)



        for k, v in logs.items():

            self.history.setdefault(k, []).append(v)

        

        K.set_value(self.model.optimizer.lr, self.clr())
def create_model(input_shape, n_out):

    inp = Input(input_shape)

    pretrain_model = ResNet50(include_top=False, weights='imagenet', input_tensor=inp)

    #x = pretrain_model.get_layer(name="block_13_expand_relu").output

    x = pretrain_model.output

    

    x = GlobalAveragePooling2D()(x)

    x = Dropout(0.5)(x)

    x = Dense(n_out, activation="sigmoid")(x)

    

    #for layer in pretrain_model.layers[:160]:

        #layer.trainable = False

    

    for layer in pretrain_model.layers:

        layer.trainable = False

        

    return Model(inp, x)
from tensorflow.keras.metrics import categorical_accuracy,top_k_categorical_accuracy

def top_5_accuracy(y_true, y_pred):

    return top_k_categorical_accuracy(y_true, y_pred, k=5)
nlabls = top100class_train['landmark_id'].nunique()

model = create_model(input_shape=(224,224,3), n_out=nlabls)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc',top_5_accuracy])

model.summary()
paths, labels,_ = getTrainParams()
keys = np.arange(paths.shape[0], dtype=np.int)  

np.random.seed(seed)

np.random.shuffle(keys)

lastTrainIndex = int((1-val_sample) * paths.shape[0])



pathsTrain = paths[0:lastTrainIndex]

labelsTrain = labels[0:lastTrainIndex]



pathsVal = paths[lastTrainIndex:]

labelsVal = labels[lastTrainIndex:]



print(paths.shape, labels.shape)

print(pathsTrain.shape, labelsTrain.shape, pathsVal.shape, labelsVal.shape)
train_generator = Landmark2020_DataGenerator(pathsTrain, labelsTrain, batch_size, shape, use_cache=False, augment = False, shuffle = True)

val_generator = Landmark2020_DataGenerator(pathsVal, labelsVal, batch_size, shape, use_cache=False, shuffle = False)
clr = CyclicLR(base_lr=0.005, max_lr=0.01,step_size=2000., mode='triangular')

epochs = 2

use_multiprocessing = True 

#workers = 1 
base_cnn = model.fit_generator(

    train_generator,

    steps_per_epoch=len(train_generator),

    validation_data=val_generator,

    validation_steps=8,

    #class_weight = class_weights,

    epochs=epochs,

    callbacks = [clr],

    use_multiprocessing=use_multiprocessing,

    #workers=workers,

    verbose=1)
model.save('ResNet50.h5')