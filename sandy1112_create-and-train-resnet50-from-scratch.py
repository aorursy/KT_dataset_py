from IPython.display import Image

Image(filename='../input/landmark-retrieval-2020/train/0/0/0/000014b1f770f640.jpg') 



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

from tensorflow.keras.preprocessing import image

from tensorflow.keras.utils import to_categorical

from tensorflow.keras import metrics

from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications.imagenet_utils import preprocess_input

from tensorflow.keras.initializers import glorot_uniform

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

train_path["id"] = train_path.id.map(lambda path: f"../input/landmark-retrieval-2020/train/{path[0]}/{path[1]}/{path[2]}/{path}.jpg")
##Old implementation - changed after suggestion from @nawidsayed

'''

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

'''
batch_size = 128

seed = 42

shape = (64, 64, 3) ##desired shape of the image for resizing purposes

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
nlabls = top100class_train['landmark_id'].nunique()
def identity_block(X, f, filters, stage, block):

    

    conv_name_base = 'res' + str(stage) + block + '_branch'

    bn_name_base = 'bn' + str(stage) + block + '_branch'

    

    F1, F2, F3 = filters

    

    X_shortcut = X

        

    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)

    X = Activation('relu')(X)

        

    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)

    X = Activation('relu')(X)



    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)



    # Add shortcut value to main path

    X = Add()([X_shortcut, X])

    X = Activation('relu')(X)

        

    return X
def convolutional_block(X, f, filters, stage, block, s = 2):

        

    conv_name_base = 'res' + str(stage) + block + '_branch'

    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)

    X = Activation('relu')(X)

    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)

    X = Activation('relu')(X)

    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    X_shortcut = Conv2D(filters = F3, kernel_size = (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)

    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    X = Add()([X_shortcut, X])

    X = Activation('relu')(X)

   

    return X
def ResNet50(input_shape = (64, 64, 3), classes = nlabls):

    X_input = Input(input_shape)

    X = ZeroPadding2D((3, 3))(X_input)

    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)

    X = Activation('relu')(X)

    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)

    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')

    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)

    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')

    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')

    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)

    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')

    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')

    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')

    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')

    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)

    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')

    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    X = AveragePooling2D(pool_size=(2, 2),name='avg_pool')(X)

    X = Flatten()(X)

    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)

    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model
from tensorflow.keras.metrics import categorical_accuracy,top_k_categorical_accuracy

def top_5_accuracy(y_true, y_pred):

    return top_k_categorical_accuracy(y_true, y_pred, k=5)
model = ResNet50(input_shape = (64, 64, 3), classes = nlabls)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc',top_5_accuracy])

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
epochs = 2

use_multiprocessing = True 

#workers = 1 
base_cnn = model.fit_generator(

    train_generator,

    steps_per_epoch=len(train_generator),

    validation_data=val_generator,

    validation_steps=64,

    #class_weight = class_weights,

    epochs=epochs,

    #callbacks = [clr],

    use_multiprocessing=use_multiprocessing,

    #workers=workers,

    verbose=1)
model.save('ResNet50.h5')