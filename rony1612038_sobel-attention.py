!pip install ipython-autotime

 

%load_ext autotime
import os

import pandas as pd

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()

from PIL import Image

from glob import glob

from skimage.io import imread

from os import listdir

from sklearn.preprocessing import LabelEncoder

import time

import cv2

import copy

from random import shuffle

from tqdm import tqdm_notebook as tqdm

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import plot_roc_curve

from sklearn.metrics import precision_recall_fscore_support

from imblearn.metrics import sensitivity_specificity_support

from imgaug import augmenters as iaa

import imgaug as ia





# import numpy as np

# import matplotlib.pyplot as plt

from itertools import cycle



# from sklearn import svm, datasets

from sklearn.metrics import roc_curve, auc

# from sklearn.model_selection import train_test_split

# from sklearn.preprocessing import label_binarize

# from sklearn.multiclass import OneVsRestClassifier

from scipy import interp

from sklearn.metrics import roc_auc_score



from keras.applications.resnet50 import ResNet50, preprocess_input

from keras.utils.vis_utils import plot_model

from keras.optimizers import SGD,Adam

import numpy as np

from keras.applications.vgg16 import VGG16

from keras.layers import Dense, GlobalAveragePooling2D, Dropout,concatenate

from keras.layers import Conv2D, MaxPooling2D, Input, Flatten, BatchNormalization

from keras.layers import Input

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard,CSVLogger

# import tools

import gc

from sklearn.metrics import precision_score,recall_score,f1_score,confusion_matrix

from keras.models import Model

import keras

# import channel_attention
folder = os.listdir("../input/lung-colon-sobel/trainable_sobel")

print(folder)
base_path = "../input/lung-colon-sobel/trainable_sobel"

total_images = 0

image_class =[]

for n in range(len(folder)):

  image_path = os.path.join(base_path, folder[n]) 

  print(image_path)

  # class_path = patient_path + "/" + str(c) + "/"

  subfiles = os.listdir(image_path)

  print(len(subfiles))

  image_class.append(len(subfiles))

  total_images += len(subfiles)

print("The number of total images are:{}".format(total_images))  

print(image_class)
data = pd.DataFrame(index=np.arange(0, total_images), columns=["path", "target"])



k = 0

for n in range(len(folder)):

    class_id = folder[n]

    final_path = os.path.join(base_path,class_id) 

    subfiles = os.listdir(final_path)

    for m in range(len(subfiles)):

      image_path = subfiles[m]

      data.iloc[k]["path"] = os.path.join(final_path,image_path)

      data.iloc[k]["target"] = class_id

      k += 1  



data.head()
data['target'].unique()
# creating instance of labelencoder

labelencoder = LabelEncoder()

# Assigning numerical values and storing in another column

data['target_label'] = labelencoder.fit_transform(data['target'])

data = data.sample(frac=1).reset_index(drop=True)

data
#data.iloc[1000,:]
#data.groupby("target_label").size()
# cancer_perc = data.groupby("patient_id").target.value_counts()/ data.groupby("patient_id").target.size()

# cancer_perc = cancer_perc.unstack()



fig, ax = plt.subplots(1,2,figsize=(20,5))

sns.distplot(data.groupby("target_label").size(), ax=ax[0], color="Orange", kde=False)

ax[0].set_xlabel("Number of images")

ax[0].set_ylabel("Frequency");



sns.countplot(data.target, palette="Set2", ax=ax[1]);

ax[1].set_xlabel("Names of Class")

ax[1].set_title("Data Distribution");
#data.info()
#data.describe()
#data.target_label
X = data.path

y = data.target_label

X_train, X_test_sub ,y_train,y_test_sub= train_test_split(X,y, test_size=0.3, random_state=0,shuffle = True)

print(X_train.shape)

print(X_test_sub.shape)
X_test,X_valid,y_test,y_valid = train_test_split(X_test_sub, y_test_sub, test_size=0.5, random_state=0 , shuffle =False)

print(X_test.shape)

print(X_valid.shape)
fig, ax = plt.subplots(1,3,figsize=(20,5))

sns.countplot(y_train, ax=ax[0], palette="Reds")

ax[0].set_title("Train data")

sns.countplot(y_valid, ax=ax[1], palette="Blues")

ax[1].set_title("Dev data")

sns.countplot(y_test, ax=ax[2], palette="Greens");

ax[2].set_title("Test data");
data.path.values
target_label_map = {k:v for k,v in zip(data.path.values,data.target_label.values)}
def chunker(seq, size):

    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def get_seq():

    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    seq = iaa.Sequential(

        [

            # apply the following augmenters to most images

            iaa.Fliplr(0.5), # horizontally flip 50% of all images

            iaa.Flipud(0.2), # vertically flip 20% of all images

            sometimes(iaa.Affine(

                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, # scale images to 80-120% of their size, individually per axis

                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # translate by -20 to +20 percent (per axis)

                rotate=(-10, 10), # rotate by -45 to +45 degrees

                shear=(-5, 5), # shear by -16 to +16 degrees

                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)

                cval=(0, 255), # if mode is constant, use a cval between 0 and 255

                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)

            )),

            # execute 0 to 5 of the following (less important) augmenters per image

            # don't execute all of them, as that would often be way too strong

            iaa.SomeOf((0, 5),

                [

                    sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation

                    iaa.OneOf([

                        iaa.GaussianBlur((0, 1.0)), # blur images with a sigma between 0 and 3.0

                        iaa.AverageBlur(k=(3, 5)), # blur image using local means with kernel sizes between 2 and 7

                        iaa.MedianBlur(k=(3, 5)), # blur image using local medians with kernel sizes between 2 and 7

                    ]),

                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.9, 1.1)), # sharpen images

                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images

                    # search either for all edges or for directed edges,

                    # blend the result with the original image using a blobby mask

                    iaa.SimplexNoiseAlpha(iaa.OneOf([

                        iaa.EdgeDetect(alpha=(0.5, 1.0)),

                        iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),

                    ])),

                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01*255), per_channel=0.5), # add gaussian noise to images

                    iaa.OneOf([

                        iaa.Dropout((0.01, 0.05), per_channel=0.5), # randomly remove up to 10% of the pixels

                        iaa.CoarseDropout((0.01, 0.03), size_percent=(0.01, 0.02), per_channel=0.2),

                    ]),

                    iaa.Invert(0.01, per_channel=True), # invert color channels

                    iaa.Add((-2, 2), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)

                    iaa.AddToHueAndSaturation((-1, 1)), # change hue and saturation

                    # either change the brightness of the whole image (sometimes

                    # per channel) or change the brightness of subareas

                    iaa.OneOf([

                        iaa.Multiply((0.9, 1.1), per_channel=0.5),

                        iaa.FrequencyNoiseAlpha(

                            exponent=(-1, 0),

                            first=iaa.Multiply((0.9, 1.1), per_channel=True),

                            second=iaa.ContrastNormalization((0.9, 1.1))

                        )

                    ]),

                    sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)

                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around

                    sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))

                ],

                random_order=True

            )

        ],

        random_order=True

    )

    return seq



def data_gen(list_files, target_label_map, batch_size, augment=False):

    seq = get_seq()

    while True:

        # shuffle(list_files)

        for batch in chunker(list_files, batch_size):

            X = [cv2.resize(cv2.imread(x),(224,224),interpolation=cv2.INTER_CUBIC) for x in batch]

            # for x in X:

            #   X.append(cv2.resize(x,(224,224),interpolation=cv2.INTER_CUBIC))

            # X = [cv2.resize(x,(224,224,3)) for x in X]

            Y = [target_label_map[x] for x in batch]

            # print(Y)

            Y = to_categorical(Y, num_classes = 5)

            # print(Y)

            if augment:

                X = seq.augment_images(X)

            X = [preprocess_input(x) for x in X]

                

            yield np.array(X), np.array(Y)

    
from keras.layers import Activation, Reshape, Lambda, dot, add

from keras.layers import Conv1D, Conv2D, Conv3D

from keras.layers import MaxPool1D,GlobalAveragePooling2D,Dense,multiply,Activation,concatenate

from keras import backend as K





def squeeze_excitation_layer(x, out_dim, ratio = 4, concate = True):

    '''

    SE module performs inter-channel weighting.

    '''

    squeeze = GlobalAveragePooling2D()(x)



    excitation = Dense(units=out_dim //ratio)(squeeze)

    excitation = Activation('relu')(excitation)

    excitation = Dense(units=out_dim)(excitation)

    excitation = Activation('sigmoid')(excitation)

    print(excitation.shape)

    excitation = Reshape((1, 1, out_dim))(excitation)



    scale = multiply([x, excitation])



    if concate:

        scale = concatenate([scale, x],axis=3)

    return scale




adam = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0009)

sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)



input_tensor = Input(shape=(224,224, 3))

#backbone

base_model = ResNet50(input_tensor= input_tensor, weights='imagenet', include_top=False)

base_output = base_model.output

print(base_output.shape)

# channel-attention

x = squeeze_excitation_layer(base_output, 2048, ratio=4, concate=False)

x = BatchNormalization()(x)



# #concat

x = concatenate([base_output, x], axis=3)

# spp



gap = GlobalAveragePooling2D()(x)

x = Flatten()(x)

x = concatenate([gap,x])

x = Dense(512, activation='relu')(x)

x = BatchNormalization()(x)

x = Dense(512, activation='relu')(x)

x = BatchNormalization()(x)

predict = Dense(5, activation='softmax')(x)

model = Model(inputs=input_tensor, outputs=predict)



for layer in (base_model.layers):

    layer.trainable = False



model.compile(optimizer=adam,

                      loss='categorical_crossentropy',

                      metrics=[keras.metrics.categorical_accuracy])    



# for l in model.layers:

#   print(l.name)


model.summary()

batch_size=32

history = model.fit_generator(

    data_gen(X_train, target_label_map, batch_size, augment=True),

    validation_data=data_gen(X_valid, target_label_map, batch_size),

    epochs=50, 

    verbose = 1,

    #callbacks=callbacks,

    steps_per_epoch=  int(len(X_train)//batch_size),

    validation_steps= int(len(X_valid)// batch_size)

)
import tensorflow as tf

from tensorflow.keras.models import Sequential, save_model, load_model



filepath = './'
tf.keras.models.save_model(

    model,

    filepath,

    overwrite=True,

    include_optimizer=True,

    save_format=None,

    signatures=None,

    options=None

)