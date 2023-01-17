#Import packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Model, Sequential

from keras.applications.mobilenet import MobileNet

from keras.applications.xception import Xception

from keras.applications.vgg16 import VGG16

from keras.layers import Input, Dense, GlobalAveragePooling2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator
AL0625_1_dam = pd.read_csv('../input/shmxinyao/AL0625_1_dam.csv', header=None)

AL0625_1_dam.shape 
sLength = 10000

AL0625_1_dam['shm'] = pd.Series(np.ones(sLength)) #damaged1:undamaged0
AL0625_1_dam['material'] = pd.Series(np.ones(sLength)) #material 1
AL0625_2_dam = pd.read_csv('../input/shmxinyao/AL0625_2_dam.csv',header=None)

AL0625_2_dam['shm'] = pd.Series(np.ones(sLength))

AL0625_2_dam['material'] = pd.Series(2*np.ones(sLength))  #material 2



AL25_dam = pd.read_csv('../input/shmxinyao/AL25_dam.csv',header=None)

AL25_dam['shm'] = pd.Series(np.ones(sLength))

AL25_dam['material']= pd.Series(3*np.ones(sLength))  #material 3



ST0625_dam = pd.read_csv('../input/shmxinyao/ST0625_dam.csv',header=None)

ST0625_dam['shm'] = pd.Series(np.ones(sLength))

ST0625_dam['material']= pd.Series(4*np.ones(sLength))  #material 4



AL0625_1_undam = pd.read_csv('../input/shmxinyao/AL0625_1_undam.csv',header=None)

AL0625_1_undam['shm'] = pd.Series(np.zeros(sLength)) 

AL0625_1_undam['material'] = pd.Series(np.ones(sLength)) #material 1



AL0625_2_undam = pd.read_csv('../input/shmxinyao/AL0625_2_undam.csv',header=None)

AL0625_2_undam['shm'] = pd.Series(np.zeros(sLength))

AL0625_2_undam['material'] = pd.Series(2*np.ones(sLength))  #material 2



AL25_undam = pd.read_csv('../input/shmxinyao/AL25_undam.csv',header=None)

AL25_undam['shm'] = pd.Series(np.zeros(sLength))

AL25_undam['material']= pd.Series(3*np.ones(sLength))  #material 3



ST0625_undam = pd.read_csv('../input/shmxinyao/ST0625_undam.csv',header=None)

ST0625_undam['shm'] = pd.Series(np.zeros(sLength))

ST0625_undam['material']= pd.Series(4*np.ones(sLength))  #material 4
train = (pd.concat([AL0625_1_dam,AL0625_2_dam,AL25_dam,ST0625_dam,

        AL0625_1_undam,AL0625_2_undam,AL25_undam,ST0625_undam],ignore_index=True))



y_train = train['shm']

y_train = to_categorical(y_train, num_classes = 2)

feature = train['material']

x_train = train.drop(labels=['shm','material'],axis=1)

x_train = x_train.values.reshape(-1,50,60,1)



x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
input_tensor = Input(shape=(50,60, 1)) 
base_model = Xception(weights=None,include_top = False,input_tensor=input_tensor)

base_model.load_weights('../input/xception-weights-tf-dim-ordering-tf-kernels-notop/xception_weights_tf_dim_ordering_tf_kernels_notop.h5',by_name=True)

im = base_model.output

im = GlobalAveragePooling2D()(im)

predictions = Dense(2, activation='softmax')(im)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=["accuracy"])
datagen = ImageDataGenerator(featurewise_center=True,featurewise_std_normalization=True)

datagen.fit(x_train)

model.fit_generator(datagen.flow(x_train, y_train, batch_size=86),steps_per_epoch=len(x_train) / 86, epochs=5, shuffle=True)

score = model.evaluate(x_test, y_test, batch_size=86)
print(score)