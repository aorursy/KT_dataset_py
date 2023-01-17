import tensorflow

print(tensorflow.__version__)
!pip install livelossplot
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import PIL

from IPython.display import Image, display

from keras.applications.vgg16 import VGG16,preprocess_input

# Plotly for the interactive viewer (see last section)

import plotly.graph_objs as go

import plotly.graph_objects as go

from sklearn.metrics import cohen_kappa_score

from sklearn.model_selection import train_test_split

from keras.models import Sequential, Model,load_model

from keras.applications.vgg16 import VGG16,preprocess_input

from keras.applications.resnet50 import ResNet50

from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten,BatchNormalization,Activation

from keras.layers import GlobalMaxPooling2D

from keras.models import Model

from keras.optimizers import Adam, SGD, RMSprop

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

from keras.utils import to_categorical

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import gc

import skimage.io

import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow.python.keras import backend as K

from livelossplot import PlotLossesKeras
train_dir='/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'

test_dir='/kaggle/input/siim-isic-melanoma-classification/jpeg/test/'

train=pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')

test=pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv')

submission=pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')
train.head()
## Original train data

train['target'].value_counts()
## Undersampled train data

df_0=train[train['target']==0].sample(2000)

df_1=train[train['target']==1]

train=pd.concat([df_0,df_1])

train=train.reset_index()

train['target'].value_counts()
labels=[]

data=[]

for i in range(train.shape[0]):

    data.append(train_dir + train['image_name'].iloc[i]+'.jpg')

    labels.append(train['target'].iloc[i])

df=pd.DataFrame(data)

df.columns=['images']

df['target']=labels
test_data=[]

for i in range(test.shape[0]):

    test_data.append(test_dir + test['image_name'].iloc[i]+'.jpg')

df_test=pd.DataFrame(test_data)

df_test.columns=['images']
X_train, X_val, y_train, y_val = train_test_split(df['images'],df['target'], test_size=0.2, random_state=1234)



train=pd.DataFrame(X_train)

train.columns=['images']

train['target']=y_train



validation=pd.DataFrame(X_val)

validation.columns=['images']

validation['target']=y_val
train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2,horizontal_flip=True)



val_datagen=ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_dataframe(

    train,

    x_col='images',

    y_col='target',

    target_size=(224, 224),

    batch_size=8,

    shuffle=True,

    class_mode='raw')



validation_generator = val_datagen.flow_from_dataframe(

    validation,

    x_col='images',

    y_col='target',

    target_size=(224, 224),

    shuffle=False,

    batch_size=8,

    class_mode='raw')
def vgg16_model( num_classes=None):



    model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    x=Flatten()(model.output)

    output=Dense(1,activation='sigmoid')(x) # because we have to predict the AUC

    model=Model(model.input,output)

    

    return model



vgg_conv=vgg16_model(1)
def focal_loss(alpha=0.25,gamma=2.0):

    def focal_crossentropy(y_true, y_pred):

        bce = K.binary_crossentropy(y_true, y_pred)

        

        y_pred = K.clip(y_pred, K.epsilon(), 1.- K.epsilon())

        p_t = (y_true*y_pred) + ((1-y_true)*(1-y_pred))

        

        alpha_factor = 1

        modulating_factor = 1



        alpha_factor = y_true*alpha + ((1-alpha)*(1-y_true))

        modulating_factor = K.pow((1-p_t), gamma)



        # compute the final loss and return

        return K.mean(alpha_factor*modulating_factor*bce, axis=-1)

    return focal_crossentropy
opt = Adam(lr=1e-5)

vgg_conv.compile(loss=focal_loss(), metrics=[tf.keras.metrics.AUC()],optimizer=opt)
nb_epochs = 2

batch_size=8

nb_train_steps = train.shape[0]//batch_size

nb_val_steps=validation.shape[0]//batch_size

print("Number of training and validation steps: {} and {}".format(nb_train_steps,nb_val_steps))
cb = [PlotLossesKeras()]

vgg_conv.fit_generator(

    train_generator,

    steps_per_epoch=nb_train_steps,

    epochs=nb_epochs,

    validation_data=validation_generator,

    callbacks=cb,

    validation_steps=nb_val_steps)
target=[]

for path in df_test['images']:

    img=cv2.imread(str(path))

    img = cv2.resize(img, (224,224))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32)/255.

    img=np.reshape(img,(1,224,224,3))

    prediction=vgg_conv.predict(img)

    target.append(prediction[0][0])



submission['target']=target
submission.to_csv('submission.csv', index=False)

submission.head()