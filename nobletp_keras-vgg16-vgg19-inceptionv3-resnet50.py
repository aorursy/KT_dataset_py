# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import openslide

import os

import cv2

import PIL

from IPython.display import Image, display

from keras.applications.vgg16 import VGG16,preprocess_input

# Plotly for the interactive viewer (see last section)

import plotly.graph_objs as go

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

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau

from keras.utils import to_categorical

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.layers.normalization import BatchNormalization

import gc

import matplotlib.pyplot as plt

%inline matplotlib

import skimage.io

from sklearn.model_selection import KFold

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import tensorflow as tf

from tensorflow.python.keras import backend as K

sess = K.get_session()
from keras.callbacks.callbacks import ReduceLROnPlateau, EarlyStopping

import tensorflow as tf

from sklearn.preprocessing import LabelEncoder

from keras.models import load_model

from keras.applications.xception import Xception

import time

import seaborn as sns

from sklearn.metrics import roc_curve

from sklearn.metrics import auc


from numpy.random import seed

seed(1)
train=pd.read_csv('/kaggle/input/prostate-cancer-grade-assessment/train.csv')

train.head()

train_copy = train.copy()
img=openslide.OpenSlide('/kaggle/input/prostate-cancer-grade-assessment/train_images/2fd1c7dc4a0f3a546a59717d8e9d28c3.tiff')

display(img.get_thumbnail(size=(512,512)))
img.dimensions
patch = img.read_region((18500,4100), 0, (256, 256))



# Display the image

display(patch)

# Close the opened slide after use

img.close()
train['isup_grade'].value_counts()
train.head()
labels=[]

data=[]

data_dir='/kaggle/input/panda-resized-train-data-512x512/train_images/train_images/'

for i in range(train.shape[0]):

    data.append(data_dir + train['image_id'].iloc[i]+'.png')

    labels.append(train['isup_grade'].iloc[i])

df=pd.DataFrame(data)

df.columns=['images']

df['isup_grade']=labels
df.head()
X_train, X_val, y_train, y_val = train_test_split(df['images'],df['isup_grade'], test_size=0.1, random_state=42)

train=pd.DataFrame(X_train)

train.columns=['images']

train['isup_grade']=y_train



validation=pd.DataFrame(X_val)

validation.columns=['images']

validation['isup_grade']=y_val



train['isup_grade']=train['isup_grade'].astype(str)

validation['isup_grade']=validation['isup_grade'].astype(str)
#Image data generator generates varied images with the input data for better prediction for the models

#shear is avoided since the cancerous cells look like sheared normal cells
train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=40,

    featurewise_center=True,

    featurewise_std_normalization=True,

    #zoom_range=[0.8, 1.2],        

    horizontal_flip=True, vertical_flip = True,

    brightness_range=[0.9, 1.1],

    width_shift_range=1.0,

    height_shift_range=1.0)#,

    #validation_split=0.1)





val_datagen=train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(

    train,

    x_col='images',

    y_col='isup_grade',

    target_size=(224, 224),

    batch_size=32,

    shuffle = True,

    class_mode='categorical')



validation_generator = val_datagen.flow_from_dataframe(

    validation,

    x_col='images',

    y_col='isup_grade',

    target_size=(224, 224),

    batch_size=32,

    class_mode='categorical')
filenames = validation_generator.filenames

nb_samples = len(filenames)



#y_true = validation_generator.classes



#y_true
#VGG16 model
def vgg16_model( num_classes=None):



    model = VGG16(weights='/kaggle/input/keras-pretrained-models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, input_shape=(224, 224, 3))

    #x=Dropout(0.2)(model.output)

    x=Flatten()(model.output)

    #x =Dense(32, activation = 'relu')(x)

    output=Dense(num_classes,activation='softmax')(x)

    model=Model(model.input,output)

    return model



#vgg_conv=vgg16_model(6)
#VGG19 model-- Uncomment last line to run the model
from keras.applications.vgg19 import VGG19

def vgg19_model(num_classes = None):

    #vgg19_weights = '../input/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels.h5'

    model = VGG19(weights='imagenet',include_top=False, input_shape=(224, 224, 3))

    #model = VGG19(weights='/input/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels.h5', include_top=False, input_shape=(224, 224, 3))

    x=Dropout(0.3)(model.output)

    x=Flatten()(x)

    x =Dense(32, activation = 'relu')(x)

    x =Dropout(0.2)(x)

    output=Dense(num_classes,activation='softmax')(x)

    model=Model(model.input,output)

    return model

#vgg19_conv = vgg19_model(6)
#InceptionV3 model- Uncomment last line to run the model
from keras.applications.inception_v3 import InceptionV3

def InceptionV3_model(num_classes = None):

    InceptionV3_weights = '../input/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

    #model = ResNet50(weights='imagenet', include_top = False, input_shape = (224,224,3))

    model = InceptionV3(weights= InceptionV3_weights, include_top=False, input_shape=(224, 224, 3))

    #model = InceptionV3(weights='imagenet', include_top = False, input_shape = (224,224,3))

    x=Dropout(0.3)(model.output)

    #x=Flatten()(model.output)

    #x =Dense(64, activation = 'relu')(model.output)

    #x = tf.compat.v1.keras.layers.GlobalAveragePooling2D()(model.output)

    x=Flatten()(x)

    #x =Dropout(0.2)(x)

    x =Dense(32, activation = 'relu')(x)

    x =Dropout(0.2)(x)

    output=Dense(num_classes,activation='softmax')(x)

    model=Model(model.input,output)

    return model

InceptionV3_conv = InceptionV3_model(6)
#Resnet50 model- Uncomment last line to run the model
#!pip install tensorflow==2.1.0
from keras.applications.resnet50 import ResNet50

def ResNet50_model(num_classes = None):

    ResNet_weights = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

    #model = ResNet50(weights='imagenet', include_top = False, input_shape = (224,224,3))

    model = ResNet50(weights= ResNet_weights, include_top=False, input_shape=(224, 224, 3))

    #x=Dropout(0.2)(model.output)

    #x = GlobalAveragePooling2D()(model.output)

    x=Flatten()(model.output)

    #x =Dropout(0.2)(x)

    x =Dense(16, activation = 'relu')(x)

    x =Dropout(0.2)(x)

    output=Dense(num_classes,activation='softmax')(x)

    model=Model(model.input,output)

    return model

#ResNet50_conv = ResNet50_model(6)
from keras.applications import InceptionResNetV2

def InceptionResnet_model(num_classes = None):

    InceptionResnet_weights = '../input/inceptionresnetv2/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5'

    #model = inception_resnet_v2(weights='imagenet', include_top = False, input_shape = (224,224,3))

    model = InceptionResNetV2(weights= InceptionResnet_weights, include_top=False, input_shape=(224, 224, 3))

    #x=Dropout(0.2)(model.output)

    #x = GlobalAveragePooling2D()(model.output)

    x=Flatten()(model.output)

    #x =Dropout(0.2)(x)

    x =Dense(16, activation = 'relu')(x)

    x =Dropout(0.2)(x)

    output=Dense(num_classes,activation='softmax')(x)

    model=Model(model.input,output)

    return model

#InceptionResnet_conv = InceptionResnet_model(6)
#Uncomment any one to run the corresponding summary
#vgg_conv.summary()
#vgg19_conv.summary()


InceptionV3_conv.summary()
#ResNet50_conv.summary()
#InceptionResnet_conv.summary()
#Including batch Normalization
def freeze(model):

    # freeze layers before 99

    for layer in model.layers[:99]:

        layer.trainable = False

        if layer.name.startswith('batch_normalization'):

            layer.trainable = True

        if layer.name.endswith('bn'):

            layer.trainable = True



    # unfreeze layers after 99

    for layer in model.layers[99:]:

        layer.trainable = True

        



        

        

        
#freeze(vgg16_conv)

#freeze(vgg19_conv)

#freeze(InceptionV3_conv)

#freeze(ResNet50_conv)
#y_pred = np.argmax(InceptionV3_conv.predict_generator(validation_generator, steps= len(validation_generator)), axis=1)
#kappa score is the metric we use since two types of evaluation is done on the data set 



#The kappa statistic.

#According to Cohen's original article, 

#values ≤ 0 as indicating no agreement and 0.01–0.20 as none to slight,

#0.21–0.40 as fair, 0.41– 0.60 as moderate,

#0.61–0.80 as substantial, 

#0.81–1.00 as almost perfect agreement.
def kappa_score(y_true, y_pred):

    

    y_true=tf.math.argmax(y_true)

    y_pred=tf.math.argmax(y_pred)

    #return tf.compat.v1.py_func(cohen_kappa_score,(y_true, y_pred),tf.double)

    return tf.compat.v1.py_func(cohen_kappa_score,(y_true,y_pred),tf.double)

    #return (y_true,y_pred)
import numpy as np

from sklearn.metrics import cohen_kappa_score

from sklearn.metrics import confusion_matrix



import keras.backend as K

import tensorflow as tf





def kappa_keras(y_true, y_pred):



    y_true = K.cast(K.argmax(y_true, axis=-1), dtype='int32')

    y_pred = K.cast(K.argmax(y_pred, axis=-1), dtype='int32')

    #print(y_true)

    #print(y_pred)

    # Figure out normalized expected values

    min_rating = K.minimum(K.min(y_true), K.min(y_pred))

    max_rating = K.maximum(K.max(y_true), K.max(y_pred))



    # shift the values so that the lowest value is 0

    # (to support scales that include negative values)

    y_true = K.map_fn(lambda y: y - min_rating, y_true, dtype='int32')

    y_pred = K.map_fn(lambda y: y - min_rating, y_pred, dtype='int32')



    # Build the observed/confusion matrix

    num_ratings = max_rating - min_rating + 1

    observed = tf.math.confusion_matrix(y_true, y_pred,

                                num_classes=num_ratings)

    num_scored_items = K.shape(y_true)[0]



    weights = K.expand_dims(K.arange(num_ratings), axis=-1) - K.expand_dims(K.arange(num_ratings), axis=0)

    weights = K.cast(K.pow(weights, 2), dtype='float64')



    hist_true = tf.math.bincount(y_true, minlength=num_ratings)

    hist_true = hist_true[:num_ratings] / num_scored_items

    hist_pred = tf.math.bincount(y_pred, minlength=num_ratings)

    hist_pred = hist_pred[:num_ratings] / num_scored_items

    expected = K.dot(K.expand_dims(hist_true, axis=-1), K.expand_dims(hist_pred, axis=0))



    # Normalize observed array

    observed = observed / num_scored_items



    # If all weights are zero, that means no disagreements matter.

    score = tf.where(K.any(K.not_equal(weights, 0)), 

                     K.sum(weights * observed) / K.sum(weights * expected), 

                     0)

    

    return 1. - score



if __name__ == '__main__':

    y_true = np.array([2, 0, 2, 2, 0, 1])

    y_pred = np.array([0, 0, 2, 2, 0, 2])

    # Testing Keras implementation of QWK

    

    # Calculating QWK score with scikit-learn

   

    skl_score = cohen_kappa_score(y_true, y_pred, weights='quadratic')

    

    # Keras implementation of QWK work with one hot encoding labels and predictions (also it works with softmax probabilities)

    # Converting arrays to one hot encoded representation

    shape = (y_true.shape[0], np.maximum(y_true.max(), y_pred.max()) + 1)



    y_true_ohe = np.zeros(shape)

    y_true_ohe[np.arange(shape[0]), y_true] = 1



    y_pred_ohe = np.zeros(shape)

    y_pred_ohe[np.arange(shape[0]), y_pred] = 1

    

    # Calculating QWK score with Keras

    with tf.compat.v1.Session() as sess:

        keras_score = kappa_keras(y_true_ohe, y_pred_ohe).eval()

    

    #print('Scikit-learn score: {:.03}, Keras score: {:.03}'.format(skl_score, keras_score))

    
opt = SGD(lr= 0.0005, momentum=0.9,decay=1e-4)

#vgg_conv.compile(loss='binary_crossentropy',optimizer=opt,metrics=[kappa_keras, 'accuracy'])

#vgg19_conv.compile(loss='binary_crossentropy',optimizer=opt,metrics=[kappa_keras, 'accuracy'])

InceptionV3_conv.compile(loss='binary_crossentropy',optimizer=opt,metrics=[kappa_score,kappa_keras,'accuracy'])

#ResNet50_conv.compile(loss='binary_crossentropy',optimizer=opt,metrics=[kappa_score,kappa_keras, 'accuracy'])

#InceptionResnet_conv.compile(loss='binary_crossentropy',optimizer=opt,metrics=[kappa_score,kappa_keras, 'accuracy'])
nb_epochs = 4

batch_size=16

nb_train_steps = train.shape[0]//batch_size

nb_val_steps=validation.shape[0]//batch_size

#nb_train_steps = 128

#nb_val_steps = 64

print("Number of training and validation steps: {} and {}".format(nb_train_steps,nb_val_steps))
#vgg_conv.fit_generator( train_generator,steps_per_epoch=nb_train_steps,epochs=10,validation_data=validation_generator,

#validation_steps=nb_val_steps),

#generator to activate the augmentation.

callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=1, verbose=1, factor=0.5),

             EarlyStopping(monitor='val_loss', patience=3),

             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
#vgg19_conv.fit_generator( train_generator,steps_per_epoch=nb_train_steps,epochs=5,validation_data=validation_generator,validation_steps=nb_val_steps,callbacks = callbacks)
history =InceptionV3_conv.fit_generator( train_generator,steps_per_epoch=nb_train_steps,epochs=nb_epochs,validation_data=validation_generator,

validation_steps=nb_val_steps, callbacks = callbacks)
#ResNet50_conv.fit_generator( train_generator,steps_per_epoch=nb_train_steps,epochs=20,validation_data=validation_generator,

#validation_steps=nb_val_steps, callbacks = callbacks)
#history = InceptionResnet_conv.fit_generator( train_generator,steps_per_epoch=nb_train_steps,epochs=2,validation_data=validation_generator,

#validation_steps=nb_val_steps, callbacks = callbacks)
def show_history(history):

    fig, ax = plt.subplots(1, 3, figsize=(15,5))

    ax[0].set_title('loss')

    ax[0].plot(history.epoch, history.history["loss"], label="Train loss")

    ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")

    ax[1].set_title('kappa_keras')

    ax[1].plot(history.epoch, history.history["kappa_keras"], label="Train Quadratic Kappa score")

    ax[1].plot(history.epoch, history.history["val_kappa_keras"], label="Validation Quadratic Kappa score")

    ax[2].set_title('accuracy')

    ax[2].plot(history.epoch, history.history["accuracy"], label="Train acc")

    ax[2].plot(history.epoch, history.history["val_accuracy"], label="Validation accuracy")

    ax[0].legend()

    ax[1].legend()

    ax[2].legend()
show_history(history)
#vgg_baseline = Model.save('VGG16_Baseline.h5')  # creates a HDF5 file 'my_model.h5'

InceptionV3_baseline =InceptionV3_conv.save('InceptionV3_Baseline.h5')  # creates a HDF5 file 'my_model.h5'

#Resnet50_baseline =ResNet50_conv.save('ResNet50_Baseline.h5')  # creates a HDF5 file 'my_model.h5'

#InceptionResnet_baseline =InceptionResnet_conv.save('InceptionResnet_Baseline.h5')  # creates a HDF5 file 'my_model.h5'
#vgg_baseline_weights = Model.save_weights('vgg_baseline_weights.h5')

InceptionV3_weights =InceptionV3_conv.save_weights('InceptionV3_weights.h5')  # creates a HDF5 file 'my_model.h5'

#Resnet50_weights =ResNet50_conv.save_weights('Resnet50_weights.h5')  # creates a HDF5 file 'my_model.h5'

#InceptionResnet_weights =InceptionResnet_conv.save_weights('InceptionResnet_weights.h5')  # creates a HDF5 file 'my_model.h5'
os.listdir('.')
InceptionV3_conv.load_weights("best_model.h5")

#InceptionResnet_conv.load_weights("best_model.h5")
def predict_submission(df, path, passes=1):

    

    df["image_path"] = [path+image_id+".tiff" for image_id in df["image_id"]]

    df["isup_grade"] = 0

    

    for idx, row in df.iterrows():

        prediction_per_pass = []

        for i in range(passes):

            model_input = np.array([get_random_samples(row.image_path)/255.])

            input_image1 = model_input[:,0,:,:]

            input_image2 = model_input[:,1,:,:]

            input_image3 = model_input[:,2,:,:]



            prediction = InceptionResnet_conv.predict([input_image1,input_image2,input_image3])

            prediction_per_pass.append(np.argmax(prediction))

            

        df.at[idx,"isup_grade"] = np.mean(prediction_per_pass)

    df = df.drop('image_path', 1)

    return df[["image_id","isup_grade"]]
# submission code from https://www.kaggle.com/frlemarchand/high-res-samples-into-multi-input-cnn-keras

def predict_submission(df, path):

    

    df["image_path"] = [path+image_id+".tiff" for image_id in df["image_id"]]

    df["isup_grade"] = 0

    predictions = []

    for idx, row in df.iterrows():

        print(row.image_path)

        img=skimage.io.imread(str(row.image_path))

        img = cv2.resize(img, (224,224))

        img = cv2.resize(img, (224,224))

        img = img.astype(np.float32)/255.

        img=np.reshape(img,(1,224,224,3))

       

    

        prediction=InceptionResnet_conv.predict(img)

        predictions.append(np.argmax(prediction))

            

    df["isup_grade"] = predictions

    df = df.drop('image_path', 1)

    return df[["image_id","isup_grade"]]
test_path = "../input/prostate-cancer-grade-assessment/test_images/"

submission_df = pd.read_csv("../input/prostate-cancer-grade-assessment/sample_submission.csv")



if os.path.exists(test_path):

    test_df = pd.read_csv("../input/prostate-cancer-grade-assessment/test.csv")

    submission_df = predict_submission(test_df, test_path)

else:

    print('submission csv not found')



    

del InceptionV3_conv

submission_df.to_csv('submission.csv', index=False)

submission_df.head()