import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import Dropout
from keras.models import  Model
from keras import optimizers
from os import makedirs
from os.path import join, exists, expanduser
import keras
from keras.layers import BatchNormalization
import os
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
cache_dir = expanduser(join('~', '.keras'))
if not exists(cache_dir):
    makedirs(cache_dir)
models_dir = join(cache_dir, 'models')
if not exists(models_dir):
    makedirs(models_dir)
!cp  ../input/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 ~/.keras/models/
TRAIN_DIR = '../input/whale-categorization-playground/train/'
train_label = pd.read_csv('../input/whale-categorization-playground/train.csv')
train_labels = []
for name in os.listdir(TRAIN_DIR):
    index =  np.where(train_label.Image == name)
    train_labels.append(train_label.Id[index[0][0]])
X_train = np.zeros((len(os.listdir(TRAIN_DIR)), 224, 224, 3), dtype=np.int16)
for n, id_ in tqdm(enumerate(os.listdir(TRAIN_DIR)), total=len(os.listdir(TRAIN_DIR))):
    path = TRAIN_DIR + id_
    img = imread(path)
    if len(img.shape)==2:
        img = np.append(img.reshape(img.shape[0],img.shape[1],1),np.append((img.reshape(img.shape[0],img.shape[1],1)),(img.reshape(img.shape[0],img.shape[1],1)),axis=-1),axis=-1)
    img = resize(img, (224, 224), mode='constant', preserve_range=True)
    X_train[n] = img
class LabelOneHotEncoder():
    def __init__(self):
        self.ohe = OneHotEncoder()
        self.le = LabelEncoder()
    def fit_transform(self, x):
        features = self.le.fit_transform( x)
        return self.ohe.fit_transform( features.reshape(-1,1))
    def transform( self, x):
        return self.ohe.transform( self.la.transform( x.reshape(-1,1)))
    def inverse_tranform( self, x):
        return self.le.inverse_transform( self.ohe.inverse_tranform( x))
    def inverse_labels( self, x):
        return self.le.inverse_transform( x)
lohe = LabelOneHotEncoder()
y_cat = lohe.fit_transform(train_labels)
train_datagen = ImageDataGenerator(rotation_range=10, shear_range=0.2,zoom_range=0.2,vertical_flip=True)       
train_generator = train_datagen.flow(X_train,y_cat.toarray(),batch_size=32)
base_model=InceptionV3(include_top=False, weights='imagenet', input_shape=(224, 224, 3),classes=10)
# Adding custom Layers
x = base_model.output
x = Flatten()(x)
x = Dense(512, kernel_regularizer=keras.regularizers.l2(1e-5),activation='relu',name='fc-1')(x)
x = BatchNormalization()(x)
x = Dense(256, kernel_regularizer=keras.regularizers.l2(1e-5),activation='relu',name='fc-2')(x)
x = BatchNormalization()(x)
x = Dense(128, kernel_regularizer=keras.regularizers.l2(1e-5), activation='relu',name='fc-3')(x)
x = BatchNormalization()(x)
predictions = Dense(4251, kernel_regularizer=keras.regularizers.l2(1e-5),activation="softmax")(x)
# creating the final model
model_final = Model(input=base_model.input, output=predictions)
for layer in model_final.layers[:-7]:
    layer.trainable = False
model_final.layers[-7].trainable
#model_final.summary()
model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=0.01),metrics=["accuracy"])
model_final.fit_generator(train_generator,steps_per_epoch=100,epochs=5)
model_final.fit_generator(train_generator,steps_per_epoch=100,epochs=5)
model_final.fit_generator(train_generator,steps_per_epoch=100,epochs=5)
model_final.fit_generator(train_generator,steps_per_epoch=100,epochs=5)
model_final.save('incep_fish_round1.h5')