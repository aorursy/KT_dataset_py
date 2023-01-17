# Ignore  the warnings

import warnings

warnings.filterwarnings('always')

warnings.filterwarnings('ignore')



# data visualisation and manipulation

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns

 

#configure

# sets matplotlib to inline and displays graphs below the corressponding cell.

%matplotlib inline  

style.use('fivethirtyeight')

sns.set(style='whitegrid',color_codes=True)



#model selection

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import LabelEncoder



#preprocess.

from keras.preprocessing.image import ImageDataGenerator



#dl libraraies

from keras import backend as K

from keras import regularizers

from keras.models import Sequential

from keras.models import Model

from keras.layers import Dense

from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop

from keras.utils import to_categorical

from keras.callbacks import ReduceLROnPlateau

from keras.callbacks import ModelCheckpoint





# specifically for cnn

from keras.layers import Dropout, Flatten,Activation

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from keras.layers import InputLayer

 

import tensorflow as tf

import random as rn



# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.

import cv2                  

import numpy as np  

from tqdm import tqdm

import os                   

from random import shuffle  

from zipfile import ZipFile

from PIL import Image



from keras.applications.resnet50 import ResNet50
def assign_label(img,label):

    return label
def make_train_data(label,DIR):

    for img in tqdm(os.listdir(DIR)):

        path = os.path.join(DIR,img)

        img = cv2.imread(path,cv2.IMREAD_COLOR)

        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))

        

        X.append(np.array(img))

        Z.append(str(label))
X=[]

Z=[]

IMG_SIZE=224

NOTFIRE='../input/fire-detection-cleaned/non_fire'

FIRE='../input/fire-detection-cleaned/fire'



make_train_data('NOTFIRE',NOTFIRE)

make_train_data('FIRE',FIRE)
le=LabelEncoder()

Y=le.fit_transform(Z)

Y=to_categorical(Y,2)

print(Y)

X=np.array(X)

X=X/255



x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=1337)



np.random.seed(42)

rn.seed(42)

#tf.set_random_seed(42)
fig,ax=plt.subplots(2,5)

plt.subplots_adjust(bottom=0.3, top=0.7, hspace=0)

fig.set_size_inches(10,10)



for i in range(2):

    for j in range (5):

        l=rn.randint(0,len(Z))

        ax[i,j].grid(False)

        ax[i,j].imshow(X[l][:,:,::-1])

        ax[i,j].set_title(Z[l])

        ax[i,j].set_aspect('equal')
base_model=ResNet50(include_top=False, weights=None,input_shape=(224,224,3), pooling='max')

#base_model=ResNet50(include_top=False, weights='imagenet',input_shape=(224,224,3), pooling='max')
model=Sequential()

model.add(base_model)

model.add(Dropout(0.20))

model.add(Dense(2048,activation='relu'))

model.add(Dense(1024,activation='relu'))

model.add(Dense(512,activation='relu'))

model.add(Dense(2,activation='softmax'))
epochs=250

batch_size=32

red_lr=ReduceLROnPlateau(monitor='val_acc', factor=0.5, min_lr='10e-6', patience=5, verbose=1)



# checkpoint

filepath="/kaggle/working/weights-improvement.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

callbacks_list = [checkpoint,red_lr]



base_model.trainable=True # setting the Resnet50 model to be trainable.



model.compile(optimizer=Adam(lr=1e-5),loss='binary_crossentropy',metrics=['accuracy'])

model.summary()
History = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test,y_test), callbacks=callbacks_list)
plt.plot(History.history['accuracy'])

plt.plot(History.history['val_accuracy'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epochs')

plt.legend(['train', 'test'])

plt.show()
plt.plot(History.history['loss'])

plt.plot(History.history['val_loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epochs')

plt.legend(['train', 'test'])

plt.show()