# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import sys
import numpy as np # linear algebra
import pandas as pd
import cv2# data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pickle
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
#from keras.models import load_model
#from tensorflow.keras.applications.vgg16 import VGG16
from keras.optimizers import Adam
#from efficientnet import efficientnet
from keras.models import load_model

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
#sys.path.insert(0,'/kaggle/input/kerasefficientnetb3')
sys.path.insert(0,'/kaggle/input/')
import os

'''
for dirname, _, filenames in os.walk('/kaggle/input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
'''
# Install EfficientNet
# Install EfficientNet
!pip install '../input/kerasefficientnetb3/efficientnet-1.0.0-py3-none-any.whl'

#!pip install '../input/kerasefficientnetb3/efficientnet-1.0.0-py3-none-any.whl'
#!pip install '../input/vgg16-weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
#vggmodel=load_model('../input/vggmodel/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
import efficientnet.keras as efn

#sys.path.insert(0,'../input/kerasefficientnetb3')
os.listdir('../input/')
train=pd.read_csv('/kaggle/input/auto-tag-gala/dataset/train.csv')
test=pd.read_csv('/kaggle/input/auto-tag-gala/dataset/test.csv')
train.shape
train.head()
train_dir='/kaggle/input/auto-tag-gala/dataset/Train Images/'
test_dir='/kaggle/input/auto-tag-gala/dataset/Test Images/'
## train df prep ::
tr_df=[]
imgage_size=150
for img in train.Image:
    path=os.path.join(train_dir,img)
    try :
        
        img_array=cv2.imread(path,cv2.IMREAD_COLOR)
        new_array=cv2.resize(img_array,(imgage_size,imgage_size))
        clas=train.Class[train.Image==img]
        clas1=clas.iloc[0]
        tr_df.append([new_array,clas1])
    except Exception as e:
        pass
    
tr_df
tr_df
#df=pd.DataFrame(tr_df)
#df.head(5)
#df.columns=['array','Class']
#df.Class.unique()
mapping={'Food':0,'misc':1,'Attire':2,'Decorationandsignage':3}
#df['Class']=df['Class'].map(mapping).astype(int)
#df.head(5)
X=[]
Y=[]
for ar,cl in tr_df:
    X.append(ar)
    Y.append(mapping[cl])
X=np.array(X).reshape(-1,imgage_size,imgage_size,3)
X=X/255.0
Y=to_categorical(Y,num_classes=4)
#Y
x_tr,x_ts,y_tr,y_ts=train_test_split(X,Y,test_size=0.1,stratify=Y,shuffle=True)
x_tr.shape,x_ts.shape
len(y_ts)
from keras.callbacks import Callback,ModelCheckpoint
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout,Conv2D,MaxPool2D,Activation,Flatten,GlobalAveragePooling2D,BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
import keras.backend as K

## Model design ::

model1=Sequential()
model1.add(Conv2D(64,(3,3),input_shape=X.shape[1:]))
model1.add(BatchNormalization())
model1.add(Activation('relu'))
model1.add(MaxPool2D(pool_size=(2,2)))
model1.add(Dropout(0.25))

model1.add(Conv2D(64,(3,3)))
model1.add(BatchNormalization())
model1.add(Activation('relu'))
model1.add(MaxPool2D(pool_size=(2,2)))
model1.add(Dropout(0.25))

#model1.add(GlobalAveragePooling2D())
model1.add(Flatten())
model1.add(Dense(64))
model1.add(BatchNormalization())
model1.add(Activation('relu'))
model1.add(Dropout(0.5))
model1.add(BatchNormalization())
model1.add(Dense(4,activation='softmax'))

#model.add(Activation('softmax'))
model1.summary()
#check1=ModelCheckpoint(filepath='model1.basic.hdf5',verbose=1,save_best_only=True)     
       


from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(X)
#score=model.evaluate(x_ts,y_ts)
#score[1]  ## Vgg166  72
## eff7 51.08 54.24
## sgd 0.5058
os.listdir('../input/')
from keras.layers import Dense
from keras.models import Sequential
#import efficientnet.keras  as enf
from keras.callbacks import ReduceLROnPlateau
import keras
from keras import backend as K
from keras.models import Model, Input
from keras.layers import Dense, Lambda
from math import ceil
#import efficientnet.keras as efn
import efficientnet.keras as efn
# Generalized mean pool - GeM
gm_exp = tf.Variable(3.0, dtype = tf.float32)
def generalized_mean_pool_2d(X):
    pool = (tf.reduce_mean(tf.abs(X**(gm_exp)), 
                        axis = [1, 2], 
                        keepdims = False) + 1.e-7)**(1./gm_exp)
    return pool
def create_model(input_shape):
    # Input Layer
    input = Input(shape = input_shape)
    
    # Create and Compile Model and show Summary
    x_model = efn.EfficientNetB3(weights = 'imagenet', include_top = False, input_tensor = input, pooling = None, classes = None)
    
    # UnFreeze all layers
    for layer in x_model.layers:
        layer.trainable = True
    
    # GeM
    lambda_layer = Lambda(generalized_mean_pool_2d)
    lambda_layer.trainable_weights.extend([gm_exp])
    x = lambda_layer(x_model.output)
    
    # multi output
    classes= Dense(4, activation = 'softmax', name = 'class')(x)
    

    # model
    model = Model(inputs = x_model.input, outputs = [classes])

    return model
# Create Model
HEIGHT_NEW=WIDTH_NEW=150
CHANNELS=3

model3 = create_model(input_shape = (HEIGHT_NEW, WIDTH_NEW, CHANNELS))

reduce_learning_rate = ReduceLROnPlateau(monitor='loss',factor=0.2,patience=2,cooldown=2,min_lr=0.00001,verbose=1)
callbacks = [reduce_learning_rate]
model3.compile( optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model3.fit_generator(datagen.flow(X,Y, batch_size=32),
                    epochs=50,callbacks=callbacks)
model3.save('eff3.h5')
import gc
gc.collect()
#del model3,tr_df
del tr_df
#model.evaluate(x=x_ts, y=y_ts, batch_size=32, verbose=1)
#import efficientnet.tfkeras
#import tensorflow as tf
from keras.models import load_model
#from tensorflow.keras.models import load_model
#load=load_model('../working/eff3.h5')
#os.listdir()

## Test dataframe preparation :
test_df=[]
imgage_size=150
for img in test.Image:
    path=os.path.join(test_dir,img)
    try :
        
        img_array=cv2.imread(path,cv2.IMREAD_COLOR)
        new_array=cv2.resize(img_array,(imgage_size,imgage_size))
        #clas=train.Class[train.Image==img]
        #clas1=clas.iloc[0]
        test_df.append([new_array])
    except Exception as e1:
        pass
    
test_df=np.array(test_df).reshape(-1,imgage_size,imgage_size,3)
test_df=test_df/255
    
test_df.shape,X.shape
#sub.head()
#op=model3.predict(test_df)
#op
inverse={'0':'Food', '1':'misc', '2':'Attire', '3':'Decorationandsignage'}
labels = model3.predict(test_df)
print(labels[:4])
label = [np.argmax(i) for i in labels]
class_label = [inverse[str(x)] for x in label]
print(class_label[:3])
submission = pd.DataFrame({ 'Image': test.Image, 'Class': class_label })
submission.head(10)
submission.to_csv('submission.csv', index=False)
submission