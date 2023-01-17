# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        os.path.join(dirname, filename)

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from tensorflow import keras
from keras.layers import Input, Dense
from keras import applications
from keras.layers import Activation,Add, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, MaxPooling2D, Dropout,GlobalAveragePooling2D
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,load_model,Model
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.initializers import *
from keras import regularizers
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

train=os.listdir('../input/image-classification/train')
X_train=[]
y_train=[]
for folderName in train:
    for image_filename in tqdm(os.listdir('../input/image-classification/train/' + folderName +'/'+'images')):
        
                img_file = cv2.imread('../input/image-classification/train/'+ folderName + '/' +'images' + '/'+image_filename)
                #img_file = skimage.transform.resize(img_file, (64, 64, 3))
                img_arr = np.asarray(img_file)
                #img_arr=img_arr/255
                
                X_train.append(img_arr)
                y_train.append(folderName)
print(np.array(X_train).shape)
print(np.array(y_train).shape)

from tqdm import tqdm

val_dataframe=pd.read_csv('/kaggle/input/image-classification/val/val_annotations.txt',sep='\t',names=['id','label','d1','d2','d3','d4'])
val_dataframe=val_dataframe.iloc[0:,0:2]
val_dataframe.sort_values(["id"], axis=0, ascending=True, inplace=True)

X_val=[]
y_val=[]
for image_filenames in tqdm(os.listdir("../input/image-classification/val/images")):
    img_file = cv2.imread('../input/image-classification/val/images/' + image_filenames) 
    img_arr = np.asarray(img_file)
    #img_arr=img_arr/255
    X_val.append(img_arr)
    
    for i in range (len(val_dataframe['id'].values)):
        if(image_filenames==val_dataframe['id'].values[i]):
             y_val.append(val_dataframe['label'].values[i])
        
    
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical


y_train= LabelEncoder().fit_transform(y_train)
y_train_hot=keras.utils.to_categorical(y_train, num_classes=200)


y_val=LabelEncoder().fit_transform(y_val)
y_val_hot=keras.utils.to_categorical(y_val, num_classes=200)
y_val_hot.shape
X_val=np.array(X_val)
X_train=np.array(X_train)
X_train.shape

datagen_train = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
train_generator = datagen_train.flow_from_directory('../input/image-classification/train', target_size = (64,64), batch_size= 32, class_mode="categorical")


datagen_val=ImageDataGenerator(rescale=1./255)
val_generator=datagen_val.flow(X_val,y_val_hot,batch_size=32)
img_height,img_width = 64,64 
num_classes = 200

#base_model = applications.resnet50.ResNet50(weights= None, include_top=False, input_shape= (img_height,img_width,3))
#from keras import regularizers

#x = base_model.output
#x = GlobalAveragePooling2D()(x)
#x = Dropout(0.6)(x)
#predictions = Dense(num_classes, activation= 'softmax', kernel_regularizer=regularizers.l2(0.001))(x)
#model = Model(inputs = base_model.input, outputs = predictions)
#model.summary()
def identity_block(X,f,filters,stage,block):
    
    conv_name_base = 'res_'+str(stage)+block+'_branch'
    bn_name_base = 'bn_'+str(stage)+block+'_branch'
    
    F1,F2,F3 = filters
    
    X_shortcut = X
    
    # First Component of Main Path
    X = Conv2D(filters=F1,kernel_size=(3,3),strides=(1,1),
               padding='same',name=conv_name_base+'2a',
               kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(name=bn_name_base+'2a')(X)
    X = Activation('relu')(X)
    
    # Second Component of Main Path
    X = Conv2D(filters=F2,kernel_size=(f,f),strides=(1,1),
              padding='same',name=conv_name_base+'2b',
              kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(name=bn_name_base+'2b')(X)
    X = Activation('relu')(X)
    
    # Third Component of Main Path
    X = Conv2D(filters=F3,kernel_size=(3,3),strides=(1,1),
              padding='same',name=conv_name_base+'2c',
              kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(name=bn_name_base+'2c')(X)
    
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)
    
    return X
def convolutional_block(X,f,filters,stage,block,s=2):
    
    conv_base_name = 'res_' + str(stage) + block + '_branch'
    bn_base_name = 'bn_' + str(stage) + block + '_branch'
    
    F1,F2,F3 = filters
    
    X_shortcut = X
    
    ### MAIN PATH ###
    # First component of main path
    X = Conv2D(filters=F1,kernel_size=(3,3),strides=(s,s),
              padding='same',name=conv_base_name+'2a',
              kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(name=bn_base_name+'2a')(X)
    X = Activation('relu')(X)
    
    # Second Component of main path
    X = Conv2D(filters=F2,kernel_size=(f,f),strides=(1,1),
              padding='same',name=conv_base_name+'2b',
              kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(name=bn_base_name+'2b')(X)
    X = Activation('relu')(X)
    
    # Third Component of main path
    X = Conv2D(filters=F3,kernel_size=(3,3),strides=(1,1),
              padding='same',name=conv_base_name+'2c',
              kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(name=bn_base_name+'2c')(X)
    
    # Shortcut path
    X_shortcut = Conv2D(filters=F3,kernel_size=(1,1),strides=(s,s),
                       padding='same',name=conv_base_name+'1',
                       kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(name=bn_base_name+'1')(X_shortcut)
    
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)
    
    return X
def ResNet(input_shape,classes):
    
    X_input = Input(input_shape)
    
    # Zero Padding
    X = ZeroPadding2D((3,3))(X_input)
    
    # Stage 1
    X = Conv2D(64,(7,7),strides=(2,2),name='conv1',kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3,3),strides=(2,2))(X)
    
    # Stage 2
    X = convolutional_block(X,f=3,filters=[64,64,128],stage=2,block='A',s=1)
    X = identity_block(X,3,[64,64,128],stage=2,block='B')
    X = identity_block(X,3,[64,64,128],stage=2,block='C')
    
    # Stage 3
    X = convolutional_block(X,f=3,filters=[128,128,256],stage=3,block='A',s=2)
    X = identity_block(X,f=3,filters=[128,128,256],stage=3,block='B')
    X = identity_block(X,f=3,filters=[128,128,256],stage=3,block='C')
    X = identity_block(X,f=3,filters=[128,128,256],stage=3,block='D')
    
    # Stage 4
    X = convolutional_block(X,f=3,filters=[256,256,512],stage=4,block='A',s=2)
    X = identity_block(X,f=3,filters=[256,256,512],stage=4,block='B')
    X = identity_block(X,f=3,filters=[256,256,512],stage=4,block='C')
    X = identity_block(X,f=3,filters=[256,256,512],stage=4,block='D')
    X = identity_block(X,f=3,filters=[256,256,512],stage=4,block='E')
    X = identity_block(X,f=3,filters=[256,256,512],stage=4,block='F')
    
    # Stage 5
    X = convolutional_block(X,f=3,filters=[512,512,1024],stage=5,block='A',s=1)
    X = identity_block(X,f=3,filters=[512,512,1024],stage=5,block='B')
    X = identity_block(X,f=3,filters=[512,512,1024],stage=5,block='C')
    
#     # Stage 6
#     X = convolutional_block(X,f=3,filters=[1024,1024,2048],stage=6,block='A',s=2)
#     X = identity_block(X,f=3,filters=[1024,1024,2048],stage=6,block='B')
#     X = identity_block(X,f=3,filters=[1024,1024,2048],stage=6,block='C')
#     X = identity_block(X,f=3,filters=[1024,1024,2048],stage=6,block='D')
    
    # Average Pool Layer
    X =  GlobalAveragePooling2D()(X)
    X = Dropout(0.6)(X)    

    
    # Output layer
    X = Flatten()(X)
    X = Dense(200,activation='softmax',name='fc'+str(classes), kernel_regularizer=regularizers.l2(0.0001),
              kernel_initializer=glorot_uniform(seed=0))(X)
    
    model = Model(inputs=X_input,outputs=X,name='ResNet')
    
    return model
model = ResNet(input_shape=(64,64,3),classes=200)
model.summary()
opt =keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
model.save_weights("best_model.h5")
checkpoint = ModelCheckpoint("best_model.h5",monitor='val_accuracy',verbose=1,
                              save_best_only=True,mode='max')
callback=[checkpoint]

history = model.fit_generator(train_generator,
                              steps_per_epoch = 90000//100,
                              epochs = 50,
                              validation_data = val_generator,
                              validation_steps = 10000//100,callbacks=[checkpoint])
                             
                             
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="validation loss")
ax1.set_xticks(np.arange(1, 50, 1))
#ax1.set_yticks(np.arange(0, 10, 0.1))

ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
ax2.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
ax2.set_xticks(np.arange(1, 50, 1))

legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

data_test=[]
file=glob('/kaggle/input/image-classification/test/images/*.JPEG')
for f in file:
    images=load_img(f)
    image_array=img_to_array(images)
    image_array=image_array/255
   
    data_test.append(image_array)
    images=None
    image_array=None
x_test=np.array(data_test)
x_test.shape
prediction = model.predict(x_test)
prediction
predicted_class_indices=np.argmax(prediction,axis=1)
predicted_class_indices
labels = (train_generator.class_indices)
#print(labels.items())
labels = dict((v,k) for k,v in labels.items())
pred = [labels[k] for k in predicted_class_indices]
test_filenames = os.listdir("../input//image-classification/test/images")
test_df = pd.DataFrame({'file_name': test_filenames,"category":pred})
test_df.sort_values(["file_name"], axis=0, ascending=True, inplace=True)   

#test_df.drop(['filename', 'category'], axis=1, inplace=True)

test_df.to_csv('submission.csv',header=True, index=False)