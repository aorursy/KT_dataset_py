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

train=os.listdir('../input/image-detect/train')
X_train=[]
y_train=[]
for folderName in train:
    for image_filename in tqdm(os.listdir('../input/image-detect/train/' + folderName +'/'+'images')):
        
                img_file = cv2.imread('../input/image-detect/train/'+ folderName + '/' +'images' + '/'+image_filename)
                #img_file = skimage.transform.resize(img_file, (64, 64, 3))
                img_file = cv2.cvtColor(img_file, cv2.COLOR_BGR2RGB)
                img_arr = np.asarray(img_file)
                #img_arr=img_arr/255
                
                X_train.append(img_arr)
                y_train.append(folderName)
print(np.array(X_train).shape)
print(np.array(y_train).shape)

val_dataframe=pd.read_csv('/kaggle/input/image-detect/val/val_annotations.txt',sep='\t',names=['id','label','d1','d2','d3','d4'])
val_dataframe=val_dataframe.iloc[0:,0:2]
val_dataframe.sort_values(["id"], axis=0, ascending=True, inplace=True)

X_val=[]
y_val=[]
for image_filenames in tqdm(os.listdir("../input/image-detect/val/images")):
    img_file = cv2.imread('../input/image-detect/val/images/' + image_filenames) 
    img_file = cv2.cvtColor(img_file, cv2.COLOR_BGR2RGB)
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
train_generator = datagen_train.flow_from_directory('../input/image-detect/train', target_size = (64,64), batch_size= 32, class_mode="categorical")


datagen_val=ImageDataGenerator(rescale=1./255)
val_generator=datagen_val.flow(X_val,y_val_hot,batch_size=32)
img_height,img_width = 64,64 
num_classes = 200

base_model =keras.applications.ResNet101V2(include_top=False, weights='imagenet',input_shape=(64,64,3), classes=200)
from keras import regularizers

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.8)(x)
predictions = Dense(num_classes, activation= 'softmax', kernel_regularizer=regularizers.l2(0.001))(x)
model = Model(inputs = base_model.input, outputs = predictions)
model.summary()
opt =keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
model.save_weights("best_model_resnet50v2_2.h5")
checkpoint = ModelCheckpoint("best_model_resnet50v2_2.h5",monitor='val_accuracy',verbose=1,
                              save_best_only=True,mode='max')
callback=[checkpoint]
history = model.fit_generator(train_generator,
                              steps_per_epoch = 90000//100,
                              epochs = 70,
                              validation_data = val_generator,
                              validation_steps = 10000//100,callbacks=callback)
                             
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="validation loss")
ax1.set_xticks(np.arange(1, 70, 1))
#ax1.set_yticks(np.arange(0, 10, 0.1))

ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
ax2.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
ax2.set_xticks(np.arange(1, 70, 1))

legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

data_test=[]
file=glob('/kaggle/input/image-detect/test/images/*.JPEG')
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
test_filenames = os.listdir("../input//image-detect/test/images")
test_df = pd.DataFrame({'file_name': test_filenames,"category":pred})
test_df.sort_values(["file_name"], axis=0, ascending=True, inplace=True)   



test_df.to_csv('submission.csv',header=True, index=False)
