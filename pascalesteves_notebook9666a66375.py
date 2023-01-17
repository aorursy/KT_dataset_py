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
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from tensorflow.python import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dropout,Dense, Activation
from keras.optimizers import Adam
from keras.applications import VGG16
df = pd.read_csv('../input/aerial-cactus-identification/train.csv')
file_name = '../input/aerial-cactus-identification/train.zip'
with ZipFile(file_name, 'r') as zip: 
    # printing all the contents of the zip file 
    zip.printdir() 
  
    # extracting all the files 
    print('Extracting all the files now...') 
    zip.extractall() 
file_name = '../input/aerial-cactus-identification/test.zip'
with ZipFile(file_name, 'r') as zip: 
    # printing all the contents of the zip file 
    zip.printdir() 
  
    # extracting all the files 
    print('Extracting all the files now...') 
    zip.extractall() 
train_dir = './train/'
test_dir = './test/'
vgg16_net = VGG16(weights = 'imagenet', include_top = False, input_shape = (32,32,3))
vgg16_net.trainable = False
vgg16_net.summary()
model = Sequential()
model.add(vgg16_net)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
df.head()
x_train = []
y_train = []
images = df['id'].values
x_test = []
img = os.listdir(train_dir)
img_test = os.listdir(test_dir)
for i in range(len(img)):
    x_train.append(cv2.imread(train_dir + img[i]))
for i in range(len(img_test)):
    x_test.append(cv2.imread(test_dir + img_test[i]))

for i in img:
    y_train.append(df[df['id']==i]['has_cactus'].values[0])

x_train = np.asarray(x_train)
x_test = np.asarray(x_test)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = np.asarray(y_train)
len(y_train)
len(x_train)
history = model.fit(x_train,y_train,batch_size=32,epochs=10,validation_split=0.1,shuffle=True,verbose=2)
loss_train = history.history['loss']
loss_val = history.history['val_loss']
accuracy_train = history.history['accuracy']
accuracy_val = history.history['val_accuracy']
x = range(1,11)
plt.plot(x,loss_train,label='Training Loss')
plt.plot(x,loss_val, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.plot(x,accuracy_train,label='Train Accuracy')
plt.plot(x,accuracy_val, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
prediction = model.predict(x_test)
submission = pd.read_csv('../input/aerial-cactus-identification/sample_submission.csv')
submission['has_cactus']= prediction
submission['has_cactus']= submission['has_cactus'].apply(lambda x: 1 if x>0.75 else 0)
submission.to_csv('Submission_file.csv')

