# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import cv2
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import tensorflow as tf

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/dance-type-dataset/Dance Dataset/train.csv')
image = df['Image']
label = np.array(df['target'])
data_size= len(label)
y_label=[]
for i in range(0,data_size):
    if label[i]=='bharatanatyam':
        y_label.append(0)
    elif label[i]=='kathak':
        y_label.append(1)                        
    elif label[i]=='kathakali':
        y_label.append(2)
    elif label[i]=='kuchipudi':
        y_label.append(3)
    elif label[i]=='manipuri':
        y_label.append(4)
    elif label[i]=='mohiniyattam':
        y_label.append(5)
    elif label[i]=='odissi':
        y_label.append(6)
    elif label[i]=='sattriya':
        y_label.append(7)

y_label=np.array(y_label)
img=[]
for i in range(0,data_size):
    imgloc="../input/dance-type-dataset/Dance Dataset/train/"+str(image[i])
    img1 = cv2.imread(imgloc,1)
    img1 = cv2.resize(img1,(56,56))
    img.append(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))
            
plt.imshow(img[1])
plt.show()
from tensorflow.keras.applications.vgg16 import VGG16
vgg = VGG16(include_top=False, input_shape = (56,56,3), weights = 'imagenet')
model = tf.keras.layers.Flatten()(vgg.output)
model = tf.keras.layers.Dense(1024, activation = 'relu')(model)
model = tf.keras.layers.Dropout(0.5)(model)
model = tf.keras.layers.Dense(8, activation = 'softmax')(model)
fin_model = tf.keras.Model(vgg.input, model)
fin_model.summary()
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),activation='tanh',input_shape=(56, 56, 3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(10, activation='softmax'))
import keras
fin_model.compile(tf.keras.optimizers.SGD(lr = 0.001), loss=keras.losses.sparse_categorical_crossentropy, metrics=['acc'])
fin_model.fit(np.array(img), np.array(y_label), batch_size=10, epochs=30, verbose=1)
df2 = pd.read_csv('../input/dance-type-dataset/Dance Dataset/test.csv')
data_size= len(df2)
image2 = df2['Image']
img2=[]
for i in range(0,data_size):
    imgloc="../input/dance-type-dataset/Dance Dataset/test/"+str(image2[i])
    img1 = cv2.imread(imgloc,1)
    img1 = cv2.resize(img1,(56,56))
    img2.append(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))
pred = fin_model.predict(np.array(img2))
predicted_clases = np.argmax(pred,axis=-1)
predicted_clases
plt.figure()
f, axarr = plt.subplots(3,3) 

for i in range(0,3):
    for j in range(0,3):
        axarr[i][j].imshow((img2[(3*i)+j]))
data = {'Image': image2, 'target': predicted_clases}
df3 = pd.DataFrame(data)
df3['target']= df3['target'].map({0: 'bharatanatyam',
                                1: 'kathak',
                                2: 'kathakali',
                                3: 'kuchipudi',
                                4: 'manipuri',
                                5: 'mohiniyattam',
                                6: 'odissi',
                                7: 'sattriya'})
df3
import pandas
df3.to_csv("./file.csv", sep=',',index=True)