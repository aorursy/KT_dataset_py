# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import cv2
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
img = cv2.imread('../input/captcha-version-2-images/samples/226md.png')
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
a=0
train_labels2=np.asarray([])
import os
for dirname, _, filenames in os.walk("/kaggle/input/captcha-version-2-images/samples/samples/"):
    for filename in filenames:
        if a<5000:
            train_labels2 = np.append(train_labels2, filename.replace(".png", ""))
            m=cv2.imread(os.path.join(dirname, filename))
            m =np.asarray(cv2.cvtColor(m, cv2.COLOR_BGR2GRAY))
            m=m.reshape(1,200,50)
            if a==0:
                train_images=m
            else:
                train_images=np.vstack([train_images,m])
            train_images=train_images.reshape(a+1,200,50)
        else:
            break
        a+=1
ext=np.zeros((1,5,36))
train_labels=np.asarray([])
b=0
for x in range(0,len(train_labels2)):
    if b==0:
        train_labels=ext
    else:
        train_labels=np.vstack([train_labels,ext])
    train_labels=train_labels.reshape(-1,5,36)
    
    for y in range(0,5):
        if train_labels2[x][y].isnumeric():
            if b==0:
                train_labels[x,y,ord(str(train_labels2[x][y]))-48]=1
            else:
                train_labels[x,y,ord(str(train_labels2[x][y]))-48]=1
        else:
            if b==0:
                train_labels[x,y,ord(str(train_labels2[x][y]))-87]=1
            else:
                train_labels[len(train_labels)-1,y,ord(str(train_labels2[x][y]))-87]=1
    b+=1
train_images=train_images.reshape(-1,50,200,1)
train_labels=train_labels.reshape(-1,180)

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
model = keras.Sequential()
model.add(keras.layers.Conv2D(input_shape=(50,200,1),padding="same",filters=32, kernel_size=(3, 3), activation=tf.nn.tanh))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=500, activation=tf.nn.relu))
model.add(keras.layers.Dense(units=300, activation=tf.nn.relu))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(units=180, activation=tf.nn.sigmoid))
opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt,loss= 'mse')
train_images=train_images.reshape(-1,50,200,1)
model.fit(train_images, train_labels, epochs=1,shuffle=True,batch_size=1)
predict=model.predict(train_images)
print(predict[0])
print(train_labels[0])