# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import cv2
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import plotly.graph_objs as go
import plotly.tools as tls
import seaborn as sns
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib
%matplotlib inline

import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/digit-recognizer/train.csv')
y_train = df['label']
x_train = df.drop("label",axis=1)
x_train=np.array(x_train)
x_train= x_train.reshape(len(x_train),28,28,1)
for k in range (0,len(x_train)):
    for i in range(0,28):
        for j in range(0,28):
            if x_train[k][i][j]>25:
                x_train[k][i][j]=255
            else:
                x_train[k][i][j]=0
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),activation='tanh',input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(10, activation='softmax'))
import keras
model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01),
              metrics=['accuracy'])
model.fit(np.array(x_train), y_train, batch_size=30, epochs=30, verbose=1)
df2 = pd.read_csv('../input/digit-recognizer/test.csv')
x_test = np.array(df2)
x_test= x_test.reshape(len(x_test),28,28,1)
for k in range (0,len(x_test)):
    for i in range(0,28):
        for j in range(0,28):
            if x_test[k][i][j]>25:
                x_test[k][i][j]=255
            else:
                x_test[k][i][j]=0
pre=[]
pred = model.predict(np.array(x_test))
for i in range(0,len(x_test)):
    p=pred[i][0]
    tmp=0
    for j in range(0,10):
        if pred[i][j]>p:
            p=pred[i][j]
            tmp=j
    pre.append(tmp)
plt.figure()
f, axarr = plt.subplots(4,6) 

for i in range(0,4):
    for j in range(0,6):
        axarr[i][j].imshow((x_test[(6*i)+j]).reshape(28,28))

lst=[]
for i in range(0, len(x_test)):
    lst.append((i+1, pre[i]))
import pandas
df = pandas.DataFrame(data=lst)
df.to_csv("./file.csv", sep=',',index=True)