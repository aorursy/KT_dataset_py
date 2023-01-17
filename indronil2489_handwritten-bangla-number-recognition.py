# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import cv2
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
imgloc=[]
label=[]
for i in range(0,10):
    for dirname, _, filenames in os.walk('/kaggle/input/ekush-bangla-handwritten-data-numerals/'+str(i)):
        for filename in filenames:
            imgloc.append((os.path.join(dirname, filename)).replace("'", ""))
            label.append(i)

img=[]
for i in range(0, len(imgloc)):
    img1 = cv2.imread(imgloc[i],1)
    img2 = np.array(img1)
    retValue, img2 = cv2.threshold(img2, 55, 255, cv2.THRESH_BINARY)
    img2 = cv2.resize(img2,(56,56))
    img.append(cv2.cvtColor(img2,cv2.COLOR_BGR2RGB))
plt.figure()
f, axarr = plt.subplots(5,10) 

for i in range(0,5):
    for j in range(0,10):
        axarr[i][j].imshow(img[(10*i)+j])
        
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(img, label, test_size=0.25, random_state=11)
print (y_train)
len(x_train)
img2=img[8788]
img2 = np.array(img2)
retValue, img2 = cv2.threshold(img2, 55, 255, cv2.THRESH_BINARY)
#num_rows, num_cols = img2.shape[:2]
#rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 30, 1)
#img2 = cv2.warpAffine(img2, rotation_matrix, (num_cols, num_rows))
plt.imshow(cv2.cvtColor(img2,cv2.COLOR_BGR2RGB))
plt.show()
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
model = Sequential()
model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1),activation='tanh',input_shape=img[0].shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(32, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(10, activation='softmax'))
import keras
model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01),
              metrics=['accuracy'])
model.fit(np.array(x_train), y_train, batch_size=25, epochs=20, verbose=1)
score = model.evaluate(np.array(x_test), np.array(y_test), verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
pre=[]
pred = model.predict(np.array(x_test))
for i in range(0,len(y_test)):
    p=pred[i][0]
    tmp=0
    for j in range(0,10):
        if pred[i][j]>p:
            p=pred[i][j]
            tmp=j
    pre.append(tmp)
for i in range(0,len(y_test)):
    print(y_test[i], pre[i])
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,pre)
cm
testloc=[]
for dirname, _, filenames in os.walk('/kaggle/input/test--'):
        for filename in filenames:
            testloc.append((os.path.join(dirname, filename)).replace("'", ""))
            
test=[]
for i in range(0, len(testloc)):
    img1 = cv2.imread(testloc[i],1)
    img2 = np.array(img1)
    img2=cv2.resize(img2,(56,56))
    retValue, img2 = cv2.threshold(img2, 55, 255, cv2.THRESH_BINARY)
    img2 = cv2.bitwise_not(img2)
    strEl = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    img2 = cv2.dilate(img2, strEl)
    test.append(cv2.cvtColor(img2,cv2.COLOR_BGR2RGB))
    
plt.imshow(test[0])
plt.show()
pret=[]
pred2 = model.predict(np.array(test))
for i in range(0,len(test)):
    p=pred2[i][0]
    tmp=0
    for j in range(0,10):
        if pred2[i][j]>p:
            p=pred2[i][j]
            tmp=j
    pret.append(tmp)
plt.figure()
f, axarr = plt.subplots(5,6) 

for i in range(0,5):
    for j in range(0,6):
        axarr[i][j].imshow(test[(6*i)+j])
        
pret