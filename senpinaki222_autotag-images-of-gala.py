import numpy as np
import pandas as pd
import cv2

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tqdm import tqdm
df = pd.read_csv('../input/galasimages/dataset/train.csv')
x = df['Image']
y = df['Class']
x_train = []


for t in range(len(x)):
    k = '../input/galasimages/dataset/Train Images/' + x[t]
    img = cv2.imread(k, cv2.IMREAD_COLOR)

    rsz_img = cv2.resize(img, (28, 28))

    #cv2.imshow(x[t], rsz_img)

    arr = []
    for i in range(28):
        ar = []
        for j in range(28):
            ar.append(rsz_img[i][j])
        arr.append(ar)

    #ar = np.array(ar)

    x_train.append(arr)

    #print(len(x_train))

x_train = np.array(x_train)
print("shape of training image => ", end = '')
print(x_train.shape)
#y_train = pd.read_csv('../input/galasimages/dataset/train.csv')
y_train = pd.get_dummies(y)
y_train.shape
y_train
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
model.fit(x_train, y_train, epochs=30, validation_split = 0.1)
df_test = pd.read_csv('../input/galasimages/dataset/test.csv')
tx = df_test['Image']
x_test = []


for t in range(len(tx)):
    k = '../input/galasimages/dataset/Test Images/' + tx[t]
    img = cv2.imread(k, cv2.IMREAD_COLOR)

    rsz_img = cv2.resize(img, (28, 28))

    #cv2.imshow(x[t], rsz_img)

    arr = []
    for i in range(28):
        ar = []
        for j in range(28):
            ar.append(rsz_img[i][j])
        arr.append(ar)

    #ar = np.array(ar)

    x_test.append(arr)

    #print(len(x_train))

x_test = np.array(x_test)
print("shape of training image => ", end = '')
print(x_test.shape)
y_pred = model.predict(x_test)
classes = ['Attire', 'Decorationandsignage', 'Food', 'misc']

ans = []
for i in range(3219):
    k = max(y_pred[i])
    if(k==y_pred[i][0]):
        a = 0
    elif(k==y_pred[i][1]):
        a = 1
    elif(k==y_pred[i][2]):
        a = 2
    elif(k==y_pred[i][3]):
        a = 3
    
    ans.append(classes[a])
ans = np.array(ans)
y_pred
ans
df_prediction = pd.DataFrame({'Image': tx, 'Class': ans})
df_prediction.to_csv('submission_4_autotag.csv', index=False)
