import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import os, sys

from PIL import Image

import glob

import cv2

from keras.utils import to_categorical

import keras
os.listdir('../input/')
train = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
train.head()
len(train)
train_list = [[Image.open('../input/aptos2019-blindness-detection/train_images/'+i+'.png'),j] for i,j in zip(train.id_code[:5],train.diagnosis[:5])]

train_list
for i,j in train_list:

    plt.figure(figsize=(5,3))

    i = cv2.resize(np.asarray(i),(256,256))

    plt.title(j)

    plt.imshow(i)

    plt.show
x_train = [cv2.resize(np.asarray(Image.open('../input/aptos2019-blindness-detection/train_images/'+i+'.png')),(256,256)) for i in train.id_code]
x_train = np.array(x_train)
y_train = train.diagnosis
y_train = to_categorical(y_train)

y_train
model = keras.applications.densenet.DenseNet121(input_shape=(256,256,3),include_top=True,weights=None)
model.summary()
model.load_weights('../input/densenet-keras/DenseNet-BC-121-32.h5')
x = model.layers[-2].output

d = keras.layers.Dense(512,activation='relu')(x)

e = keras.layers.Dense(5,activation='softmax')(d)
model1 = keras.models.Model(model.input,e)
model1.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model1.fit(x_train,y_train,validation_split=0.20,epochs=10)
test_df = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')

test = []

for i in test_df.id_code:

    temp = np.array(cv2.resize(np.array(Image.open('../input/aptos2019-blindness-detection/test_images/'+i+'.png')),(256,256)))

    test.append(temp)

test = np.array(test)
np.random.seed(42)

result = model1.predict(test)
res = []

for i in result:

    res.append(np.argmax(i))
df_test = pd.DataFrame({"id_code": test_df["id_code"].values, "diagnosis": res})

df_test.head(20)
df_test.to_csv('submission.csv',index=False)