# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from PIL import Image

import os

import glob

import matplotlib.pyplot as plt

import cv2

from keras.utils import to_categorical

import keras

train = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")


def convert_img(img):

    desired_size = 224

    im_pth = img



    im = Image.open(im_pth)

    old_size = im.size  # old_size[0] is in (width, height) format



    ratio = float(desired_size)/max(old_size)

    new_size = tuple([int(x*ratio) for x in old_size])

    # use thumbnail() or resize() method to resize the input image



    # thumbnail is a in-place operation



    # im.thumbnail(new_size, Image.ANTIALIAS)



    im = im.resize(new_size, Image.ANTIALIAS)

    # create a new image and paste the resized on it



    new_im = Image.new("RGB", (desired_size, desired_size))

    new_im.paste(im, ((desired_size-new_size[0])//2,

                        (desired_size-new_size[1])//2))



    new_im.show()
def convert_img(img):    

    desired_size = 224

    im_pth = img

    im = Image.open(im_pth)

    old_size = im.size  # old_size[0] is in (width, height) format



    ratio = float(desired_size)/max(old_size)

    new_size = tuple([int(x*ratio) for x in old_size])

    # use thumbnail() or resize() method to resize the input image



    # thumbnail is a in-place operation



    # im.thumbnail(new_size, Image.ANTIALIAS)



    im = im.resize(new_size, Image.ANTIALIAS)

    # create a new image and paste the resized on it



    new_im = Image.new("RGB", (desired_size, desired_size))

    new_im.paste(im, ((desired_size-new_size[0])//2,

                        (desired_size-new_size[1])//2))

    return new_im
x_train = [np.array(convert_img("../input/aptos2019-blindness-detection/train_images/"+i+".png")) for i in train.id_code]  
x_train = np.array(x_train)
y_train = train.diagnosis
train.groupby(train.diagnosis).count()
y_train = to_categorical(y_train)
model = keras.applications.densenet.DenseNet121(include_top=True, weights=None)
model.load_weights("../input/densenet121/densenet121.h5")
x = model.layers[-2].output

d = keras.layers.Dense(512,activation='relu')(x)

e = keras.layers.Dense(5,activation='softmax')(d)
model1 = keras.models.Model(model.input,e)
model1.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model1.fit(x_train,y_train,validation_split=0.10,epochs=20,class_weight={0:0.1,1:0.5,2:0.4,3:0.8,4:0.6})
model1.fit(x_train,y_train,validation_split=0.10,epochs=10)

testdf = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")

testx = []

for i in testdf.id_code:

    temp = np.array(cv2.resize(np.array(Image.open('../input/aptos2019-blindness-detection/test_images/'+i+".png")),(224,224)))

    testx.append(temp)

testx = np.array(testx)
result = model1.predict(testx)
res = []

for i in result:

    res.append(np.argmax(i))
df_test = pd.DataFrame({"id_code": testdf["id_code"].values, "diagnosis": res})

df_test.head()
len(testdf)
df_test.to_csv('submission.csv', index=False)
