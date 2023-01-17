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
#Kullancağımız kütüphaleri ekliyoruz
from keras import models
from keras import layers
model=models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation="relu",input_shape=(28,28,1)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation="relu"))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation="relu"))
model.summary()
model.add(layers.Flatten())
model.add(layers.Dense(64,activation="relu"))
model.add(layers.Dense(10,activation="softmax")) #0-9 arasında ki rakamlar toplam 10 tane olduğu için çıkış katmanımızı 10 olarak belirledik.
#Ağımızın son halini gözden geçirelim:
model.summary()
from keras.datasets import mnist
from keras.utils import to_categorical

(train_images,train_labels),(test_images,test_labels)=mnist.load_data() #datamızı indirecek kod 

train_images=train_images.reshape((60000,28,28,1)) #Eğitim için 60000 resim,28x28 boyutunda,1 yani siyah beyaz kanal derinliğini ifade eder
train_images=train_images.astype("float32")/255 #255 rgb renk max değerine bölerek normalizasyon işlemi yapıyoruz astype ile veri değiştirme işlemi yapılır

test_images=test_images.reshape(10000,28,28,1) #Test için10000 resim,28x28 boyutunda,1 yani siyah beyaz kanal derinliğini ifade eder
test_images=test_images.astype("float32")/255

train_labels=to_categorical(train_labels) #0-9 arası rakamları kategorize ediyoruz
test_labels=to_categorical(test_labels)

model.compile(optimizer="rmsprop",
             loss="categorical_crossentropy",
             metrics=["accuracy"]) #Tek etiketli çoklu sınıflandırma için "categorical_crossentropy" kullanılır.Sınıf rakam etikeler 0-9 arasu-ı rakamlar.Bu doğrultuda en iyi optimizer rmspropdur
model.fit(train_images,train_labels,epochs=5,batch_size=64)
test_loss,test_acc=model.evaluate(test_images,test_labels)
test_acc
model_no_max_pool=models.Sequential()
model_no_max_pool.add(layers.Conv2D(32,(3,3),activation="relu",
                                   input_shape=(28,28,1)))
model_no_max_pool.add(layers.Conv2D(64,(3,3),activation="relu"))
model_no_max_pool.add(layers.Conv2D(64,(3,3),activation="relu"))
model_no_max_pool.summary()