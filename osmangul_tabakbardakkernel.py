import numpy as np 

import keras

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns

from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import RMSprop

from keras.layers import Conv2D, MaxPooling2D

from keras.layers.convolutional import *

from keras.layers import Activation

from keras.layers.core import Dense,Flatten

from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf

import random as rn

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import cv2                  

from tqdm import tqdm

print(os.listdir("../input"))
X=[]

Y=[]

img_size=150

bardaklar= '../input/bilgi/bilgi/bardak'

tabaklar= '../input/bilgi/bilgi/tabak'
def assign_label(img,img_type): 

    return img_type
#Fotoğrafların yüklenmesi

def make_train_data(img_type,DIR):

    for img in tqdm(os.listdir(DIR)):

        #tqdm veri yüklemesi sırasında tren tipi görsel yükleme ekranı çıkması için kullanılan bir yapı...

        label=assign_label(img,img_type) #verilerin etiketlerini alması

        path = os.path.join(DIR,img)

        img = cv2.imread(path,cv2.IMREAD_COLOR)

        if img is not None:

            img = cv2.resize(img, (img_size,img_size))

            X.append(np.array(img)) #X dizininde resimlerimizi

            Y.append(str(label))# Y dizininde ise resimlerimizin etiketlerini tutuyoruz

        else:

            print("resim yüklenemedi")
make_train_data('bardaklar',bardaklar)

print(len(X))
make_train_data('tabaklar',tabaklar)

print(len(X))
fig,ax=plt.subplots(5,2)

fig.set_size_inches(17,17)

for i in range(5):

    for j in range (2):

        l=rn.randint(0,len(Y))

        ax[i,j].imshow(X[l])

        ax[i,j].set_title('Veri: '+Y[l])        

plt.tight_layout()
X=np.array(X)

X=X.astype('float32') / 255

Y = np.asarray(Y)
pd.unique(Y)
from sklearn.model_selection import train_test_split

x_train,x_valid,y_train,y_valid=train_test_split(X,Y,test_size=0.20,random_state=42)
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

y_valid=le.fit_transform(y_valid)

y_train=le.fit_transform(y_train)
from keras import layers #modeli tanımladık

from keras import models



model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu',

                        input_shape=(150, 150, 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))



model.summary()
from keras import optimizers



model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-4),

              metrics=['acc'])

y_train_binary=to_categorical(y_train,2)

y_valid_binary=to_categorical(y_valid,2)
history =model.fit(x_train,y_train,epochs=10,batch_size=10,validation_data = (x_valid,y_valid))

acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



acc
epochs = range(len(acc))





plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()



plt.figure()



plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()
datagen = ImageDataGenerator(

      rotation_range=40,

      width_shift_range=0.2,

      height_shift_range=0.2,

      shear_range=0.2,

      zoom_range=0.2,

      horizontal_flip=True,

      fill_mode='nearest')
batch_size=80

epochs=10

# Ezberlemeyi önlemek için Data Augmentation 

datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=True,  # randomly flip images

        vertical_flip=False)  # randomly flip images

datagen.fit(x_train)
model.compile(optimizer=optimizers.RMSprop(lr=1e-4),loss='binary_crossentropy',metrics=['acc'])

model.summary()

History=model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size,validation_data = (x_valid,y_valid))
model.save('bilgi_model.h5')
acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()



plt.figure()



plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()
# validation setini tahmin etme

pred=model.predict(x_valid)

pred_digits=np.argmax(pred,axis=1)
#Doğru ve yanlış tahmin sonuçları

true=0

false=0

for i in range(len(pred)):

   if(pred_digits[i]==y_valid[i]):

    true=true+1

   else:

    false=false+1

print("Doğru Tahmin Sayısı:",+true)

print("Yanlış Tahmin Sayısı:",+false)
i=0

prop_class=[]

mis_class=[]



for i in range(len(y_valid)):

    if(y_valid[i]==pred_digits[i]):

        prop_class.append(i)

    if(len(prop_class)==8):

        break



i=0

for i in range(len(y_valid)):

    if(not y_valid[i]==pred_digits[i]):

        mis_class.append(i)

    if(len(mis_class)==8):

        break
#Doğru tahminlerin görselleştirilmesi

count=0

fig,ax=plt.subplots(4,2)

fig.set_size_inches(17,17)

for i in range (4):

    for j in range (2):

        ax[i,j].imshow(x_valid[prop_class[count]])

        ax[i,j].set_title("Doğru tahmin edilen veriler : "+str(le.inverse_transform([pred_digits[prop_class[count]]]))+"\n"+"Doğru veiler : "+str(le.inverse_transform([y_valid[prop_class[count]]])))

        plt.tight_layout()

        count+=1
#Yanlış tahminlerin görselleştirilmesi

count=0

fig,ax=plt.subplots(4,2)

fig.set_size_inches(17,17)

for i in range (4):

    for j in range (2):

        ax[i,j].imshow(x_valid[mis_class[count]])

        ax[i,j].set_title("Tahmin Verisi : "+str(le.inverse_transform([pred_digits[mis_class[count]]]))+"\n"+"Gerçek Veri : "+str(le.inverse_transform([y_valid[mis_class[count]]])))

        plt.tight_layout()

        count+=1
#Yeni eğitim verisi ile yaptığımı işlemleri test etmek istersek...

from keras.preprocessing import image

test_image = image.load_img('../input/bilgi/bilgi/tabak/tabak (12).jpg', target_size = (img_size, img_size))

test_image = image.img_to_array(test_image)

test_image = np.expand_dims(test_image, axis = 0)

result = model.predict(test_image)

print(result)

if result[0][0] == 1:

    prediction = 'Tabak'

else:

    prediction = 'Bardak'

print('tahmine göre yeni görsel bir %s' %prediction)