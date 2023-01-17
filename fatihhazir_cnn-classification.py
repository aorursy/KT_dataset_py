import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv("../input/digit-recognizer/train.csv")

print(train.shape)

train.head()
Y_train = train["label"]

X_train = train.drop(labels = ["label"],axis = 1) 
plt.figure(figsize=(15,7))

sns.countplot(Y_train, palette="icefire")

plt.title("Sayi kumeleri")
img = X_train.iloc[4].to_numpy()

img = img.reshape((28,28))

plt.imshow(img,cmap='gray')

plt.title(train.iloc[4,0])

plt.axis("off")

plt.show()
img = X_train.iloc[2].to_numpy()

img = img.reshape((28,28))

plt.imshow(img,cmap='gray')

plt.title(train.iloc[2,0])

plt.axis("off")

plt.show()
X_train = X_train / 255.0
X_train = X_train.values.reshape(-1,28,28,1)

print("x_train shape: ",X_train.shape)
Y_train.head()
Y_train = pd.Categorical(Y_train)

Y_train = pd.get_dummies(Y_train)
Y_train.values
from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)
from sklearn.metrics import confusion_matrix,accuracy_score

import itertools



from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop,Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau



model = Sequential()
#Conv layer

model.add(Conv2D(filters = 8,##Feature detection

                 kernel_size = (5,5),##n*n

                 padding = 'Same',##Filtre turu. Veri kaybini engeller.

                 activation ='relu', 

                 input_shape = (28,28,1)))##Keras 3 boyut istiyor.

#Pooling layer

model.add(MaxPool2D(pool_size=(2,2)))## Veri boyutunu dusurur. Overfittingi engeller.

#Conv katmaninda kernele 5*5 verdik fakat pooling ile tararken 2*2 olarak tarayacak.

model.add(Dropout(0.25))##Yine overfittingi azaltmak adina bir dropout.



#Ayni islemler tekrardan gerceklesiyor.

#Ilk Convolution layerda input shape vermistik. Sonrakilerde gerek yok.

model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2),

                    strides=(2,2)))#Shiftleme yaparken kacar kacar atlayacagi.

model.add(Dropout(0.25))



########################

model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2),

                    strides=(2,2)))#Shiftleme yaparken kacar kacar atlayacagi.

model.add(Dropout(0.25))

#########################

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2),

                    strides=(2,2)))#Shiftleme yaparken kacar kacar atlayacagi.

model.add(Dropout(0.25))

#########################



#########################



model.add(Flatten())#Input sekline gelebilmesi icin son islem.Matrisi inputun anlayacagi boyuta indirgiyor.



model.add(Dense(256, activation = "relu"))#Hidden layer

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))#Multi output classification icin tercih edilir. sigmoid ise binary classification

                                            #icin kullanilir.

optimizer = Adam(lr=0.001,#learning rate

                 beta_1=0.9, beta_2=0.999)#lr degisimini etklieyen parametreler.
model.compile(optimizer = optimizer ,#bir ustte hazirlamistik.

              loss = "categorical_crossentropy",#ikiden cok siniflandirma oldugunda categorical tercih edilebilir. Fakat ikili 

                                                #siniflandirmada binary tercih edilir.

              metrics=["accuracy"])#Degerlendirme teknigi.
epochs = 60

batch_size = 250
datagen = ImageDataGenerator(

        featurewise_center=False,  # input ortalamasini 0'a esitler.

        samplewise_center=False,  # her ornegi 0'a esitler.

        featurewise_std_normalization=False,  # inputu genel std degerine boler.

        samplewise_std_normalization=False,  # her inputu kendi std degerine boler.

        zca_whitening=False,  # boyut azaltma

        rotation_range=0.5,  # icerigi rotasyon yaptirir. Buradaki 5 derece.

        zoom_range = 0.5, # icerige zoom yapar. Buradaki %5

        width_shift_range=0.5,  # icerigi yatay duzlemde %5 kaydirir.

        height_shift_range=0.5,  # icerigi dikey duzlemde %5 kaydirir.

        horizontal_flip=False,  # yatay dondurme

        vertical_flip=False)  # dikey dondurme



datagen.fit(X_train)
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_val,Y_val), steps_per_epoch=X_train.shape[0] // batch_size)
plt.plot(history.history['val_loss'], color='b', label="validation loss")

plt.title("Loss degeri testi")

plt.xlabel("Epochs sayisi")

plt.ylabel("Loss degeri")

plt.legend()

plt.show()
Y_pred = model.predict(X_val)
Y_pred
Y_pred_degerler = np.argmax(Y_pred,axis = 1)
Y_gercek_degerler = np.argmax(Y_val.values,axis = 1)
confusion_mtx = confusion_matrix(Y_gercek_degerler, Y_pred_degerler) 

f,ax = plt.subplots(figsize=(8, 8))

sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)

plt.xlabel("Tahmin edilen deger")

plt.ylabel("Gercek deger")

plt.title("")

plt.show()
print(accuracy_score(Y_pred_degerler,Y_gercek_degerler))
from tensorflow.keras.preprocessing.image import load_img,img_to_array,array_to_img 
def load_image(filename):

    img = load_img(filename,color_mode = "grayscale",target_size=(28, 28))

    plt.imshow(img,cmap='Greys')

    img = img_to_array(img)

    img = img.reshape(1, 28, 28, 1)

    img = img.astype('float32')

    img = img / 255.0

    return img
quary_img=load_image("../input/sayilar/3.png")

pred = model.predict(quary_img)

print(" Predict :",pred.argmax())
quary_img=load_image("../input/sayilar/7.png")

pred = model.predict(quary_img)

print(" Predict :",pred.argmax())
quary_img=load_image("../input/sayilar/0.png")

pred = model.predict(quary_img)

print(" Predict :",pred.argmax())
quary_img=load_image("../input/sayilar/2.png")

pred = model.predict(quary_img)

print(" Predict :",pred.argmax())
quary_img=load_image("../input/sayilar/9.png")

pred = model.predict(quary_img)

print(" Predict :",pred.argmax())
quary_img=load_image("../input/sayilar/1.png")

pred = model.predict(quary_img)

print(" Predict :",pred.argmax())
quary_img=load_image("../input/sayilar/3H.png")

pred = model.predict(quary_img)

print(" Predict :",pred.argmax())
quary_img=load_image("../input/sayilar/6H.png")

pred = model.predict(quary_img)

print(" Predict :",pred.argmax())