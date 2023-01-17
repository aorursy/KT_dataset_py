import numpy as np 

import keras

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns

from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import RMSprop,Adam

from keras.layers import Conv2D, MaxPooling2D,BatchNormalization,Dropout

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

from random import shuffle  

from PIL import Image





%matplotlib inline  

style.use('fivethirtyeight')

sns.set(style='whitegrid',color_codes=True)



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import LabelEncoder

from keras.utils import to_categorical



print(os.listdir("../input"))
X=[]

Y=[]

img_size=150

pantolonlar= '../input/fotograflar/fotograflar/pantolon'

tshirt= '../input/fotograflar/fotograflar/tshirt'
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
make_train_data('pantolonlar',pantolonlar)

print(len(X))
make_train_data('tshirt',tshirt)

print(len(X))
X=np.array(X)

X=X/255

Y = np.asarray(Y)
pd.unique(Y)
x_train,x_test,y_train,y_test= train_test_split(X,Y,test_size=0.20,random_state=42)
le=LabelEncoder()

y_test=le.fit_transform(y_test)

y_train=le.fit_transform(y_train)


y_train_binary=to_categorical(y_train,5)

y_test_binary=to_categorical(y_test,5)
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3,3),activation ='relu', input_shape = (img_size,img_size,3)))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 64, kernel_size = (3,3),activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))

 

model.add(Conv2D(filters =64, kernel_size = (3,3),activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 32, kernel_size = (3,3),activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dropout(0.5))

model.add(Activation('relu'))

model.add(Dense(512,activation='relu'))

model.add(Dense(5, activation = "softmax"))



batch_size=10

epochs=5
from keras import optimizers



model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-4),

              metrics=['acc'])
datagen = ImageDataGenerator(

        

        rotation_range=40,

        zoom_range = 0.2, 

        width_shift_range=0.2,  

        height_shift_range=0.2,

        shear_range=0.2,   

        horizontal_flip=True,

        fill_mode='nearest')  

datagen.fit(x_train)

model.compile(optimizer=Adam(lr=0.001),loss='binary_crossentropy',metrics=['accuracy'])
model.summary()
History = model.fit_generator(datagen.flow(x_train,y_train_binary, batch_size=batch_size),

                              epochs = epochs, validation_data = (x_test,y_test_binary),

                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size)
model.save('dlproject_model.h5')
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

pred = model.predict_classes(x_test)

cm = confusion_matrix(y_test,pred)



f,ax = plt.subplots(figsize=(8, 8))

sns.heatmap(cm, annot=True, linewidths=0.01,cmap="Blues",linecolor="yellow", fmt= '.1f',ax=ax)

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.title("Confusion Matrix")

plt.show()
crlr1 =classification_report(y_test,pred)

print(crlr1)

acclr1 =accuracy_score(y_test,pred) 

print("Accuracy Score:",+acclr1)
import matplotlib.pyplot as plt



acc = History.history['acc']

val_acc = History.history['val_acc']

loss = History.history['loss']

val_loss = History.history['val_loss']



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
def plot_model_history(model_history):

    fig, axs = plt.subplots(1,2,figsize=(15,5))

    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])

    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])

    axs[0].set_title('Model Accuracy')

    axs[0].set_ylabel('Accuracy')

    axs[0].set_xlabel('Epoch')

    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)

    axs[0].legend(['train', 'val'], loc='best')

    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])

    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])

    axs[1].set_title('Model Loss')

    axs[1].set_ylabel('Loss')

    axs[1].set_xlabel('Epoch')

    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)

    axs[1].legend(['train', 'val'], loc='best')

    plt.show()

plot_model_history(History)
pred=model.predict(x_test)

pred_digits=np.argmax(pred,axis=1)
true=0

false=0

for i in range(len(pred)):

   if(pred_digits[i]==y_test[i]):

    true=true+1

   else:

    false=false+1

print("Doğru Tahmin Sayısı:",+true)

print("Yanlış Tahmin Sayısı:",+false)
count=0

fig,ax=plt.subplots(4,2)

fig.set_size_inches(17,17)

for i in range (4):

    for j in range (2):

        ax[i,j].imshow(x_test[prop_class[count]])

        ax[i,j].set_title("Predicted Object : "+str(le.inverse_transform([pred_digits[prop_class[count]]]))+"\n"+"Actual Object : "+str(le.inverse_transform([y_test[prop_class[count]]])))

        plt.tight_layout()

        count+=1
count=0

fig,ax=plt.subplots(3,2)

fig.set_size_inches(17,17)

for i in range (3):

    for j in range (2):

        ax[i,j].imshow(x_test[mis_class[count]])

        ax[i,j].set_title("Predicted Object : "+str(le.inverse_transform([pred_digits[mis_class[count]]]))+"\n"+"Actual Object : "+str(le.inverse_transform([y_test[mis_class[count]]])))

        plt.tight_layout()

        count+=1
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3,3),activation ='relu', input_shape = (img_size,img_size,3)))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 64, kernel_size = (3,3),activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))

 

model.add(Conv2D(filters =64, kernel_size = (3,3),activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 32, kernel_size = (3,3),activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dropout(0.5))

model.add(Activation('relu'))

model.add(Dense(512,activation='relu'))

model.add(Dense(5, activation = "softmax"))



batch_size=10

epochs=3
from keras import optimizers



model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-4),

              metrics=['acc'])
datagen = ImageDataGenerator(

        

        rotation_range=40,

        zoom_range = 0.2, 

        width_shift_range=0.2,  

        height_shift_range=0.2,

        shear_range=0.2,   

        horizontal_flip=True,

        fill_mode='nearest')  

datagen.fit(x_train)
model.compile(optimizer=Adam(lr=0.001),loss='binary_crossentropy',metrics=['accuracy'])
model.summary()
History = model.fit_generator(datagen.flow(x_train,y_train_binary, batch_size=batch_size),

                              epochs = epochs, validation_data = (x_test,y_test_binary),

                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size)
def plot_model_history(model_history):

    fig, axs = plt.subplots(1,2,figsize=(15,5))

    # summarize history for accuracy

    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])

    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])

    axs[0].set_title('Model Accuracy')

    axs[0].set_ylabel('Accuracy')

    axs[0].set_xlabel('Epoch')

    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)

    axs[0].legend(['train', 'val'], loc='best')

    # summarize history for loss

    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])

    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])

    axs[1].set_title('Model Loss')

    axs[1].set_ylabel('Loss')

    axs[1].set_xlabel('Epoch')

    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)

    axs[1].legend(['train', 'val'], loc='best')

    plt.show()

plot_model_history(History)