import warnings

warnings.filterwarnings('always')

warnings.filterwarnings('ignore')



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import LabelEncoder



from tensorflow.keras.preprocessing.image import ImageDataGenerator



from tensorflow.keras import backend as K

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop

from tensorflow.keras.utils import to_categorical



from tensorflow.keras.layers import Dropout, Flatten,Activation

from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization

 

import tensorflow as tf

import random as rn



tf.keras.backend.clear_session()



import cv2                  

import numpy as np  

from tqdm import tqdm

import os                   

from random import shuffle  

from zipfile import ZipFile

from PIL import Image
X=[]

Z=[]



# Dimensiunea pozelor

IMG_SIZE=150



# Dataset

FLOWER_ALCEAROSEA_DIR='/kaggle/input/flowers/flowers/alcea_rosea'

FLOWER_MATRICARIACHAMOMILLA_DIR='/kaggle/input/flowers/flowers/matricaria_chamomilla'

FLOWER_CALENDULAOFFICINALIS_DIR='/kaggle/input/flowers/flowers/calendula_officinalis'

FLOWER_RUDBECKIATRILOBA_DIR='/kaggle/input/flowers/flowers/rudbeckia_triloba'



# functie ce returneaza tipul de planta

def assign_label(img,flower_type):

    return flower_type



# functie de pregatire a datelor de antrenament

def train_data(flower_type,DIR):

    for img in tqdm(os.listdir(DIR)):

        try:

            label=assign_label(img,flower_type)

            path = os.path.join(DIR,img)

            img = cv2.imread(path,cv2.IMREAD_COLOR)

            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))

        

            X.append(np.array(img))

            Z.append(str(label))

        except Exception as e:

            print(str(e))
train_data('alcea_rosea',FLOWER_ALCEAROSEA_DIR)

train_data('matricaria_chamomilla',FLOWER_MATRICARIACHAMOMILLA_DIR)

train_data('calendula_officinalis',FLOWER_CALENDULAOFFICINALIS_DIR)

train_data('rudbeckia_triloba',FLOWER_RUDBECKIATRILOBA_DIR)
# Plotam o serie de imagini

fig,ax=plt.subplots(5,2)

fig.set_size_inches(15,15)

for i in range(5):

    for j in range (2):

        l=rn.randint(0,len(Z))

        ax[i,j].imshow(X[l])

        ax[i,j].set_title('Flower: '+Z[l])

plt.tight_layout()
# Codam imaginile

le=LabelEncoder()



Y=le.fit_transform(Z)

Y=to_categorical(Y,5)

X=np.array(X)

X=X/255
# Impartim dataset-ul in date de antrenare si date de test

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=42)



# Folosim seed pentru a produce rezultate reproductibile

np.random.seed(42)

rn.seed(42)

tf.random.set_seed(42)
# Construim modelul

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (150,150,3)))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters =96, kernel_size = (3,3),padding = 'Same',activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'Same',activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())

model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dense(5, activation = "softmax"))



batch_size=128

epochs=50
from keras.callbacks import ReduceLROnPlateau

red_lr= ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.1)



# Folosim ImageDataGenerator()

datagen = ImageDataGenerator(

        featurewise_center=False,

        samplewise_center=False,

        featurewise_std_normalization=False,

        samplewise_std_normalization=False,

        zca_whitening=False,

        rotation_range=10,

        zoom_range = 0.1,

        width_shift_range=0.2,

        height_shift_range=0.2,

        horizontal_flip=True,

        vertical_flip=False)

datagen.fit(x_train)
model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()



History = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (x_test,y_test),

                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size)



model.save("model.h5")



plt.plot(History.history['loss'])

plt.plot(History.history['val_loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epochs')

plt.legend(['train', 'test'])

plt.show()



plt.plot(History.history['accuracy'])

plt.plot(History.history['val_accuracy'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epochs')

plt.legend(['train', 'test'])

plt.show()
# Obtinem predictii pe datele de test

pred=model.predict(x_test)

pred_digits=np.argmax(pred,axis=1)

2

i=0

prop_class=[]

mis_class=[]



for i in range(len(y_test)):

    if(np.argmax(y_test[i])==pred_digits[i]):

        prop_class.append(i)

    if(len(prop_class)==8):

        break



i=0

for i in range(len(y_test)):

    if(not np.argmax(y_test[i])==pred_digits[i]):

        mis_class.append(i)

    if(len(mis_class)==8):

        break

    

warnings.filterwarnings('always')

warnings.filterwarnings('ignore')



count=0

fig,ax=plt.subplots(4,2)

fig.set_size_inches(15,15)





for i in range (4):

    for j in range (2):

        ax[i,j].imshow(x_test[prop_class[count]])

        ax[i,j].set_title("Floare prezisa: "+str(le.inverse_transform([pred_digits[prop_class[count]]]))+"\n"+"Floare actuala : "+str(le.inverse_transform([np.argmax(y_test[prop_class[count]])])))

        plt.tight_layout()

        count+=1