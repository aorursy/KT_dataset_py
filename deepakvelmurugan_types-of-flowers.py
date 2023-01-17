import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import warnings

warnings.filterwarnings('ignore')

from tensorflow.keras.models import Sequential #CNN model

from tensorflow.keras.layers import Dense,Flatten,MaxPooling2D #Layers

from tensorflow.keras.preprocessing.image import ImageDataGenerator #for data augmentation
#Labels:

print(os.listdir('../input/flowers-recognition/flowers'))
tulip_path = '../input/flowers-recognition/flowers/tulip'

dandelion_path = '../input/flowers-recognition/flowers/dandelion'

sunflower_path = '../input/flowers-recognition/flowers/sunflower'

daisy_path = '../input/flowers-recognition/flowers/daisy'

rose_path = '../input/flowers-recognition/flowers/rose'
from tqdm import tqdm

from keras.preprocessing.image import img_to_array

from keras.preprocessing.image import load_img

flower_types = [tulip_path,dandelion_path,sunflower_path,daisy_path,rose_path]

X_full = []

y_full = []

for flower in flower_types: 

    for i in tqdm(os.listdir(flower)):

        try:

            img = load_img(flower+'/'+i,target_size=(150,150))

        except:

            continue

        img = img_to_array(img)

        X_full.append(np.array(img))

        y_full.append(flower[37:])    
print(len(X_full))

print(len(y_full))
X_full = np.array(X_full)

X_full = X_full.reshape(4323,150,150,3)

X_full = X_full/255
#Visualising

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns

from PIL import Image

import random as rn

%matplotlib inline  





fig,ax=plt.subplots(2,2)

fig.set_size_inches(7,7)

for i in range(2):

    for j in range (2):

        l=rn.randint(0,len(y_full))

        ax[i,j].imshow(X_full[l])

        ax[i,j].set_title('Flower: '+y_full[l])

        

plt.tight_layout()
#One-hot encoding y

from sklearn.preprocessing import OneHotEncoder,LabelEncoder

import tensorflow as tf

LE = LabelEncoder()

y_full = LE.fit_transform(y_full)

#4-tulip,1-dandelion,3-Sunflower,0-Daisy,2-Rose -> after label encoding

y_full = tf.keras.utils.to_categorical(y_full,num_classes=5)
#Train-test split

from sklearn.model_selection import train_test_split

X_train,X_val,y_train,y_val = train_test_split(X_full,y_full,test_size=0.2,random_state=42)
#Create flower model

from tensorflow.keras.layers import Conv2D,Dropout 

num_classes = 5

rows,cols=150,150



#Input layer

flower_model = Sequential()

flower_model.add(Conv2D(64,activation='relu',kernel_size=(3,3),input_shape=(rows,cols,3)))

flower_model.add(MaxPooling2D(pool_size=(2,2)))

#Layer-2

flower_model.add(Conv2D(64,activation='relu',kernel_size=(3,3)))

flower_model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))



#Layer-3

flower_model.add(Conv2D(128,activation='relu',kernel_size=(4,4)))

flower_model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))



#Layer-4

flower_model.add(Conv2D(128,activation='relu',kernel_size=(4,4)))

flower_model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

flower_model.add(Dropout(0.25))

#Flatten

flower_model.add(Flatten())

#Add Dense layer

flower_model.add(Dense(512,activation='relu'))

flower_model.add(Dropout(0.25))

#Output layer

flower_model.add(Dense(num_classes,activation='softmax'))



#Metrics,loss and optimizer for our model

flower_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])



#Data-augmentation to prevent over-fitting

datagen_train = ImageDataGenerator(horizontal_flip=True,width_shift_range=0.2,height_shift_range=0.2)

train_gen = datagen_train.flow(X_train,y_train,batch_size=64)
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping,ModelCheckpoint



best_weights = ModelCheckpoint(filepath='best_weights.hdf5',verbose=2,save_best_only=True)   #Saving best weights to a file

reduce_LR = ReduceLROnPlateau(monitor='val_loss',patience=5,verbose=2,factor=0.2)          #Reducing Learning Rate when val_loss doesn't improve

early_stop = EarlyStopping(monitor='val_loss',patience=20,verbose=2)                         #Stopping model when val_loss doesn't improve
#fit

model_history = flower_model.fit_generator(

                 train_gen,

                 verbose=1,

                 epochs=100,

                 validation_data=(X_val,y_val),

                 callbacks=[best_weights,reduce_LR])
plt.plot(model_history.history['loss'])

plt.plot(model_history.history['val_loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epochs')

plt.legend(['Train', 'Validation'])

plt.show()
plt.plot(model_history.history['accuracy'])

plt.plot(model_history.history['val_accuracy'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epochs')

plt.legend(['Train', 'Validation'])

plt.show()
flower_model.load_weights('best_weights.hdf5')

flower_model.evaluate(X_val,y_val)
#4-tulip,1-dandelion,3-Sunflower,0-Daisy,2-Rose -> after label encoding

target_labels = {}

flower_labels = ['Daisy','Dandelion','Rose','Sunflower','Tulip']

for i in range(0,4):

    target_labels[i] = flower_labels[i]
#Method for preprocessing a single image

def to_img(PATH):

    img = load_img(PATH,target_size=(150,150))

    img = img_to_array(img)

    img = img.reshape(1,150,150,3)

    img = img/255

    return img



#Any flower image from the net

path_of_image = '../input/my-images/rose.jpg'

image_conv = to_img(path_of_image)

predicted_flower = flower_model.predict_classes(image_conv) #Returns the label-encoded value

print("The picture is a:",target_labels[predicted_flower[0]])

plt.imshow(image_conv[0])