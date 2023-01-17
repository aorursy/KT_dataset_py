import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns



%matplotlib inline  

style.use('fivethirtyeight')

sns.set(style='whitegrid',color_codes=True)



from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import LabelEncoder



from keras.preprocessing.image import ImageDataGenerator



from keras import backend as K

from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop

from keras.utils import to_categorical



from keras.layers import Dropout, Flatten,Activation

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

 

import tensorflow as tf

import random as rn



import cv2                  

import numpy as np  

from tqdm import tqdm

import os                   

from random import shuffle  

from zipfile import ZipFile

from PIL import Image



import warnings

warnings.filterwarnings('always')

warnings.filterwarnings('ignore')
X=[]

Z=[]

IMG_SIZE=150

BUOY_DIR='../input/boat-types-recognition/buoy'

CRUISE_SHIP_DIR='../input/boat-types-recognition/cruise ship'

FERRY_BOAT_DIR='../input/boat-types-recognition/ferry boat'

FREIGHT_BOAT_DIR='../input/boat-types-recognition/freight boat'

GONDOLA_DIR='../input/boat-types-recognition/gondola'

INFLATABLE_BOAT_DIR='../input/boat-types-recognition/inflatable boat'

KAYAK_DIR='../input/boat-types-recognition/kayak'

PAPER_BOAT_DIR='../input/boat-types-recognition/paper boat'

SAILBOAT_DIR='../input/boat-types-recognition/sailboat'
def check_files_extension(DIR):

    result_list = list()

    for img in os.listdir(DIR):

        filename, file_extension = os.path.splitext(img)

        result_list.append(file_extension)

    myset = set(result_list)

    print(myset)

    

check_files_extension(BUOY_DIR)

check_files_extension(CRUISE_SHIP_DIR)

check_files_extension(FERRY_BOAT_DIR)

check_files_extension(FREIGHT_BOAT_DIR)

check_files_extension(GONDOLA_DIR)

check_files_extension(INFLATABLE_BOAT_DIR)

check_files_extension(KAYAK_DIR)

check_files_extension(PAPER_BOAT_DIR)

check_files_extension(SAILBOAT_DIR)
def is_correct_file(file_name):

    filename, file_extension = os.path.splitext(file_name)

    is_file = os.path.isfile(file_name)

    is_image = file_extension.lower() == ".jpg"

    return is_file and is_image
def assign_label(img,boat_type):

    return boat_type



def make_train_data(boat_type,DIR):

    for img in tqdm(os.listdir(DIR)):

        label=assign_label(img,boat_type)

        path = os.path.join(DIR,img)

        if is_correct_file(path):

            img = cv2.imread(path,cv2.IMREAD_COLOR)

            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))



            X.append(np.array(img))

            Z.append(str(label))

        

make_train_data('Buoy',BUOY_DIR)

print(len(X))



make_train_data('Cruise ship',CRUISE_SHIP_DIR)

print(len(X))



make_train_data('Ferry boat',FERRY_BOAT_DIR)

print(len(X))



make_train_data('Freight boat',FREIGHT_BOAT_DIR)

print(len(X))



make_train_data('Gondola',GONDOLA_DIR)

print(len(X))



make_train_data('Inflatable boat',INFLATABLE_BOAT_DIR)

print(len(X))



make_train_data('Kayak',KAYAK_DIR)

print(len(X))



make_train_data('Paper boat',PAPER_BOAT_DIR)

print(len(X))



make_train_data('Sailboat',SAILBOAT_DIR)

print(len(X))
fig,ax=plt.subplots(5,5)

fig.set_size_inches(15,15)

for i in range(5):

    for j in range (5):

        l=rn.randint(0,len(Z))

        ax[i,j].imshow(X[l])

        ax[i,j].set_title('Boat: '+Z[l])

        

plt.tight_layout()
le=LabelEncoder()

Y=le.fit_transform(Z)

Y=to_categorical(Y,9)

X=np.array(X)

X=X/255
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=42)
np.random.seed(42)

rn.seed(42)

tf.random.set_seed(42)
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

model.add(Dense(9, activation = "softmax"))
batch_size=128

epochs=70



from keras.callbacks import ReduceLROnPlateau

red_lr= ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.1)
datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.05, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=True,  # randomly flip images

        vertical_flip=False)  # randomly flip images





datagen.fit(x_train)
model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()
#History = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),

#                              epochs = epochs, validation_data = (x_test,y_test),

#                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size)



History = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=x_train.shape[0] // batch_size, 

                    epochs=epochs, validation_data = (x_test,y_test), verbose = 1, callbacks=red_lr)
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
pred=model.predict(x_test)

pred_digits=np.argmax(pred,axis=1)
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

        ax[i,j].set_title("Predicted boat : "+str(le.inverse_transform([pred_digits[prop_class[count]]])))

        plt.tight_layout()

        count+=1
warnings.filterwarnings('always')

warnings.filterwarnings('ignore')



count=0

fig,ax=plt.subplots(4,2)

fig.set_size_inches(15,15)

for i in range (4):

    for j in range (2):

        ax[i,j].imshow(x_test[mis_class[count]])

        ax[i,j].set_title("Predicted boat : "+str(le.inverse_transform([pred_digits[mis_class[count]]])))

        plt.tight_layout()

        count+=1