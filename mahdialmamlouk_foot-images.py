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
import tensorflow as tf
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns
%matplotlib inline  

style.use('fivethirtyeight')

sns.set(style='whitegrid',color_codes=True)




#model selection

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import LabelEncoder



#preprocess.

from keras.preprocessing.image import ImageDataGenerator



#dl libraraies

from keras import backend as K

from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop

from keras.utils import to_categorical



# specifically for cnn

from keras.layers import Dropout, Flatten,Activation

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

 

import tensorflow as tf

import random as rn



# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.

import cv2                  

import numpy as np  

from tqdm import tqdm

import os                   

from random import shuffle  

from zipfile import ZipFile

from PIL import Image
X=[]

Z=[]

IMG_SIZE=150

FOOD_BIRIYANI_DIR='../input/recipes/briyani'

FOOD_BURGER_DIR='../input/recipes/burger'

FOOD_DOSA_DIR='../input/recipes/dosa/'

FOOD_IDLY_DIR='../input/recipes/idly'

FOOD_PIZZA_DIR='../input/recipes/pizza/'
def assign_label(img,food_type):

    return food_type
def make_train_data(food_type,DIR):

    for img in tqdm(os.listdir(DIR)):

        label=assign_label(img,food_type)

        path = os.path.join(DIR,img)

        img = cv2.imread(path,cv2.IMREAD_COLOR)

        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))

        

        X.append(np.array(img))

        Z.append(str(label))

   
make_train_data('Biriyani',FOOD_BIRIYANI_DIR)

print(len(X))
make_train_data('Burger',FOOD_BURGER_DIR)

print(len(X))
make_train_data('Dosa',FOOD_DOSA_DIR)

print(len(X))
make_train_data('Idly',FOOD_IDLY_DIR)

print(len(X))
make_train_data('Pizza',FOOD_PIZZA_DIR)

print(len(X))
fig,ax=plt.subplots(5,2)

fig.set_size_inches(15,15)

for i in range(5):

    for j in range (2):

        l=rn.randint(0,len(Z))

        ax[i,j].imshow(X[l])

        ax[i,j].set_title('FOOD: '+Z[l])

        

plt.tight_layout
le=LabelEncoder()

Y=le.fit_transform(Z)

Y=to_categorical(Y,5)

X=np.array(X)

X=X/255
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.20,random_state=1)
np.random.seed(42)

rn.seed(42)
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

batch_size=130

epochs=50



from keras.callbacks import ReduceLROnPlateau

red_lr= ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.1)
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
model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (x_test,y_test),

                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size)

test_loss, test_acc = model.evaluate(x_test , y_test, verbose=2)



print('\nTest accuracy:', test_acc)
model.evaluate(x_test  , y_test , verbose=2)

model.predict(x_test)
model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
model.save('my_model')

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epochs')

plt.legend(['train', 'test'])

plt.show()
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

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
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('always')

warnings.filterwarnings('ignore')



count=0

fig,ax=plt.subplots(4,2)

fig.set_size_inches(15,15)

for i in range (4):

    for j in range (2):

        ax[i,j].set_title("Predicted Food :"+str(le.inverse_transform([pred_digits[prop_class[count]]]))

                          +"\n"+"Actual Food : "+str(le.inverse_transform([np.argmax(y_test[prop_class[count]])])))
! wget https://i.ytimg.com/vi/qh5FCFELyFM/maxresdefault.jpg
