import pandas as pd

import numpy as  np

import matplotlib.pyplot as plt

import seaborn as sns







from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



from keras.utils.np_utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,BatchNormalization

from keras.optimizers import RMSprop,Adam

from keras.callbacks import ReduceLROnPlateau
import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow import keras



img_rows, img_cols = 28, 28

num_classes = 10



def prep_data(raw):

    y = raw[:, 0]

    out_y = keras.utils.to_categorical(y, num_classes)

    

    x = raw[:,1:]

    num_images = raw.shape[0]

    out_x = x.reshape(num_images, img_rows, img_cols, 1)

    out_x = out_x / 255

    return out_x, out_y



fashion_file = "../input/Kannada-MNIST/train.csv"

fashion_data = np.loadtxt(fashion_file, skiprows=1, delimiter=',')

x, y = prep_data(fashion_data)



print("Setup Complete")
from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Conv2D





# Your Code Here

model=Sequential()
model.add(Conv2D(12, kernel_size=(3,3),

                 activation='relu',

                 input_shape=(img_rows, img_cols,1)))
model.add(Conv2D(20, activation='relu', kernel_size=3))

model.add(Conv2D(20, activation='relu', kernel_size=3))

model.add(Flatten())

model.add(Dense(100, activation='relu'))

model.add(Dense(10, activation='softmax'))
model.compile(loss = "categorical_crossentropy",

             optimizer = 'adam',

             metrics = ['accuracy'])
model.fit(x,y,

                 batch_size = 100,

                 epochs = 4,

                 validation_split = 0.2)
#define model name

second_model=Sequential()



#fist lay

second_model.add(Conv2D(17, kernel_size=(3,3),

                 activation='relu',

                 input_shape=(img_rows, img_cols,1)))



#the remaining layers

second_model.add(Conv2D(50, activation='relu', kernel_size=3))

second_model.add(Conv2D(50, activation='relu', kernel_size=3))

second_model.add(Flatten())

second_model.add(Dense(100, activation='relu'))

second_model.add(Dense(10, activation='softmax'))



#compile

second_model.compile(loss = "categorical_crossentropy",

             optimizer = 'adam',

             metrics = ['accuracy'])



#fit/train model

second_model.fit(x,y,

                 batch_size = 100,

                 epochs = 7,

                 validation_split = 0.2)

train=pd.read_csv('../input/Kannada-MNIST/train.csv')

test=pd.read_csv('../input/Kannada-MNIST/test.csv')

sample_sub=pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')
#print('The Train  dataset has {} rows and {} columns'.format(train.shape[0],train.shape[1]))

#print('The Test  dataset has {} rows and {} columns'.format(test.shape[0],test.shape[1]))

#train.head(3)
#test.head(3)

#test=test.drop('id',axis=1)

test=pd.read_csv('../input/Kannada-MNIST/test.csv')


test=test.drop('id',axis=1)

test=test/255

test=test.values.reshape(-1,28,28,1)
test.shape
y_pre=second_model.predict(test)     ##making prediction

y_pre=np.argmax(y_pre,axis=1) ##changing the prediction intro labels
sample_sub['label']=y_pre

sample_sub.to_csv('submission.csv',index=False)
sample_sub.head()