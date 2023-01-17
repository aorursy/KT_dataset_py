import pandas as pd

import numpy as np

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import scale



# Feature Dimension Engineering

from sklearn.decomposition import PCA

from sklearn.decomposition import IncrementalPCA

from sklearn.feature_selection import RFE

from statsmodels.stats.outliers_influence import variance_inflation_factor



#

import numpy as np

import pickle

import gzip

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import h5py

import sklearn

import sklearn.datasets

import scipy



from PIL import Image

from scipy import ndimage





from keras.models import Sequential

from keras.layers import Dense, Dropout, BatchNormalization, Activation

from keras import regularizers



np.random.seed(7)



import random

import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras import backend as K

pixel_frame=pd.read_csv('../input/train.csv')
pixel_frame.head()
pixel_frame.shape
pixel_frame.columns
pixel_frame.isnull().sum().reset_index()[0].sort_values(ascending=False)
pixel_frame.head(2)
x=pixel_frame.drop(columns=['label'],axis=1)

y=pixel_frame['label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 4)
x_train=x_train.values

a=np.zeros((x_train.shape[0],28,28))

b=list(range(0,x_train.shape[0]))

for i in b:

    a[i]=a[i]+x_train[i].reshape((28,28))

x_train=a



# apply on x_test

x_test=x_test.values

a=np.zeros((x_test.shape[0],28,28))

b=list(range(0,x_test.shape[0]))

for i in b:

    a[i]=a[i]+x_test[i].reshape((28,28))

x_test=a

# check shapes

print(x_train.shape)

print(x_test.shape)
# specify input dimensions of each image

img_rows, img_cols = 28, 28

input_shape = (img_rows, img_cols, 1)



# batch size, number of classes, epochs

batch_size = 700

num_classes = 10

epochs = 300
# check if y has all values

print(list(np.sort(pixel_frame['label'].unique())))

print(list(np.sort(y_train.unique())))

print(list(np.sort(y_test.unique())))



y_train_set = keras.utils.to_categorical(y_train, num_classes)

y_test_set = keras.utils.to_categorical(y_test, num_classes)

print(y_train_set.shape)

print(y_test_set.shape)
# reshape x_train and x_test

x_train_set = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)

x_test_set = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

print(x_train_set.shape)

print(x_test_set.shape)

x_train_set=x_train_set/255

x_test_set=x_test_set/255
# model

model = Sequential()



# a keras convolutional layer  



# first conv layer

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=input_shape))  



# second conv layer

model.add(Conv2D(64, kernel_size=(3, 3), 

                 activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



# flatten and put a fully connected layer

model.add(Flatten())

model.add(Dense(128, activation='relu')) # fully connected

model.add(Dropout(0.5))



# softmax layer

model.add(Dense(num_classes, activation='softmax'))



# model summary

model.summary()

# usual cross entropy loss

# choose any optimiser such as adam, rmsprop etc

# metric is accuracy

model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adadelta(),

              metrics=['accuracy'])
# fit the model

model.fit(x_train_set, y_train_set,

          batch_size=batch_size,

          epochs=epochs,

          verbose=1,

          validation_data=(x_test_set, y_test_set))
# evaluate the model on test data

print(model.evaluate(x_test_set, y_test_set))

print(model.metrics_names)
pixel_frame=pd.read_csv('../input/test.csv')
x_train=pixel_frame.values

a=np.zeros((x_train.shape[0],28,28))

b=list(range(0,x_train.shape[0]))

for i in b:

    a[i]=a[i]+x_train[i].reshape((28,28))

x_train=a





# check shapes

print(x_train.shape)



x_train_set = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)



print(x_train_set.shape)



x_train_set=x_train_set/255

predictions = model.predict(x_train_set)

predictions = np.argmax(predictions, axis = 1)

predictions
result={'ImageId': range(1,len(predictions)+1) , 'Label':predictions}

result_df=pd.DataFrame(result)

result_df.to_csv('santosh_DigitRecognizer_modified_CNN_.csv')