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
#importing Libraries

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

#from keras.preprocessing.image import ImageDataGenerator

#from keras.callbacks import LearningRateScheduler
training_set = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

training_set.head()
training_set.shape
test_set = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

test_set.head()
test_set.shape
submission = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")

submission.head(10)
#preprocessing

Ytrain = training_set['label'].values

Xtrain = training_set.drop(labels = ['label'], axis = 1).values

Xtest = test_set.values
Ytrain.shape
Ytrain[3]
Ytrain = to_categorical(Ytrain, num_classes = 10)
Ytrain.shape
Ytrain[3]
import matplotlib.pyplot as plt
Xtrain[0]
Xtrain = Xtrain.reshape(42000, 28, 28, 1)

Xtest = Xtest.reshape(28000, 28, 28, 1)
for i in range(0, 10):

    plt.figure()

    plt.imshow(Xtrain[i].reshape(28,28) , cmap = plt.cm.binary)
# BUILD CONVOLUTIONAL NEURAL NETWORKS

'''model = Sequential()

model.add(Conv2D(64, kernel_size = 3, activation='relu', input_shape = (28, 28, 1)))

model.add(Conv2D(64, kernel_size = 3, activation='relu'))

model.add(Conv2D(32, kernel_size = 3, activation='relu'))

model.add(Flatten())

model.add(Dense(10, activation='softmax'))'''



model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))





model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))
# COMPILE WITH ADAM OPTIMIZER AND CROSS ENTROPY COST

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
# TRAIN NETWORKS

epoch = 20

x_train, x_test, y_train, y_test = train_test_split(Xtrain, Ytrain, test_size = 0.1)

model.fit(x_train,y_train, validation_data = (x_test, y_test),epochs = epoch)

#print("CNN {0:d}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}"

 #     .format(j+1,epochs,max(history[j].history['accuracy']),max(history[j].history['val_accuracy']) ))
# ENSEMBLE PREDICTIONS AND SUBMIT



results = model.predict(Xtest)

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("submission_digit2.csv",index=False)