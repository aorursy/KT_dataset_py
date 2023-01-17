# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

%load_ext tensorboard



%tensorboard --logdir logs

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#reading in csv files

train_val_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')



#show small part of training & validation dataframe

train_val_data.head()
train_val_data['label'].value_counts()
#number of each label in the dataset

sns.countplot(train_val_data['label'])
#checking for null data in training/validation set

train_val_data.isnull().any().describe()
#null data in test set?

test_data.isnull().any().describe()
#x and y for training

y_train = train_val_data['label']

x_train = train_val_data.drop(labels = ['label'], axis = 1).values
import matplotlib.pyplot as plt

#the first 4 images in the training set are

fig, ax = plt.subplots(1,4, figsize = (15,6))

for i in range(4):

    ax[i].imshow(x_train[i].reshape(28,28), cmap = 'gray')

    ax[i].set_title('Label: ' +  str(train_val_data.iloc[i]['label']))

plt.show()
x_train.shape
#reshaping into images

x_train = x_train.reshape(x_train.shape[0],28,28,1)

x_train.shape
#test data shape

x_test = test_data.values

x_test.shape
#reshaping test data

x_test = x_test.reshape(x_test.shape[0],28,28,1)

x_test.shape
#normalize data

x_train = x_train/255

x_test = x_test/255
#one hot encoding the train labels

from keras.utils import to_categorical

y_train = to_categorical(y_train, num_classes = 10)
#splitting into training and validation data

from sklearn.model_selection import train_test_split

X_train, X_val, Y_train,Y_val = train_test_split(x_train,y_train,test_size = 0.2, random_state = 2)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D,Dense,Flatten,MaxPooling2D,BatchNormalization

from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping

#define model

model = Sequential()

model.add(Conv2D(32, (5,5), input_shape = (28,28,1), activation = 'relu' ))

model.add(Conv2D(32, (5,5), activation = 'relu'))

model.add(Conv2D(32, (5,5), activation = 'relu'))

model.add(MaxPooling2D(2,2))

model.add(BatchNormalization())

model.add(Conv2D(64, (3,3), activation = 'relu'))

model.add(Conv2D(64, (3,3), activation = 'relu'))

model.add(Conv2D(64, (3,3), activation = 'relu'))

model.add(MaxPooling2D(2,2))

model.add(BatchNormalization())

model.add(Flatten())

model.add(Dense(512, activation = 'relu'))

model.add(BatchNormalization())

model.add(Dense(10, activation = 'softmax'))



red_lr_plat = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

tb = TensorBoard("logs")

early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0,patience=2,verbose=0, mode='auto')
#compile model

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',metrics = ['accuracy'])
#fit model

history = model.fit(x = X_train, y = Y_train, epochs = 10, validation_data = (X_val,Y_val), callbacks = [red_lr_plat, tb, early_stopping])
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Accuracy vs Epochs')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend(['Train', 'Val'], loc = 'upper right')

plt.show()
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Loss vs Epochs')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend(['Train', 'Val'], loc = 'upper right')

plt.show()
results = model.predict(x = x_test)
results = np.argmax(results, axis = 1)

submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

submission['Label'] = results

submission.to_csv('my_submission.csv', index = False)