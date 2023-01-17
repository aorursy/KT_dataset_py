# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd



import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout,Flatten

from keras.layers import Conv2D, MaxPool2D

from keras.layers.normalization import BatchNormalization

from keras.optimizers import SGD, RMSprop

from keras.utils.np_utils import to_categorical

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from keras.preprocessing.image import ImageDataGenerator



from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import train_test_split

import itertools



np.random.seed(2)



import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.image as npimg

%matplotlib inline
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
train.shape , test.shape
train.head(5)
test.head(3)
y_train = train['label']

x_train = train.drop(labels = ['label'], axis = 1)



del train
g = sns.countplot(y_train)

x_train.shape , y_train.shape, test.shape, test.size , x_train.size
x_train.isnull().any().describe() ,test.isnull().any().describe()
x_train = x_train / 255.0

test = test / 255.0
x_train = x_train.values.reshape(x_train.shape[0],28,28,1)

test = test.values.reshape(test.shape[0],28,28,1)



x_train.shape, test.shape
y_train = to_categorical(y_train, num_classes = 10)
random_seed = 2

x_train,x_val,y_train,y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state = random_seed)



x_train.shape, x_val.shape, y_train.shape, y_val.shape
model = Sequential()



model.add(Conv2D(32, (5,5) , padding = 'Same', activation ='relu', input_shape = (28,28,1)))

BatchNormalization(axis=1)

model.add(Conv2D(32, (5,5) ,padding = 'Same', activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))





model.add(Conv2D(64, (3,3) ,padding = 'Same', activation ='relu'))

BatchNormalization(axis=1)

model.add(Conv2D(64, (3,3),padding = 'Same', activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))





model.add(Flatten())

BatchNormalization()

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

BatchNormalization()

model.add(Dense(10, activation = "softmax"))



print("input shape ",model.input_shape)

print("output shape ",model.output_shape)

print(model.summary())





#sgd = SGD(lr = 0.01, decay = 0.0, momentum = 0.9, nesterov = True )

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
#checkpoint = ModelCheckpoint( filepath = '/home/savariya/Desktop.hdf5',monitor = 'val_loss', mode = 'min', save_best_only = True, verbose = 1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor = 0.2, patience = 3, verbose = 1, min_delta = 0.0001)

earlystop = EarlyStopping(monitor = 'val_loss',min_delta = 0, patience = 3, verbose = 2, restore_best_weights = True)

epochs = 30

batch_size = 64
history = model.fit(x_train, y_train,  epochs = epochs, batch_size = batch_size, 

                     validation_data = (x_val,y_val),verbose = 2 ,

                    callbacks = [reduce_lr, earlystop])
loss_and_predict = model.evaluate(x_val, y_val, batch_size = batch_size)

y_pred = model.predict_classes(x_val, batch_size = batch_size)

print(loss_and_predict)
print(classification_report(np.argmax(y_val,axis=1),y_pred))

print(confusion_matrix(np.argmax(y_val,axis=1),y_pred))
history_dict  = history.history

loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)



line1 = plt.plot(epochs, val_loss_values, label='Test loss')

line2 = plt.plot(epochs, loss_values, label='Training Loss')



plt.setp(line1, linewidth = 2.0, marker = '+', markersize = 10.0)

plt.setp(line2, linewidth = 2.0, marker = '+', markersize = 10.0)



plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.grid(True)

plt.legend()

plt.show()
history_dict = history.history



acc_values = history_dict['acc']

val_acc_values = history_dict['val_acc']

epochs = range(1, len(acc_values) + 1)



line1 = plt.plot(epochs, val_acc_values, label='Test Accuracy')

line2 = plt.plot(epochs, acc_values, label='Training Accuracy')



plt.setp(line1, linewidth = 2.0, marker = '+', markersize = 10.0)

plt.setp(line2, linewidth = 2.0, marker = '*', markersize = 10.0)



plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.grid(True)

plt.legend()

plt.show()
results = model.predict(test)
results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("keras_data_digit.csv",index=False)