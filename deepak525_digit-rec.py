# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#import the necessary libraries

import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.optimizers import Adam,RMSprop

from keras.layers.normalization import BatchNormalization

from keras.utils import np_utils,plot_model

from keras.preprocessing.image import ImageDataGenerator

from IPython.display import Image

from keras.layers import Conv2D, MaxPooling2D

from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,EarlyStopping

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,accuracy_score

from keras.datasets import mnist



#set the backgroung style sheet

sns.set_style("whitegrid")

import os

print(os.listdir("../input"))



import warnings

warnings.filterwarnings('ignore')



# Any results you write to the current directory are saved as output.
#load the csv file in a dataframe using read_csv function

df = pd.read_csv('../input/digit-recognizer/train.csv')

test_df = pd.read_csv('../input/digit-recognizer/test.csv') 

df.info()
df1_train = pd.read_csv('../input/mnist-in-csv/mnist_train.csv')

df1_train.shape
X = df.drop("label",axis=1)

y = df['label']

X_1 = df1_train.drop("label",axis=1)

y_1 = df1_train['label']
X = X / 255.0

X_1 = X_1 / 255.0

test_df = test_df / 255.0
X = np.concatenate((X, X_1), axis=0)

y = np.concatenate((y, y_1), axis=0)
X = pd.DataFrame(X)

y = pd.DataFrame(y)



x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.15,random_state=0)

x_train = x_train.values.reshape(-1,28,28,1)

x_test = x_test.values.reshape(-1,28,28,1)

test_df = test_df.values.reshape(-1,28,28,1)
model = Sequential()



model.add(Conv2D(64, (3, 3), input_shape=(28,28,1),padding="SAME"))

model.add(BatchNormalization(axis=-1))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3, 3),padding="SAME"))

model.add(BatchNormalization(axis=-1))

model.add(Activation('relu'))



model.add(Conv2D(128,(3, 3),padding="SAME"))

model.add(BatchNormalization(axis=-1))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, (3, 3),padding="SAME"))

model.add(BatchNormalization(axis=-1))

model.add(Activation('relu'))



model.add(Conv2D(192,(3, 3),padding="SAME"))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(192,(5, 5),strides=2,padding="SAME"))

model.add(Activation('relu'))



model.add(Flatten())



# Fully connected layer

model.add(Dense(256))

# model.add(Dense(256))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(0.3))

model.add(Dense(10))



model.add(Activation('softmax'))
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',patience=3, verbose=1,factor=0.5,min_lr=0.00001)

best_model = ModelCheckpoint('mnist_weights.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')



early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=10,restore_best_weights=True)
model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
aug = ImageDataGenerator(

    featurewise_center=False,  

        samplewise_center=False,  

        featurewise_std_normalization=False,  

        samplewise_std_normalization=False, 

        rotation_range=10, 

        zoom_range = 0.,

        width_shift_range=0.1,  

        height_shift_range=0.1,  

        horizontal_flip=False,

        vertical_flip=False)



aug.fit(x_train)
h = model.fit_generator(

    aug.flow(x_train, y_train, batch_size=64),

    validation_data=(x_test, y_test),

    steps_per_epoch=len(x_train) // 64,

    epochs=50, verbose=1,

    callbacks=[learning_rate_reduction,best_model,early_stopping]

    )



# h = model.fit(x_train,y_train,validation_data = (x_test,y_test),epochs=50,batch_size=100,

#                  callbacks=[learning_rate_reduction,best_model,early_stopping],shuffle=True)
pd.DataFrame(h.history).plot()
y_pred = model.predict(x_test)

y_pred = np.argmax(y_pred,axis = 1)

accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test,y_pred)

f,ax = plt.subplots(figsize=(7, 7))

sns.heatmap(cm, cmap='Blues',annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.figure(figsize=(16,10))

count = 1

y_true = list(y_test.values)

for i in range(len(y_pred)):

    if count==11:

        break

    if y_true[i][0]!=y_pred[i]:

        plt.subplot(2,5,count)

        plt.imshow(x_test[i].reshape(28,28),cmap=plt.cm.binary)

        plt.xlabel("Predicted label :{}\nTrue label :{}".format(y_pred[i],y_true[i][0]))

        count+=1
result = model.predict(test_df)

results = np.argmax(result,axis = 1)

results
Label = pd.Series(results,name = 'Label')

ImageId = pd.Series(range(1,28001),name = 'ImageId')

submission = pd.concat([ImageId,Label],axis = 1)

submission.to_csv('submission.csv',index = False)
model.summary()
model_json = model.to_json()

with open("model.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

model.save_weights("model.h5")

print("Saved model to disk")