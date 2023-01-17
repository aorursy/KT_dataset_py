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
import time

import warnings

#warnings.simplefilter('ignore', FutureWarning)

import numpy as np

import pandas as pd
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

#print(os.listdir("../input"))



import warnings

warnings.filterwarnings('ignore')
import numpy as np

import pandas as pd



# 変数名をそのままprint関数内で表示させる関数

def chkprint(*args):

    for obj in args:

        for k, v in globals().items():

            if id(v) == id(obj):

                target = k

                break          

    return target



# データがどのデータ型か、列数、行数を表示する関数

def typeInfo(targetData):

    if (type(targetData) is pd.core.frame.DataFrame):

        print("{} は DataFrame型".format(chkprint(targetData)))

        print("{} の行数, 列数・・・{}\n".format(chkprint(targetData), targetData.shape))     # shapeの表示内容は、(行数, 列数)となる

    if (type(targetData) is list):

        print("{} は list型".format(chkprint(targetData)))

        print("{} の行数, 列数・・・{}\n".format(chkprint(targetData), pd.DataFrame(targetData).shape))    # shapeの表示内容は、(行数, 列数)となる

    if (type(targetData) is np.ndarray):

        print("{} は ndarray型".format(chkprint(targetData)))

        print("{} の行数, 列数・・・{}\n".format(chkprint(targetData), targetData.shape))     # shapeの表示内容は、(行数, 列数)となる

    if (type(targetData) is pd.core.series.Series):

        print("{} は Series型".format(chkprint(targetData)))

        print("{} の行数, 列数・・・{}\n".format(chkprint(targetData), targetData.shape))  
# 1.加工前の元データを表示させるだけ

# ごちゃごちゃしているのを確認して、どう整理するかを考えるため



train=pd.read_csv("../input/digit-recognizer/train.csv")

submit=pd.read_csv("../input/digit-recognizer/test.csv")

#df1_train=pd.read_csv("../data/mnist/mnist_train.csv")



# print(train.head()) #→ <class 'sklearn.utils.Bunch'>

# print(submit.head())

print(typeInfo(train))

print(typeInfo(submit))

#print(typeInfo(df1_train))

#print(train.dtypes)

# ここでは、まだ sklearn.utils.Bunch 型でも pandas.core.frame.DataFrame 型でも構わない
x_train = train.drop('label', axis=1)

y_train = train['label']                   #y

# x_1 = df1_train.drop("label",axis=1)

# y_1 = df1_train['label']
typeInfo(submit)

typeInfo(train)

#typeInfo(x_1)
# x_train = np.concatenate((x_train, x_1), axis=0)

# y_train = np.concatenate((y_train, y_1), axis=0)



x = x_train

y = y_train



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=0)
x_train = x_train.values.reshape(-1,28,28,1)

x_test = x_test.values.reshape(-1,28,28,1)

submit = submit.values.reshape(-1,28,28,1)
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
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',

                                            patience=3, verbose=1,factor=0.5,min_lr=0.00001)

best_model = ModelCheckpoint('mnist_weights.h5', monitor='val_acc',

                             verbose=1, save_best_only=True, mode='max')



early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-10, 

                               patience=10,restore_best_weights=True)
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

    epochs=20, verbose=1,

    callbacks=[learning_rate_reduction,best_model,early_stopping]

    )
pd.DataFrame(h.history).plot()
y_pred = model.predict(x_test)

y_pred = np.argmax(y_pred,axis = 1)

accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

f, ax = plt.subplots(figsize=(7, 7))

sns.heatmap(cm, cmap='Blues', annot=True, linewidths=.5, fmt= '.1f', ax=ax)
plt.figure(figsize=(16,10))

count = 1

y_true = list(y_test.values)

for i in range(len(y_pred)):

    if count == 11:

        break

    if y_true[i] != y_pred[i]:

        plt.subplot(2, 5, count)

        plt.imshow(x_test[i].reshape(28, 28), cmap = plt.cm.binary)

        plt.xlabel("Predicted label :{}\nTrue label :{}".format(y_pred[i], y_true[i]))

        count+=1
result = model.predict(submit)

results = np.argmax(result,axis = 1)

results
Label = pd.Series(results, name = 'Label')

ImageId = pd.Series(range(1,28001), name = 'ImageId')

submission = pd.concat([ImageId,Label], axis = 1)

submission.to_csv('submission.csv', index = False)