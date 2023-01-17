# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plot 

from sklearn.model_selection import train_test_split 

from sklearn.metrics import confusion_matrix







import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.utils import to_categorical

from keras.optimizers import RMSprop

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

#read dataset

train_set = pd.read_csv('../input/train.csv')

test_set = pd.read_csv('../input/test.csv')
print(train_set.shape)

print(test_set.shape) #one less column

train_set.head()
x_train = train_set.iloc[:,1:]

y_train = train_set.loc[:,['label']]

#Getting the train and test set

X_train,X_test,y_train,y_test = train_test_split(x_train, y_train,test_size=0.3,shuffle = False)
X_train/=255

X_test/=255

test_set/=255
print(X_train.shape[0])

print(y_train.shape)
#for conv net

X_train = X_train.values.reshape(X_train.shape[0],28,28,1)

X_test = X_test.values.reshape(X_test.shape[0],28,28,1)

val_test = test_set.values.reshape(test_set.shape[0],28,28,1)



print(X_train.shape)

print(y_train.shape)
#visual = X_train.values.reshape(-1,28,28,1)

g = plt.imshow(X_train[0][:,:,0])

print(y_train.values[0])

#one hot encoder

y_train =  to_categorical(y_train, 10)

y_test =  to_categorical(y_test, 10)

#image argumentation

from keras.preprocessing.image import ImageDataGenerator

X_train2 = np.array(X_train, copy=True) 

y_train2 = np.array(y_train, copy=True) 



datagen = ImageDataGenerator(

    featurewise_center=True,

    featurewise_std_normalization=True,

    rotation_range=20,

    )



datagen.fit(X_train)



print(type(X_train2))

print(type(X_train))



# Concatenating

result_x  = np.concatenate((X_train, X_train2), axis=0)

result_y  = np.concatenate((y_train, y_train2), axis=0)
print(X_train.shape)

print(result_x.shape)
def mlp_nn():

    model = Sequential()

    model.add(Dense(64 ,activation='relu', input_dim =784))

    model.add(Dense(10,activation='softmax'))

    return model
def conv_nn():

    from keras.layers import Conv2D, MaxPooling2D, Flatten

    model = Sequential()

    model.add(Conv2D(32,(3,3),activation='relu', input_shape =(28,28,1)))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Dropout(0,4))

    model.add(Conv2D(64,(3,3),activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Dropout(0,4))

    model.add(Flatten())

    model.add(Dense(128 ,activation='relu'))

    model.add(Dropout(0,5))

    model.add(Dense(128 ,activation='relu'))

    model.add(Dropout(0,5))

    model.add(Dense(10,activation='softmax'))

    return model
model = conv_nn()

model.compile(optimizer=RMSprop(),

 loss='categorical_crossentropy',

 metrics=['accuracy'])

model.summary()
model.fit(result_x,result_y,epochs=12,batch_size=128,validation_data=(X_test,y_test))

prediction = model.evaluate(X_test,y_test,batch_size=32)
model.predict_classes(val_test, batch_size=32)
g2 = plt.imshow(val_test[-3][:,:,0])

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

end_file = model.predict_classes(val_test, batch_size=32)

results = pd.Series(end_file,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("mnist_dense.csv",index=False)