# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten

import tensorflow.keras as ks

%matplotlib inline
from sklearn.utils import shuffle

from tensorflow.keras.utils import to_categorical

#for data extraxtion and preprocessing

def Extract(file,splitFact):

    print("Beginning the extraction process...")

    with open(file,'r') as f:

        data = f.read()

        data = data.split('\n')

    

    x_train = []

    y_train = []

    x_test = []

    y_test = []

    

    data = shuffle(data)

    

    l = len(data)

    

    trainData = int(l * (1-splitFact))

    testData = int(l - trainData)

    

    print("Dividing data into {} training and {} testing samples..".format(trainData,testData))

    

    for i in range(trainData):

        if data[i] != '':

            x_train.append(np.array(data[i].split(',')[1:], dtype=np.float32))

            y_train.append(np.array(data[i].split(',')[0], dtype=np.int8))



    for i in range(testData):

        if data[i+trainData] != '':

            x_test.append(np.array(data[i+trainData].split(',')[1:], dtype=np.float32))

            y_test.append(np.array(data[i+trainData].split(',')[0], dtype=np.int8))

            

    x_train = np.array(x_train)/255

    y_train = to_categorical(np.array(y_train))

    x_test = np.array(x_test)/255

    y_test = to_categorical(np.array(y_test))

    

    del data

    

    return (x_train,y_train),(x_test,y_test)
#Extracting data

(x_train,y_train),(x_test,y_test) = Extract('/kaggle/input/az-handwritten-alphabets-in-csv-format/A_Z Handwritten Data/A_Z Handwritten Data.csv',0.2)



x_train = x_train.reshape(len(x_train),28,28,1)

x_test = x_test.reshape(len(x_test),28,28,1)
labels=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
#lets display some data

for i in range(8):

    plt.subplot(2,4,i+1)

    plt.imshow(x_test[i].reshape(28,28),cmap='binary')

    plt.title(labels[np.argmax(y_test[i])])
#designing the model

model = Sequential()

model.add(Conv2D(filters=16, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(28, 28, 1), name='conv1'))

model.add(MaxPool2D(pool_size=(2, 2,),name='pool1'))

model.add(Dropout(0.2,name='dropout1'))

model.add(Conv2D(filters=32, kernel_size=2, strides=1, padding='same', activation='relu',name='conv2'))

model.add(MaxPool2D(pool_size=(2, 2),name='pool2'))

model.add(Flatten(name='flat'))

model.add(Dropout(0.1,name='dropout2'))

model.add(Dense(128,activation='relu',name='dense'))

model.add(Dense(26,activation='softmax',name='res'))



model.summary()

checkpoint = ks.callbacks.ModelCheckpoint('model.h5',save_best_only=True)

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#training..

history = model.fit(x_train, y_train, validation_data=(x_test,y_test), epochs=19, batch_size=1024, callbacks=[checkpoint], verbose=1)
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')
model = ks.models.load_model('model.h5')

acc = model.evaluate(x_test,y_test)

print(f"Final test loss : {acc[0]}, final test accuracy : {acc[1]}")
#predicting

for i in range(8):

    plt.subplot(2,4,i+1)

    plt.imshow(x_test[i+5].reshape(28,28),cmap='binary')

    prob = model.predict(x_test[i+5].reshape(1,28,28,1))

    pred = int(np.argmax(prob, axis=1))

    response = labels[pred] +" (" + str(round(prob[0][pred] * 100,2)) + " %)"

    plt.title(response)