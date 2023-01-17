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
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten

import tensorflow.keras as ks

%matplotlib inline
def Extract(file,labels=False):

	print("Beginning the extraction process...")

	with open(file,'r') as f:

		data = f.read()

		data = data.split('\n')

	

	Data = []

	label = []

	

	dataLen = len(data)

	

	

	print("preparing data into {} samples..".format(dataLen))

	

	for i in range(1,dataLen):

		if data[i] != '':

			if labels == False:

				Data.append(np.array(data[i].split(','), dtype=np.float32))

			else:

				Data.append(np.array(data[i].split(',')[1:], dtype=np.float32)) 

				label.append(np.array(data[i].split(',')[0], dtype=np.int8))

			



	Data = np.array(Data)/255

	del data

	if labels == False: 

		return Data

	else:

		return (Data,label)
from tensorflow.keras.datasets.mnist import load_data #req internet



#training on original mnist dataset

(x_train,y_train),(x_test,y_test) = load_data()



data = Extract('/kaggle/input/digit-recognizer/test.csv')

(moreTrain,label) = Extract('/kaggle/input/digit-recognizer/train.csv',labels=True)
from tensorflow.keras.utils import to_categorical

y_test = to_categorical(y_test) #one hot encoding



x_train = x_train.astype(np.float32) / 255

x_test = x_test.astype(np.float32) / 255

x_test = x_test.reshape(len(x_test),28,28,1)

moreTrain /= 255

print(x_train.shape)

print(y_train.shape)

x_train = np.concatenate((x_train,moreTrain.reshape(len(moreTrain),28,28)),axis=0)

y_train = np.concatenate((y_train,label),axis=0)



x_train = x_train.reshape(len(x_train),28,28,1)

y_train = to_categorical(y_train)

print(x_train.shape)

print(y_train.shape)
#lets display some data

for i in range(8):

    plt.subplot(2,4,i+1)

    plt.imshow(x_test[i].reshape(28,28),cmap='binary')

    plt.title(np.argmax(y_test[i]))
#designing the model

model = Sequential()

model.add(Conv2D(filters=16, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(28, 28, 1), name='conv1'))

model.add(Conv2D(filters=16, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(28, 28, 1), name='conv2'))

model.add(MaxPool2D(pool_size=(2, 2),name='pool1'))

model.add(Conv2D(filters=32, kernel_size=5, strides=1,padding='same', activation='relu',name='conv3'))

model.add(Conv2D(filters=32, kernel_size=3, strides=1,padding='same', activation='relu',name='conv4'))

model.add(MaxPool2D(pool_size=(2, 2),name='pool2'))

model.add(Conv2D(filters=64, kernel_size=3, strides=1,padding='same', activation='relu',name='conv5'))

model.add(Conv2D(filters=64, kernel_size=3, strides=1,padding='valid', activation='relu',name='conv6'))

model.add(Flatten(name='flat'))

model.add(Dense(512,activation='relu',name='dense1'))

model.add(Dropout(0.2,name='dropout'))

model.add(Dense(10,activation='softmax',name='res'))



model.summary()
class Callback(ks.callbacks.Callback):

    def on_epoch_end(self,epoch,logs={}):

        if(logs['val_acc'] >= 0.999):

            print("\nValication accuracy reached more than 99.9%, training stopped")

            self.model.stop_training=True
earlyStop = Callback()

adam = ks.optimizers.Adam(lr=1e-3)

model.compile(loss='categorical_crossentropy',optimizer=adam, metrics=['accuracy'])

checkpoint = ks.callbacks.ModelCheckpoint('model.h5',save_best_only=True, monitor='val_acc', mode='max')
history = model.fit(x_train, y_train, validation_data=(x_test,y_test),batch_size=1024, epochs=100, shuffle=True, verbose=1, callbacks=[checkpoint,earlyStop])
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')



modelBest = ks.models.load_model('model.h5')
print(model.evaluate(x_test,y_test))

print(modelBest.evaluate(x_test,y_test)) #for me, modelBest gave better accuracy
with open('result.csv','w') as f:

    f.write('ImageId,Label\n')
i=0

for x in data:

    with open('result.csv','a') as f:

        pred = modelBest.predict(x.reshape(1,28,28,1))

        val = int(np.argmax(pred,axis=1))

        i += 1

        res = str(i) + ',' + str(val) + '\n'

        f.write(res)
#predict

for i in range(8):

    plt.subplot(2,4,i+1)

    plt.imshow(data[-i].reshape(28,28),cmap='binary')

    pred = model.predict(data[-i].reshape(1,28,28,1))

    val = int(np.argmax(pred,axis=1))

    plt.title(str(val))