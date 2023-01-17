# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np



from keras.utils import to_categorical

import matplotlib.pyplot as plt



import keras

from keras.models import Sequential, load_model

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.layers.normalization import BatchNormalization

from keras.layers.advanced_activations import LeakyReLU

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/digit-recognizer/train.csv")

X_train = np.array(df.iloc[:,1:])

y_train = np.array(df.iloc[:,0])



X_train = np.reshape(X_train,(-1,28,28,1))
def create_dev_set(X_train, Y_train):

    ## split 42000 into 33600 and 8400 (0.2)

    return train_test_split(X_train, Y_train, test_size = 0.2, random_state = 0)
X_train, X_dev, y_train, y_dev = create_dev_set(X_train, y_train)

print('Training data shape : ', X_train.shape, y_train.shape)

print('Dev data shape : ', X_dev.shape, y_dev.shape)
classes = np.unique(y_train)

nClasses = len(classes)

print('Total number of outputs : ', nClasses)

print('Output classes : ', classes)
X_train = X_train.astype('float32')

X_dev = X_dev.astype('float32')

X_train = X_train / 255.

X_dev = X_dev / 255.



y_train_one_hot = np.array(to_categorical(y_train))

y_dev_one_hot = np.array(to_categorical(y_dev))
batch_size = 64

epochs = 30

num_classes = 10



dr = Sequential()

dr.add(Conv2D(32, kernel_size=(3,3),activation='linear',input_shape=(28,28,1),padding='same'))

dr.add(BatchNormalization(axis=-1))

dr.add(LeakyReLU(alpha=0.1))

dr.add(MaxPooling2D((2,2),padding='same'))

dr.add(Dropout(0.3))

dr.add(Conv2D(64, (3,3), activation='linear',padding='same'))

dr.add(BatchNormalization(axis=-1))

dr.add(LeakyReLU(alpha=0.1))

dr.add(MaxPooling2D(pool_size=(2,2),padding='same'))

dr.add(Dropout(0.3))

dr.add(Conv2D(128, (3,3), activation='linear',padding='same'))

dr.add(BatchNormalization(axis=-1))

dr.add(LeakyReLU(alpha=0.1))                  

dr.add(MaxPooling2D(pool_size=(2,2),padding='same'))

dr.add(Dropout(0.4))

dr.add(Flatten())

dr.add(Dense(120, activation='linear'))

dr.add(BatchNormalization(axis=-1))

dr.add(LeakyReLU(alpha=0.1))         

dr.add(Dropout(0.3))         

dr.add(Dense(40, activation='linear'))

dr.add(BatchNormalization(axis=-1))

dr.add(LeakyReLU(alpha=0.1))         

dr.add(Dropout(0.2)) 

dr.add(Dense(num_classes, activation='softmax'))



dr.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])



dr.summary()



training = dr.fit(X_train, y_train_one_hot, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_dev, y_dev_one_hot))



dr.save("Conv2D_DR_dropout.h5py")



test_eval = dr.evaluate(X_dev, y_dev_one_hot, verbose=0)

print(test_eval)



accuracy = training.history['accuracy']

val_accuracy = training.history['val_accuracy']

loss = training.history['loss']

val_loss = training.history['val_loss']

epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'bo', label='Training accuracy')

plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()



from IPython.display import FileLink

FileLink(r"Conv2D_DR_dropout.h5py")
print("Training dataset evaluation")

test_eval = dr.evaluate(X_train, y_train_one_hot, verbose=0)

print(test_eval)



print("Dev dataset evaluation")

test_eval = dr.evaluate(X_dev, y_dev_one_hot, verbose=0)

print(test_eval)
dr = load_model(r"Conv2D_DR_dropout.h5py")

df_test = pd.read_csv("../input/digit-recognizer/test.csv")

X_test = np.array(df_test.iloc[:,:])

X_test = np.reshape(X_test,(X_test.shape[0],28,28,1))



X_test = X_test.astype('float32')

X_test = X_test / 255.



res = dr.predict(X_test)

res = pd.DataFrame(np.argmax(np.round(res),axis=1))

res.columns = ["Label"]

image_id = pd.DataFrame(np.arange(1,28001,dtype=int))

image_id.columns=["ImageId"]

result = pd.concat([image_id, res], axis=1)

result.set_index("ImageId", inplace=True)

result.to_csv(r"Conv2D_Epochs20_Dropout.csv")
from IPython.display import FileLink

FileLink(r"Conv2D_Epochs20_Dropout.csv")