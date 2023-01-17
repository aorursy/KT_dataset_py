import matplotlib.pyplot as plt

import pandas as pd



import os

print(os.listdir('../input'))
data_folder = "../input"



test = pd.read_csv(data_folder + "/test.csv")

train = pd.read_csv(data_folder + "/train.csv")
x_train = (train.iloc[:,1:].values).astype('float32')

x_test = test.values.astype('float32')



x_train /= 255.0

x_test /= 255.0



y_train = train.iloc[:,0].values.astype('int32')
print('X_TRAIN SHAPE:      ', x_train.shape)

print('X_TEST SHAPE:       ', x_test.shape)

print('SAMPLES IN X_TRAIN: ', x_train.shape[0])

print('SAMPLES IN X_TEST:  ', x_test.shape[0])
x_train = x_train.reshape(42000, 28, 28,1) # 28 x 28 = 784

x_test = x_test.reshape(28000, 28, 28,1)



print(x_train.shape)

print(x_test.shape)
plt.figure(figsize=(12,10))



for img in range(10):  

    plt.subplot(5, 5, img+1)

    plt.imshow(x_train[img].reshape((28, 28)), cmap='binary_r')

    plt.axis('off')

    

    plt.title('Label: ' + y_train[img].astype('str'))

    

plt.show()
import keras 

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from keras.preprocessing.image import ImageDataGenerator

from keras.utils import to_categorical



from sklearn.model_selection import train_test_split
y_train = to_categorical(y_train, 10)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.1, random_state=1)
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))

model.add(Conv2D(64, (3, 3), activation='relu'))



model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(Conv2D(128, (3, 3), activation='relu'))



model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='Adadelta', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=1,

          validation_data=(x_test, y_test))
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)



print('Test Loss:', loss)

print('Test Accuracy:', accuracy%100)