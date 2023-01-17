import numpy as np

import pandas as pd

import cv2

import matplotlib.pyplot as plt

import tensorflow as tf

import os



from PIL import Image

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

from keras.models import load_model, Model

from keras.layers import Input, Conv2D, MaxPool2D, Dense, Flatten, Dropout
data = []

labels = [] 

classes = 43
for i in range(classes):

    path = os.path.join('../input/gtsrb-german-traffic-sign', 'Train', str(i))

    images = os.listdir(path)

    

    for a in images:

        try:

            image = Image.open(path + '/' + a)

            image = image.resize((50,50))

            image = np.array(image)

            

            data.append(image)

            labels.append(i)

        except:

            print("Error loading Image")

            

data = np.array(data)

labels = np.array(labels)



print(data.shape, labels.shape)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
y_train = to_categorical(y_train, 43)

y_test = to_categorical(y_test, 43)
input_signal = Input(shape=(x_train.shape[1:]))



conv1 = Conv2D(filters=32, kernel_size=(3,3), activation='relu')(input_signal)

conv2 = Conv2D(filters=32, kernel_size=(3,3), activation='relu')(conv1)

pool1 = MaxPool2D(pool_size=(2,2))(conv2)

drop1 = Dropout(0.25)(pool1)



conv3 = Conv2D(filters=64, kernel_size=(3,3), activation='relu')(drop1)

conv4 = Conv2D(filters=64, kernel_size=(3,3), activation='relu')(conv3)

pool2 = MaxPool2D(pool_size=(2,2))(conv4)

drop2 = Dropout(0.25)(pool2)



conv5 = Conv2D(filters=128, kernel_size=(3,3), activation='relu')(drop2)

conv6 = Conv2D(filters=128, kernel_size=(3,3), activation='relu')(conv5)

pool3 = MaxPool2D(pool_size=(2,2))(conv6)

drop3 = Dropout(0.25)(pool3)



flat = Flatten()(drop3)

hidden1 = Dense(256, activation='relu')(flat)

drop = Dropout(0.25)(hidden1)

out = Dense(43, activation='softmax')(drop)



model = Model(input_signal, out)

print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=96, epochs=25, validation_data=(x_test, y_test), verbose=1)
model.save('traffic_signal.h5')
plt.figure(0)

plt.plot(history.history['accuracy'], label='training_acc')

plt.plot(history.history['val_accuracy'], label='validation_acc')

plt.title("Accuracy")

plt.xlabel("epochs")

plt.ylabel("accuracy")

plt.legend()

plt.show()
plt.figure(1)

plt.plot(history.history['loss'], label='training_loss')

plt.plot(history.history['val_loss'], label='validation_loss')

plt.title("Loss")

plt.xlabel('epochs')

plt.ylabel('loss')

plt.legend()

plt.show()
from sklearn.metrics import accuracy_score
y = pd.read_csv('../input/gtsrb-german-traffic-sign/Test.csv')



labels = y['ClassId'].values

imgs = y['Path'].values
test_data = []

for img in imgs:

    image = Image.open('../input/gtsrb-german-traffic-sign/' + img)

    image = image.resize((50,50))

    test_data.append(np.array(image))

    

x = np.array(test_data)
pred = model.predict(x)

pred = pred.argmax(axis=-1)
print(accuracy_score(labels, pred))