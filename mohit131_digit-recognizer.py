import pandas as pds 

train_data=pds.read_csv("../input/train.csv")

train_data=train_data.values

test_data=pds.read_csv("../input/test.csv")

test_inputs=test_data.values

test_inputs.shape

train_inputs=train_data[:,1:]

train_labels=train_data[:,0]

from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
import matplotlib.pyplot as plt 

x=train_inputs[123,:]

x=x.reshape((28,28))

plt.imshow(x, cmap='gray')
train_inputs = train_inputs.astype('float32') / 255

test_inputs = test_inputs.astype('float32') / 255
from keras import models

from keras import layers

network = models.Sequential()

network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))

#network.add(layers.Dense(400, activation='relu'))

#network.add(layers.Dense(300, activation='relu'))

#network.add(layers.Dense(200, activation='relu'))

network.add(layers.Dense(100, activation='relu'))

network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop',

loss='categorical_crossentropy',

metrics=['accuracy'])
network.fit(train_inputs, train_labels, epochs=5, batch_size=128)
a=test_inputs

out=network.predict(a)
import random

i=random.randint(1,len(test_inputs))

import matplotlib.pyplot as plt 

x=test_inputs[i,:]

x=x.reshape((28,28))

plt.imshow(x, cmap='gray')

import numpy as np

out_list=np.argmax(out, axis=1)

print("predicted output=",out_list[i])
import csv

with open('submission.csv', mode='w') as file:

    writer = csv.writer(file, delimiter=',')



    #way to write to csv file

    writer.writerow(['ImageId','Label'])

    for i in range(0,len(out_list)):

    	writer.writerow([i+1,out_list[i]])