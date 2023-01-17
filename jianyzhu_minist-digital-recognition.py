import numpy as np

import pandas as pd



train = pd.read_csv("../input/digit-recognizer/train.csv")

train.head()

test = pd.read_csv('../input/digit-recognizer/test.csv')

test.head()
test.shape


x, y = train.iloc[:,1:785], train.iloc[:,0]

print(x.shape)

print(y.shape)
x_standard = []

for i in range(0,x.shape[0]):

    x_standard.append(x.iloc[i,:].values.reshape(28,28)/255)
x_standard_array = np.array(x_standard)

x_standard_array.shape
%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt





plt.imshow(x_standard_array[8])

plt.show()

print(y[8])
from keras import layers

from keras import models

from keras.utils import to_categorical
model = models.Sequential()

model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(28,28,1)))

model.add(layers.MaxPooling2D((2,2)))

model.add((layers.Conv2D(64,(3,3), activation = 'relu')))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64,(3,3), activation = 'relu'))

model.add(layers.Flatten())

model.add(layers.Dense(64,activation = 'relu'))

model.add(layers.Dense(10,activation = 'softmax'))
model.compile(optimizer = 'rmsprop',

             loss = 'categorical_crossentropy',

             metrics=['accuracy'])
train_images = x_standard_array.reshape((42000,28,28,1))

y_labels = to_categorical(y)
model.fit(train_images, y_labels, epochs = 5)
test_images = test.values.reshape(28000,28,28)/255

test_images = np.array(test_images)

test_images.reshape((28000,28,28,1))

test_images.shape
plt.imshow(test_images[1])

plt.show()
test_images[0].shape
pred=[]

for x in test_images:

    pred.append(model.predict(x.reshape(1,28,28,1)))

eval_results = []

for i in range(0,len(pred)):

    eval_results.append(int(np.argmax(pred[i],axis=1)))

eval_results[0:10]
results = pd.Series(eval_results, name = 'Label')



submission = pd.concat([pd.Series(range(1,28001), name = 'ImageId'), results], axis = 1)
submission.to_csv('Deep_Learning_Results.csv',index = False)