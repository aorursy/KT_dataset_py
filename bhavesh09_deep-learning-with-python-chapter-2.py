import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print(train.shape, test.shape)
train_labels = train.iloc[:,0]
train_images = train.iloc[:,1:]
#test_labels = test.iloc[:,0]
test_images = test
from keras import models, layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28, )))
#network.add(layers.Dense(64, activation='relu'))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop',
               loss='categorical_crossentropy',
               metrics=['accuracy'])
train_images = train_images.astype('float32')/255
test_images = test_images.astype('float32')/255
from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
#test_labels = to_categorical(test_labels)
network.fit(train_images, train_labels, epochs=5, batch_size=128) #0.9871
test_prediction = network.predict_classes(test_images)
submission = pd.DataFrame(test_prediction).reset_index()
submission.columns=['ImageId','Label']
submission['ImageId'] += 1
submission.to_csv("submission.csv",index=False)
