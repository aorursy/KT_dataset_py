import numpy as np
import pandas as pd

from keras import layers
from keras import models
from keras.utils import to_categorical
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print(train.shape)
print(test.shape)
train.head()
test.head()
train_images = train.drop('label', axis = 1).values
train_images = train_images.reshape((-1,28,28,1))
train_images = train_images.astype('float32') / 255

train_labels = to_categorical(train.loc[:, 'label'])
test_images = test.values
test_images = test_images.reshape((-1,28,28,1))
test_images = test_images.astype('float32') / 255
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))

model.summary()
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)
test_labels = model.predict(test_images)
pred = test_labels.argmax(axis=1)
submission = pd.DataFrame({
        "ImageId": range(1, 28001) ,
        "Label": pred
    })
submission.to_csv('submission.csv', index=False)