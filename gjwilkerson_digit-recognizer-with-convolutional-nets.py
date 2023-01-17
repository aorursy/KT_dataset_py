from keras import layers
from keras import models

import numpy as np
import pandas as pd

from keras.utils import to_categorical

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

%matplotlib notebook
model = models.Sequential()

# convolutional and maxpooling layers for image processing
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# dense layers for classification
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()
df = pd.read_csv('./train.csv.gz', compression='gzip', header = 0, index_col=None)
df
# the classes
train_labels = df.pop('label').values

# the images
train_images = df.values
# images are flat, make them square again
train_images = train_images.reshape((42000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
train_images[0, :, :, 0].shape
# slice to get one image
# indices are:  image number, height, width, and channel

image0 = train_images[0, :, :, 0]

plt.imshow(image0,aspect="auto", cmap=plt.cm.binary)
plt.show();
# show the corresponding label

train_labels[0]
train_labels = to_categorical(train_labels)

model.compile(optimizer='rmsprop', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5, batch_size=64)
df_test = pd.read_csv('./test.csv.gz', compression = "gzip", index_col=None)

df_test.head()
# the images
test_images = df_test.values
test_images.shape
# test images are flat, make them square again
test_images = test_images.reshape((28000, 28, 28, 1))
test_images = test_images.astype('float32') / 255
# generate the predictions using the trained model

y_pred = model.predict_classes(test_images)
y_pred
# save predictions to file

df_pred = pd.DataFrame({"ImageId": list(range(1, len(y_pred) + 1)), "Label": y_pred})

df_pred.to_csv('predictions.csv', index=False)

import sklearn

