import numpy as np

import pandas as pd

import keras as k

import matplotlib.pyplot as plt
k.backend.set_image_dim_ordering('th')
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
features = train.iloc[:, 1:]

labels = train.iloc[:, 0]



features = features.as_matrix().reshape(features.shape[0], 1, 28, 28)

labels = labels.as_matrix()



features = features / 255

labels = k.utils.to_categorical(labels)



test = test.as_matrix().reshape(test.shape[0], 1, 28, 28)

test = test / 255
model = k.models.Sequential()

model.add(k.layers.convolutional.Conv2D(32, (3, 3), input_shape=(1, 28, 28), activation='relu'))

model.add(k.layers.BatchNormalization())

model.add(k.layers.convolutional.MaxPooling2D(pool_size = (2, 2)))



model.add(k.layers.convolutional.Conv2D(64, (3, 3), activation = 'relu'))

model.add(k.layers.BatchNormalization())

model.add(k.layers.convolutional.MaxPooling2D(pool_size = (2, 2)))



model.add(k.layers.Dropout(0.25))

model.add(k.layers.Flatten())

model.add(k.layers.Dense(512, activation='relu'))

model.add(k.layers.Dropout(0.2))

model.add(k.layers.Dense(256, activation='relu'))

model.add(k.layers.Dropout(0.25))



model.add(k.layers.Dense(10, activation = 'softmax'))

model.compile(optimizer = k.optimizers.Nadam(lr=0.001), loss=k.losses.categorical_crossentropy, metrics=['accuracy'])

model.summary()
train = model.fit(features, labels, epochs=10, batch_size=420, verbose=1)
loss = train.history['loss']

plt.plot(loss)

plt.show()
predict = model.predict(test, batch_size=280, verbose=1)

prediction = np.argmax(predict, axis=1)



result = pd.DataFrame({'Label': prediction})

result.index += 1

result.index.name = "ImageId"

result.to_csv('cnn_on_keras_mnist_result.csv')