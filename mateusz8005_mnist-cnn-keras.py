import pandas as pd

import matplotlib.pyplot as plt

import numpy as np



from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from keras.utils import to_categorical
train_data = pd.read_csv('../input/digit-recognizer/train.csv')

test_data = pd.read_csv('../input/digit-recognizer/test.csv')
sample_image = train_data.drop(columns='label').values[6].reshape(28,28)

plt.imshow(sample_image, cmap=plt.cm.gray.reversed())
from collections import Counter

x_test = test_data

y_train = train_data['label'].values

counts = Counter(y_train)

digits_count = [counts.get(i) for i in range(10) ]

plt.title('Digits count in training datasets')

plt.xlabel('digit')

plt.ylabel('count')

plt.bar(range(10), digits_count)
y_train = train_data['label'].values

x_train = train_data.drop(columns='label').values

x_test = x_test.values
x_train = x_train.astype(np.float32) / 255

x_test = x_test.astype(np.float32) / 255
y_train = to_categorical(y_train, 10)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
model = Sequential([

    Conv2D(32, input_shape=(28,28,1), kernel_size=(4,4), activation='relu'),

    MaxPooling2D(),

    Conv2D(64, kernel_size=(4,4), activation='relu'),

    MaxPooling2D(),

    Dropout(0.5),

    Flatten(),

    Dense(32, activation='relu'),

    Dropout(0.4),

    Dense(10, activation='softmax')

])



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=16, batch_size=32)
plt.figure(figsize=(20,5))

plt.title('loss during training')

plt.xlabel('epoch')

plt.ylabel('loss')

plt.plot(history.history['loss'])
