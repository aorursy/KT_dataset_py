import warnings

warnings.filterwarnings('ignore')
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from tensorflow.keras import models, layers
train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')
train.head()
y_train = train['label']

X_train = train.drop(['label'], axis=1)
X_train /= 255.0

test /= 255.0
X_train = X_train.values.reshape(-1, 28, 28, 1)

test = test.values.reshape(-1, 28, 28, 1)
def plot(data, labels, title='Label'):

    plt.figure(figsize=(10, 9))

    for i in range(12):

        plt.subplot(3, 4, i+1)

        plt.imshow(data[i][:,:,0])

        plt.title('{}: {}'.format(title, labels[i]))

        plt.axis('off');
plot(X_train, y_train)
model = models.Sequential([

    layers.Conv2D(32, (3, 3), activation='relu', padding = 'Same', input_shape=(28, 28, 1)),

    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation='relu', padding = 'Same'),

    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), activation='relu'),

    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(512, activation='relu'),

    layers.Dense(10, activation='softmax')

])
model.compile(optimizer='adam',

             loss='sparse_categorical_crossentropy',

             metrics=['acc'])
model.summary()
EPOCHS = 5
%%time

history = model.fit(X_train, y_train.values,

                   validation_split=.1,

                   epochs=EPOCHS, batch_size=64,

                   verbose=2)
acc = history.history['acc']

val_acc = history.history['val_acc']



loss = history.history['loss']

val_loss = history.history['val_loss']
epochs = range(EPOCHS)
plt.figure(figsize=(15, 6))



plt.subplot(1, 2, 1)

plt.title('Training and Validation Loss')

plt.plot(epochs, loss, label='Training')

plt.plot(epochs, val_loss, label='Validation')

plt.legend()

plt.xlabel('Epochs')

plt.ylabel('Loss')



plt.subplot(1, 2, 2)

plt.title('Training and Validation Accuracy')

plt.plot(epochs, acc, label='Training')

plt.plot(epochs, val_acc, label='Validation')

plt.legend()

plt.xlabel('Epochs')

plt.ylabel('Accuracy')



plt.show()
%%time

results = model.predict(test)
results = np.argmax(results, axis=1)
plot(test, results, 'Predicted label')
results = pd.Series(results, name='Label')
submission = pd.concat([pd.Series(range(1, 28001), name='ImageId'), results], axis=1)
submission.to_csv('digit_recognizer.csv', index=False)