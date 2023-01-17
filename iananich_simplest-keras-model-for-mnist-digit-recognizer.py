import pandas as pd

train_df = pd.read_csv('../input/train.csv')
import numpy as np

data = train_df.values
np.random.shuffle(data)
labels = data[:, 0]
pixels = data[:, 1:]
from keras.utils import to_categorical

def prepare_pixels(arr):
    return np.array(arr / 255, dtype='float32')

pixels = prepare_pixels(pixels)
labels = to_categorical(labels)
TOTAL_TRAIN_SAMPLES = 42_000
VAL_SAMPLES = 7_000

val_pixels = pixels[:VAL_SAMPLES]
val_labels = labels[:VAL_SAMPLES]
train_pixels = pixels[VAL_SAMPLES:]
train_labels = labels[VAL_SAMPLES:]
from keras import models, layers

network = models.Sequential()
network.add(layers.Dense(256, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

history = network.fit(train_pixels, train_labels, 
                      epochs=30, 
                      batch_size=512, 
                      validation_data=(val_pixels, val_labels))
import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
acc = history.history['acc']
val_acc = history.history['val_acc']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(28 * 28,)))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
model.fit(pixels, labels, epochs=5, batch_size=512)
test_df = pd.read_csv('../input/test.csv')
test_pixels = np.array(test_df.values / 255, dtype='float32')

TEST_SAMPLES = 28_000
sample = pd.read_csv('../input/sample_submission.csv')
norm_predictions = model.predict(test_pixels)
predictions = sample.copy()
for i in range(TEST_SAMPLES):
    predictions.loc[i]['Label'] = norm_predictions[i].argmax()
predictions.to_csv('predictions.csv',index=False)