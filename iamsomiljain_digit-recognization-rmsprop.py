import os
import pandas as pd

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
for i in range(15):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train.iloc[i,1:].to_numpy().reshape((28,28)), cmap=plt.cm.binary)
    plt.xlabel(train.iloc[i,0])
plt.show()
model = models.Sequential()
model.add(layers.Conv2D(28, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
model.summary()

model.compile(optimizer='RMSProp',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

train_data = train.iloc[:,1:].to_numpy().reshape((len(train), 28, 28, 1))
train_labels = train.iloc[:,0].to_numpy().reshape((len(train), 1))
history = model.fit(train_data, train_labels, epochs=8)
import numpy as np

test_data = test.to_numpy().reshape((len(test), 28, 28, 1))
output = model.predict(test_data)
output_labels = np.argmax(output, axis =1).reshape((len(test), 1))
plt.figure(figsize=(10,10))
for i in range(15):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_data[i].reshape(28,28), cmap=plt.cm.binary)
    plt.xlabel(output_labels[i])
plt.show()
df = pd.DataFrame(output_labels)
submission = pd.DataFrame({'ImageId': df.index+1, 'Label': df[0]})
submission.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved")

