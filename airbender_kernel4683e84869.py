import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
for dirname, _, filenames in os.walk('/kaggle/input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_data = np.genfromtxt('/kaggle/input/digit-recognizer/train.csv', delimiter=',')
print(train_data.shape)

label_data = train_data[1:, 0]
train_data = train_data[1:, 1:]

print(train_data.shape)
print(label_data.shape)

train_data = np.reshape(train_data, (42000, 28, 28, 1))
train_display = np.reshape(train_data, (42000, 28, 28))
print(train_data.shape)

plt.figure(figsize=(10,10))
for i in range(5):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_display[i], cmap=plt.cm.binary)
    plt.xlabel(label_data[i])
plt.show()
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.summary()
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

batch_size = 128

history = model.fit(train_data, label_data, steps_per_epoch=42000//batch_size, epochs=10)
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

test_data = np.genfromtxt('/kaggle/input/digit-recognizer/test.csv', delimiter=',')
print(test_data.shape)

test_data = test_data[1:]
print(test_data.shape)

test_data = np.reshape(test_data, (28000, 28, 28, 1))
result_data = probability_model.predict(test_data)
print(result_data.shape)

result = np.argmax(result_data, axis=1)
print(result.shape)

imageid = list(range(1, 28001))
submission = pd.DataFrame({'ImageId': imageid, 'Label': result})
submission.head()
submission.to_csv('submission.csv', index=False)
print('File saved')

for dirname, _, filenames in os.walk('/kaggle/working/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))