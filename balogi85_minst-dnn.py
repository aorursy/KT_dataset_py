# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf
import matplotlib.pyplot as plt
%matplotlib inline
class MyCallback (tf.keras.callbacks.Callback):
    def on_epoch_end(self, epochs, log={}):
        if (log.get('accuracy')>=0.9999):
            model.stop_training=True
train_im_csv = '/kaggle/input/digit-recognizer/train.csv'
test_im_csv= '/kaggle/input/digit-recognizer/test.csv'
data = pd.read_csv(train_im_csv)
data.head()
train_label = data['label']
data.drop('label',axis=1,inplace=True)
train_im = data.values
test_im = pd.read_csv(test_im_csv).values
print(train_im.shape, test_im.shape)
callback = MyCallback()
train_im, test_im = train_im.reshape(42000,28,28,1), test_im.reshape(28000,28,28,1)
train_im, test_im = train_im/255, test_im/255
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(200,activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_im, train_label, epochs=20, callbacks=[callback])
model.summary()
acc = history.history['accuracy']
loss = history.history['loss']
epochs=range(len(loss))
plt.plot(epochs, acc, 'r')
plt.plot(epochs, loss, 'b')
plt.title('Accuracy and Loss')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Acc", "Loss"])

plt.figure()
prediction = model.predict(test_im)
prediction = np.argmax(prediction,axis = 1)
prediction = pd.Series(prediction,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),prediction],axis = 1)

submission.to_csv("cnn_mnist_datagen.csv",index=False)
