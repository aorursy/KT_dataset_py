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
train_df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
train_df
labels = train_df['label'].to_numpy()

labels
X = train_df.iloc[:, 1:]

X = X.to_numpy()

X = X.reshape(X.shape[0], 28, 28, 1)
X.shape
X = X / 255.0

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0
x_train.shape

import tensorflow as tf
model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu', input_shape = (28, 28, 1)),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.4),



    tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.4),



    tf.keras.layers.Conv2D(128, kernel_size=4, activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(10, activation='softmax'),

])

model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
class MyCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):

        if logs.get('acc') >= 0.999:

            pass

            #self.model.stop_training = True

callback = MyCallback()
history = model.fit(x_train, y_train, epochs=30, batch_size=32, callbacks=[callback], shuffle=True)
model.evaluate(x_test, y_test)
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.show()
test_df = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

test_df.head()
submission = pd.DataFrame()
X_test = test_df.to_numpy()

X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_test = X_test / 255.0
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

y_pred.shape
submission = pd.DataFrame({

    'ImageId': range(1, len(y_pred)+1),

    'Label': y_pred

})
submission.to_csv('submission.csv', index=False)