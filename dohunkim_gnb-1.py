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
train_path = '../input/digit-recognizer/train.csv'

test_path = '../input/digit-recognizer/test.csv'
train_data = pd.read_csv(train_path)

test_data = pd.read_csv(test_path)
train_data
test_data
y_train = train_data['label']

x_train = train_data.drop(labels=['label'], axis=1)
import tensorflow as tf
y_train.shape
y_train_onehot = tf.one_hot(y_train, 10)
y_train_onehot.shape
x_train = np.asarray(x_train)
import matplotlib.pyplot as plt
image = x_train[1345].reshape((28, 28))
plt.imshow(image, cmap='gray_r')

plt.show()
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense
x_train[0].shape
model = Sequential([

    Dense(50, input_shape=(784,), activation='relu'),

    Dense(50, activation='relu'),

    Dense(10, activation='softmax')

])
model.summary()
tf.keras.utils.plot_model(model, 'first_model.png', show_shapes=True)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x=x_train,

                    y=y_train_onehot, 

                    validation_split=0.1, 

                    epochs=15,

                    batch_size=32)
train_loss = history.history['loss']

valid_loss = history.history['val_loss']



train_acc = history.history['accuracy']

valid_acc = history.history['val_accuracy']
plt.plot(train_loss, label='train loss')

plt.plot(valid_loss, label='validation loss')



plt.legend()



plt.title('train loss vs validation loss')



plt.show()
plt.plot(train_acc, label='train accuracy')

plt.plot(valid_acc, label='validation accuracy')



plt.legend()



plt.title('train accuracy vs validation accuracy')



plt.show()
x_test = np.asarray(test_data)
x_test.shape
prediction = model.predict(x_test)
prediction[:5]
predict = np.argmax(prediction, axis=1)
submission = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
submission
submission['Label'] = predict
submission
submission.to_csv('simple_mnist.csv', index=False)