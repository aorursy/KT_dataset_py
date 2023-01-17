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
import pandas as pd
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

train.head()
x_train = train.drop('label', axis=1).to_numpy()

y_train = train['label'].to_numpy()

print(x_train.shape, y_train.shape)
x_train = x_train.reshape(-1, 28, 28, 1)
import tensorflow as tf
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(28,28,1)),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(10, activation='softmax')

])
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=20, validation_split=0.2, shuffle=True)
# All training data

history = model.fit(x_train, y_train, epochs=20)
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv').to_numpy()

print(test.shape)
test = test.reshape(-1, 28, 28, 1)
preds = model.predict(test)
df = pd.DataFrame(np.argmax(preds, axis=1), columns=['Label'])

df.insert(0, 'ImageId', df.index + 1)

df.to_csv('submission.csv', index=False)