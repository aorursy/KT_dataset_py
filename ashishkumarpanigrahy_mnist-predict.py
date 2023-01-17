# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

sns.set(style='dark', context='notebook', palette='deep')

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
train_data.head()
import tensorflow as tf

y_train = train_data.label

X_train = train_data.drop('label', axis=1)
import matplotlib.pyplot as plt

test_data.head()
X_train = tf.keras.utils.normalize(X_train, axis=1)
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

#from keras.optimizers import RMSprop

#optimiser = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer='adam',

             loss='sparse_categorical_crossentropy',

             metrics=['accuracy'])
model.fit(X_train, y_train, epochs=30)
model.save('mnist.model')
new_model = tf.keras.models.load_model('mnist.model')
y_pred = new_model.predict(test_data)
print(np.argmax(y_pred[67]))

ax = np.array(test_data.loc[67,:])

plt.imshow(ax.reshape(28,28))

plt.show()
# select the indix with the maximum probability

y_pred = np.argmax(y_pred,axis = 1)
y_pred

#y_pred = pd.Series(y_pred,name="Label")
submissions=pd.DataFrame({'ImageId': list(range(1,len(y_pred)+1)),

                         "Label": y_pred})
submissions.head()
#submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),y_pred],axis = 1)



submissions.to_csv("mnist_pred.csv", index=False, header=1)