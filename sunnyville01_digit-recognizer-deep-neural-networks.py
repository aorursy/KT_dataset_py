# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#  Load data into Dataframes

train_data = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

submission_data = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
# Check data with head()

# train_data.head()

submission_data.head()
# Check for any NaN values

train_data.isnull().values.sum()

# submission_data.isnull().values.sum()
X = train_data.copy().drop(['label'], 1)

y = train_data['label']



X = X/255 # Normalize

X = X.values.reshape(-1, 28, 28)



X.shape
# Check distribution of labels

import seaborn as sns



g = sns.countplot(y) 
# Plot a random image

import matplotlib.pyplot as plt



plt.imshow(X[1], cmap=plt.cm.binary)
from sklearn.model_selection import train_test_split

import numpy as np



X = np.array(X)

y = np.array(y)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=2)

X_test.shape

import tensorflow as tf



model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))



model.compile(optimizer='adam',

             loss='sparse_categorical_crossentropy',

             metrics=['accuracy'])



model.fit(X_train, y_train, epochs=3)
val_loss, val_acc = model.evaluate(X_test, y_test)
plt.imshow(X_test[44])

plt.show()



predictions = model.predict(X_test)

# print(f'Prediction: {np.argmax(predictions[44])}')



predictions = np.argmax(predictions,axis = 1)

predictions[44]
# Train with full data

model.fit(X, y, epochs=3)
submission_data = submission_data/255 # Normalize

submission_data = submission_data.values.reshape(-1, 28, 28)



submission_data = np.array(submission_data)

submission_predictions = model.predict(submission_data)

submission_predictions = np.argmax(submission_predictions,axis = 1)



len(submission_predictions)
results = pd.Series(submission_predictions,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("submission.csv",index=False)