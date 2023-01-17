# This is my first notebook for Deep Learning. Feedback will be appreciated.
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
np.random.seed(42)
train_data_full = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

labels = train_data_full.label

train_data_full.drop(['label'], axis = 1, inplace = True)
train_data_full.shape
train_data_full.head()
labels.head()
labels.value_counts(sort = False)
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder



sc = StandardScaler()

onehot = OneHotEncoder()
X_train, X_val, y_train, y_val = train_test_split(train_data_full, labels, test_size = 0.2, random_state = 42)
print(X_train.shape)

print(y_train.shape)

print(X_val.shape)

print(y_val.shape)
# Scaling training and validation sets

X_train = sc.fit_transform(X_train)

X_val = sc.transform(X_val)
# Encoding the output layer of training set

y_train = np.array(y_train).reshape(33600, 1)

y_train = onehot.fit_transform(y_train)



# Encoding the output layer of test set

y_val = np.array(y_val).reshape(8400, 1)

y_val = onehot.transform(y_val)
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Softmax

from keras.layers import Dropout

from keras.optimizers import sgd, RMSprop
optimizer = RMSprop(lr=0.0001)



model = Sequential()

model.add(Dense(64, input_dim = 784, activation = 'relu', kernel_initializer = 'glorot_normal', bias_initializer = 'zeros'))

model.add(Dropout(0.2))

model.add(Dense(64, activation = 'relu'))

model.add(Dropout(0.0))

model.add(Dense(64, activation = 'relu'))

model.add(Dropout(0.2))

model.add(Dense(64, activation = 'relu'))

model.add(Dropout(0.0))

model.add(Dense(64, activation = 'relu'))

model.add(Dropout(0.0))

model.add(Dense(10, activation = 'softmax'))
model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()
model.fit(X_train, y_train, epochs = 70, batch_size = 64, validation_data = (X_val, y_val))
# Taking all the data to train the model

X_train = train_data_full.copy()

y_train = labels.copy()
# Scaling and Encoding all data

sc = StandardScaler()

onehot = OneHotEncoder()



X_train = sc.fit_transform(X_train)



y_train = np.array(y_train).reshape(42000, 1)

y_train = onehot.fit_transform(y_train)
# Training the model again

optimizer = RMSprop(lr=0.0001)



model = Sequential()

model.add(Dense(64, input_dim = 784, activation = 'relu', kernel_initializer = 'glorot_normal', bias_initializer = 'zeros'))

model.add(Dropout(0.2))

model.add(Dense(64, activation = 'relu'))

model.add(Dropout(0.0))

model.add(Dense(64, activation = 'relu'))

model.add(Dropout(0.2))

model.add(Dense(64, activation = 'relu'))

model.add(Dropout(0.0))

model.add(Dense(64, activation = 'relu'))

model.add(Dropout(0.0))

model.add(Dense(10, activation = 'softmax'))



model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])



model.fit(X_train, y_train, epochs = 70, batch_size = 64)
# Loading the test set

test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
X_test = sc.transform(test_data)

X_test.shape
predictions = model.predict(X_test)

predictions.shape
predicted_labels = np.argmax(predictions, axis = 1)

results = pd.Series(predicted_labels, name = 'Label')

results_df = pd.DataFrame()

results_df['ImageId'] = [i for i in range(1, 28001)]

results_df['Label'] = results
results_df
results_df.to_csv('submission.csv', index = False)