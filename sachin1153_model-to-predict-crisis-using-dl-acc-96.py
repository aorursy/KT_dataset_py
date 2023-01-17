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
df = pd.read_csv('/kaggle/input/africa-economic-banking-and-systemic-crisis-data/african_crises.csv')
df.head()
df.isnull().sum()
df.drop(['cc3'],axis=1,inplace=True)

df.head()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



df.country = le.fit_transform(df.country)

df.banking_crisis = le.fit_transform(df.banking_crisis)
train_test_split = np.random.rand(len(df)) < 0.7

train = df[train_test_split]

test = df[~train_test_split]
train[:10]
len(train)
test[:10]
len(test)
train_data = train.iloc[:,0:12]

train_data.head()
train_label = train.iloc[:,12]

train_label.head()
test_data = test.iloc[:,0:12]

test_data.head()
test_label = test.iloc[:,12]

test_label.head()
from keras import models

from keras import layers



model = models.Sequential()

model.add(layers.Dense(8, activation = 'relu', input_shape = (12,)))

model.add(layers.Dense(8, activation = 'relu'))

model.add(layers.Dense(1, activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics =['accuracy'])
x_val = train_data[:100]

partial_x_train = train_data[100:]

y_val = train_label[:100]

partial_y_train = train_label[100:]
history = model.fit(partial_x_train, partial_y_train, epochs = 60, batch_size = 10, validation_data = (x_val, y_val))
model.fit(train_data,train_label, epochs = 60, batch_size = 10)

results = model.evaluate(test_data, test_label)
print(results)