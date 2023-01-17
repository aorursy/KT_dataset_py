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
import pandas as pd

import numpy as np

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from keras.models import Model

from keras.layers import Input, Dense, Dropout, Activation

from keras.optimizers import SGD, Adam

from keras.utils import to_categorical

import matplotlib.pyplot as plt
data = pd.read_csv('/kaggle/input/train-traffic/Train.csv', index_col='id_code')



train_X = data.drop('target', axis=1)



train_y = data.target



test_X = pd.read_csv('/kaggle/input/train-traffic/Test.csv', index_col='id_code')



X = pd.concat([train_X, test_X])

def deal_with_time(X):

    X['current_time'] = pd.to_datetime(train_X['current_time'])

    X['current_time'] = X['current_time'].dt.hour + (X['current_time'].dt.minute)/60 + (X['current_time'].dt.second)/3666

    return X



X = deal_with_time(X)
def deal_with_date(X):

    X['current_date'] = pd.to_datetime(X['current_date'])

    X['current_month'] = X['current_date'].dt.month

    X['current_day'] = X['current_date'].dt.day

    X = X.drop('current_date', axis=1)

    return X

    

X = deal_with_date(X)
def deal_with_categorical_missing(X):

    continuous_columns = ['current_time', 'current_month', 'current_day', 'longitude_source', 'latitude_source', 'mean_halt_times_source', 'longitude_destination', 'latitude_destination', 'mean_halt_times_destination', 'current_year', 'current_week']

    X = pd.get_dummies(X)

    continuous_index = []

    for i in continuous_columns:

        continuous_index.append(X.columns.get_loc(i))

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')

    X = imp.fit_transform(X)

    #train_X.apply(lambda x: x.fillna(x.mean()),axis=0)

    #train_X = train_X.dropna()

    scaler = StandardScaler()

    X[:, continuous_index] = scaler.fit_transform(X[:, continuous_index])

    return X



X = deal_with_categorical_missing(X)
train_X = X[:1284,:]

test_X = X[1284:,:]
model = LogisticRegression(multi_class='auto', penalty='l2', C=.01, class_weight=None, solver='lbfgs', max_iter=5000, verbose=0)

model.fit(train_X, train_y)

print(model.score(train_X, train_y))
model = SGDClassifier(loss='log',class_weight=None, penalty='l2', alpha=.0001, eta0 = 0.0, learning_rate='optimal', max_iter=5000, tol=None, verbose=0)

model.fit(train_X, train_y)

print(model.score(train_X, train_y))
train_y = pd.get_dummies(train_y)

#val_y = pd.get_dummies(val_y)

inputs = Input((1388,))

X = Dropout(0.1)(inputs)

X = Dense(256)(X)

X = Activation('relu')(X) 

X = Dropout(0.1)(X)

X = Dense(256)(inputs)

X = Activation('relu')(X) 

X = Dropout(0.1)(X)

X = Dense(3)(X)

outputs = Activation('softmax')(X) 



model = Model(inputs=inputs, outputs=outputs)



model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=.001), metrics=['accuracy'])



model.fit(train_X, train_y, validation_split=0.2, epochs=1000, verbose=0)



print(model.evaluate(train_X, train_y))

#print(model.evaluate(val_X, val_y))

y_prob = model.predict(test_X) 

y_classes = y_prob.argmax(axis=-1)
plt.figure(0)

plt.plot(model.history.history['accuracy'])

plt.plot(model.history.history['val_accuracy'])
model.history.history.keys()