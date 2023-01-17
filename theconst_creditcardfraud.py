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
dataset = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')



dataset
class_0 = dataset.loc[dataset['Class']==0]

class_1 = dataset.loc[dataset['Class']==1]

print('class 0: {} ({}%) \nclass 1: {} ({}%)'.format(len(class_0), round(len(class_0)/len(dataset)*100, 2), len(class_1), round(len(class_1)/len(dataset)*100, 2)))
dataset.drop(['Time'], axis=1, inplace=True)
from sklearn.preprocessing import StandardScaler





for col in dataset.columns:

    if col == 'Class':

        continue

    print(col)

    print(dataset[col].values.std())

    dataset[col] = StandardScaler().fit_transform(dataset[col].values.reshape(-1, 1))

    print(dataset[col].values.std(), '\n')



x = dataset.loc[:, dataset.columns != 'Class']

y = dataset.loc[:, dataset.columns == 'Class']



x
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, stratify=y)
from keras.models import Model

from keras.layers import Dense, Input



input_ = Input((29,))

hidden = Dense(50, activation='sigmoid')(input_)

output = Dense(1, activation='sigmoid')(hidden)



perc = Model(inputs=input_, outputs=output)

perc.compile(loss='mean_squared_error',

              optimizer='sgd',

              metrics=['mae', 'acc'])
perc.fit(X_train, y_train, epochs=3, validation_split=0.2, shuffle=True)
eval_scores = perc.evaluate(X_test, y_test)



for metric, score in zip(perc.metrics_names, eval_scores):

    print(metric, score)
from sklearn.metrics import precision_recall_fscore_support



results = perc.predict(X_test)

print('precision, recall, f score:', precision_recall_fscore_support(y_test,results.round(), average='binary')[:-1])
num_0 = len(dataset[dataset['Class']==0])

num_1 = len(dataset[dataset['Class']==1])

print(num_0,num_1)



# undersampling

undersampled_data = pd.concat([dataset[dataset['Class']==0].sample(num_1), dataset[dataset['Class']==1]])

print('amount undersampled data', len(undersampled_data))



# oversampling

oversampled_data = pd.concat([dataset[dataset['Class']==0], dataset[dataset['Class']==1].sample(num_0, replace=True)])

print('amount oversampled data',len(oversampled_data))
x_under = undersampled_data.loc[:, undersampled_data.columns != 'Class']

y_under = undersampled_data.loc[:, undersampled_data.columns == 'Class']



x_over = oversampled_data.loc[:, oversampled_data.columns != 'Class']

y_over = oversampled_data.loc[:, oversampled_data.columns == 'Class']
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(x_under,y_under,test_size = 0.3, stratify=y_under)
from keras.models import Model

from keras.layers import Dense, Input



input_ = Input((29,))

hidden = Dense(50, activation='sigmoid')(input_)

output = Dense(1, activation='sigmoid')(hidden)



perc = Model(inputs=input_, outputs=output)

perc.compile(loss='mean_squared_error',

              optimizer='sgd',

              metrics=['mae', 'acc'])



perc.fit(X_train, y_train, epochs=100, validation_split=0.2, shuffle=True)
eval_scores = perc.evaluate(X_test, y_test)



for metric, score in zip(perc.metrics_names, eval_scores):

    print(metric, score)
from sklearn.metrics import precision_recall_fscore_support



results = perc.predict(X_test)

print('precision, recall, f score:', precision_recall_fscore_support(y_test,results.round(), average='binary')[:-1])
results = perc.predict(x)

print('precision, recall, f score:', precision_recall_fscore_support(y,results.round(), average='binary')[:-1])
X_train, X_test, y_train, y_test = train_test_split(x_over,y_over,test_size = 0.3, stratify=y_over)



input_ = Input((29,))

hidden = Dense(50, activation='sigmoid')(input_)

output = Dense(1, activation='sigmoid')(hidden)



perc = Model(inputs=input_, outputs=output)

perc.compile(loss='mean_squared_error',

              optimizer='sgd',

              metrics=['mae', 'acc'])



perc.fit(X_train, y_train, epochs=5, validation_split=0.2, shuffle=True)
eval_scores = perc.evaluate(X_test, y_test)

for metric, score in zip(perc.metrics_names, eval_scores):

    print(metric, score)



results = perc.predict(X_test)

print('precision, recall, f score:', precision_recall_fscore_support(y_test,results.round(), average='binary')[:-1])
results = perc.predict(x)

print('precision, recall, f score:', precision_recall_fscore_support(y,results.round(), average='binary')[:-1])