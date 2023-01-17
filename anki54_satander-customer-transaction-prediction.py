# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn import metrics





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
x_train = pd.read_csv('../input/train.csv')



y_train = x_train['target']

id_code=x_train['ID_code']

x_train = x_train[x_train.columns[~x_train.columns.isin(['target','ID_code'])]]

x_train.head()
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.3) # 70% training and 30% test
from sklearn.neural_network import MLPClassifier

## Next let's try using a neural network.



nn_model = MLPClassifier(hidden_layer_sizes=(100,), activation='tanh', solver='adam', alpha=0.01,learning_rate_init=0.001, max_iter=800)

nn_model.fit(x_train,y_train)

Y_pred_train=nn_model.predict(x_train)

accuracy = metrics.accuracy_score(y_train, Y_pred_train)

print('Accuracy: {:.2f}'.format(accuracy))



Y_pred_valid=nn_model.predict(x_valid)

accuracy = metrics.accuracy_score(y_valid, Y_pred_valid)

print('Accuracy validation: {:.2f}'.format(accuracy))
accuracy = metrics.accuracy_score(y_train, Y_pred_train)

print('Accuracy: {:.2f}'.format(accuracy))



Y_pred_valid=nn_model.predict(x_valid)

accuracy = metrics.accuracy_score(y_valid, Y_pred_valid)

print('Accuracy validation: {:.2f}'.format(accuracy))
x_test=pd.read_csv('../input/test.csv')

id_code=x_test['ID_code']

x_test = x_test[x_test.columns[~x_test.columns.isin(['target','ID_code'])]]

Y_pred_test=nn_model.predict(x_test)

result = pd.DataFrame()

result['ID_code'] = id_code

result['target'] = Y_pred_test

result.to_csv('submission.csv',sep=',',index=False)
