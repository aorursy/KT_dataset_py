# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sb

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/bank-train.csv')

test = pd.read_csv('../input/bank-test.csv')
train.describe()
sb.heatmap(train.corr())  
x_train = train[['pdays','previous','nr.employed']]

y_train = train.y

# create and fit model

LogReg = LogisticRegression()

LogReg.fit(x_train, y_train)
# metrics on training data (do NOT use this as a reliable estimate)

y_train_pred = LogReg.predict(x_train)

from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_train, y_train_pred)

confusion_matrix
print(classification_report(y_train, y_train_pred))
x_test = test[['pdays','previous','nr.employed']]

predictions = LogReg.predict(x_test)
submission = pd.concat([test.id, pd.Series(predictions)], axis = 1)

submission.columns = ['id', 'Predicted']

submission.to_csv('submission.csv', index=False)