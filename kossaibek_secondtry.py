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
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.naive_bayes import GaussianNB

import seaborn as sns

import matplotlib.pyplot as plt
dataset = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')
x = dataset[dataset.columns[2:]]

y = dataset[dataset.columns[1:2]]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
sc_X = StandardScaler()

x_train = sc_X.fit_transform(x_train)

x_test = sc_X.fit_transform(x_test)
GNB = GaussianNB()
GNB.fit(x_train,y_train)
y_pred = GNB.predict(x_test)
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.metrics import roc_auc_score
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_pred)
from imblearn.over_sampling import SMOTE
sm = SMOTE()
x_res, y_res = sm.fit_resample(x,y)
x_res_train, x_res_test, y_res_train, y_res_test = train_test_split(x_res, y_res, test_size = 0.2, random_state = 0)
GNB.fit(x_res_train, y_res_train)
y_res_pred = GNB.predict(x_res_test)
print(classification_report(y_res_test, y_res_pred))
roc_auc_score(y_res_test, y_res_pred)
test = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')
test.head()
test_x = test[test.columns[1:]]
y_f = GNB.predict(test_x)
import pickle 
filename = 'GNB.sav'

pickle.dump(GNB, open(filename, 'wb'))
my_sub = pd.DataFrame({'ID_code': test.ID_code,'target': y_f})

my_sub.to_csv('subm.csv', index=False)