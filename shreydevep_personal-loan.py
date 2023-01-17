# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
original = pd.read_excel('../input/bank-loan-modelling/Bank_Personal_Loan_Modelling.xlsx',"Data")
original.head()
original.tail()
original.info()
original.describe()
df = original.copy()
df.drop(['ID'],axis = 1, inplace = True)

X = df.drop(['Personal Loan'], axis = 1)

y = df['Personal Loan']
X.info()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

from sklearn.linear_model import LogisticRegressionCV

clf = LogisticRegressionCV(cv = 5,max_iter = 1000 , random_state=0).fit(X_train, y_train)

y_pred = clf.predict(X_test)
from sklearn.metrics import accuracy_score

score = accuracy_score(y_test, y_pred, normalize=True)

score
from sklearn.metrics import confusion_matrix

matrix = confusion_matrix(y_test,y_pred)
matrix