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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
dat = pd.read_csv('../input/train.csv')

dat.head(3)
dat.dtypes
dat.isnull().sum()
predictor_set = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']
dat1 = pd.get_dummies(dat[predictor_set], columns=["Pclass", "Sex", 'Embarked'], prefix=["Class", "Sex", "Emb"])
dat1.shape
dat1['Age'] = dat1['Age'].fillna(dat1['Age'].mean())
del dat1['Class_3']



del dat1['Sex_male']



y = dat['Survived']
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dat1, y, test_size = 0.3, random_state = 2019)



from sklearn.linear_model import LogisticRegression



lr = LogisticRegression()

lr.fit(X_train, y_train)



y_pred = lr.predict(X_test)



accuracy_score(y_pred=y_pred, y_true=y_test)
out_test = pd.read_csv('../input/test.csv')



PassengerId = out_test['PassengerId']



out_test = pd.get_dummies(out_test[predictor_set], columns=["Pclass", "Sex", 'Embarked'], prefix=["Class", "Sex", "Emb"])



out_test.head()
del out_test['Class_3']



del out_test['Sex_male']



out_test['Age'] = out_test['Age'].fillna(dat1['Age'].mean())
lr_full =  LogisticRegression()

lr_full.fit(dat1, y)



y_out = lr_full.predict(out_test)



y_out

submission = pd.DataFrame({

        "PassengerId": PassengerId,

        "Survived": y_out

    })

submission.to_csv('submission.csv', index=False)