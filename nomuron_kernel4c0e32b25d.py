# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/titanic/train.csv')

test_data = pd.read_csv('../input/titanic/test.csv')

train_data = pd.DataFrame(train_data).drop(columns=['Name', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis=1)

test_data = pd.DataFrame(test_data).drop(columns=['Name', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis=1)
train_data.head()
train_data.info()
train_data_sex = pd.get_dummies(train_data['Sex'])

test_data_sex = pd.get_dummies(test_data['Sex'])

train_data_sex.head()
train_data = pd.concat([train_data, train_data_sex], axis=1)

test_data = pd.concat([test_data, test_data_sex], axis=1)

train_data.head()
train_data = train_data.drop(columns=['Sex'], axis=1)

test_data = test_data.drop(columns=['Sex'], axis=1)

train_data.head(10)
from sklearn.impute import SimpleImputer

sim_imp = SimpleImputer(strategy='mean')

imputed_train_data = pd.DataFrame(sim_imp.fit_transform(train_data))

imputed_test_data = pd.DataFrame(sim_imp.fit_transform(test_data))

imputed_train_data.columns = train_data.columns

imputed_test_data.columns = test_data.columns
y = pd.DataFrame(imputed_train_data['Survived'])

x = imputed_train_data.drop(columns=['Survived'], axis=1)

x.head()
y.head()
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=0)
#from sklearn.ensemble import RandomForestRegressor

#ForReg = RandomForestRegressor(n_estimators=1000, random_state=0)

#ForReg.fit(x_train, y_train)

from sklearn.linear_model import LogisticRegression

LogReg = LogisticRegression()

LogReg.fit(x_train, y_train)

#from xgboost import XGBRegressor

#XG = XGBRegressor(n_estimators = 100)

#XG.fit(x_train, y_train)
preds = LogReg.predict(imputed_test_data)

acc = round(LogReg.score(x_train, y_train) * 100, 2)

print(acc)

preds
submission = pd.DataFrame({

    "PassengerId": test_data["PassengerId"],

    "Survived": preds

})

submission.to_csv('submission.csv', index=False)

submission