# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
d_train = pd.read_csv('../input/train.csv')

d_test = pd.read_csv('../input/test.csv')
d_train.info()
d_train.head()
d_test.info()
d_test.head()
d_train.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

d_test.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
# Change male to 1 and female to 0:

d_train.Sex = [1 if each =='male' else 0 for each in d_train.Sex]

d_test.Sex = [1 if each =='male' else 0 for each in d_test.Sex]
d_train.info()
d_test.info()
# Lets see the mean values of Age columns:

print('Mean value of train data set Age column is: ', np.mean(d_train.Age))

print('Mean value of test data set Age column is: ', np.mean(d_test.Age))
d_train['Age'].fillna(d_train['Age'].mean(), inplace=True)

d_test['Age'].fillna(d_test['Age'].mean(), inplace=True)

d_test['Fare'].fillna(d_test['Fare'].mean(), inplace=True)
d_train.head()
plt.figure(figsize=[5,5])

plt.title('Passenger Survival')

sns.set(style='darkgrid')

ax = sns.countplot(x= 'Survived', data=d_train, palette='Set1')

d_train.loc[:,'Survived'].value_counts()
d_train.describe()
d_train.corr()
# True values:

y = d_train.Survived

# Features will be used for predictions:

x = d_train.drop(['Survived'], axis=1)
from sklearn.ensemble import RandomForestClassifier



# Defining model:

rf = RandomForestClassifier(n_estimators=100, random_state=42)



# Training Model:

rf.fit(x, y)



# Predicting test data set using trained model:

y_pred = rf.predict(d_test)



len(y_pred)
d_subm_rf = pd.DataFrame({'PassengerId': d_test['PassengerId'], 'Survived': y_pred})
d_subm_rf.to_csv('data_submission.csv', index = False)
from sklearn.naive_bayes import GaussianNB



# Defining NB model:

nb = GaussianNB()



# Training the model:

nb.fit(x,y)



# Predicting d_test:

y_pred_nb = nb.predict(d_test)
d_subm_nb = pd.DataFrame({'PassengerId': d_test['PassengerId'], 'Survived': y_pred_nb})
d_subm_nb.to_csv('data_submission_nb.csv', index = False)
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression()



lr.fit(x,y)



y_pred_lr = lr.predict(d_test)
d_subm_lr = pd.DataFrame({'PassengerId': d_test['PassengerId'], 'Survived': y_pred_lr})
d_subm_lr.to_csv('data_submission_lr.csv', index = False)