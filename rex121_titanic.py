###V2

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.tail(1)
###feature engineering & feature selection

###fillna Age w/ mean, Embarked&Cabin w/ mode

train_data1 = train_data.copy()

train_data1['Age']=train_data1['Age'].fillna(train_data1['Age'].mean())



###create one hot encoding

train_data2 = train_data1.copy()

pclass = pd.get_dummies(train_data2.Pclass)

sex = pd.get_dummies(train_data2.Sex)

train_data2 = pd.concat([pclass, sex, train_data2], axis=1)



###drop unnecessary columns

train_data3 = train_data2.copy()

# pd.pandas.set_option('display.max_columns', None)

train_data4 = train_data3.drop(columns=['PassengerId','Pclass','Name', 'Sex','Ticket', 'Cabin','Embarked'], axis=1)

train_data4.head(1)
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



Xb = train_data4.drop("Survived", axis='columns')

yb = train_data4["Survived"]



Xb_train, Xb_test, yb_train, yb_test = train_test_split(Xb, yb, test_size=0.3)



#apply MinMax scaler

from sklearn.preprocessing import MinMaxScaler

mm = MinMaxScaler()

Xb_train = mm.fit_transform(Xb_train)

Xb_test = mm.transform(Xb_test)



xgb_b = XGBClassifier()

xgb_b.fit(Xb_train, yb_train)



print('Accuracy of XGBoost Classifier on training set: {:.2f}'

     .format(xgb_b.score(Xb_train, yb_train)*100))

print('Accuracy of XGBoost Classifier on test set: {:.2f}'

     .format(xgb_b.score(Xb_test, yb_test)*100))
Xm = train_data4.drop("Survived", axis='columns')

ym = train_data4["Survived"]



#apply random over sampling

from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler()

Xm, ym = ros.fit_sample(Xm,ym)



Xm_train, Xm_test, ym_train, ym_test = train_test_split(Xm, ym, test_size=0.3)



#apply MinMax scaler

from sklearn.preprocessing import MinMaxScaler

mm = MinMaxScaler()

Xm_train = mm.fit_transform(Xm_train)

Xm_test = mm.transform(Xm_test)



xgb_mm = XGBClassifier()

xgb_mm.fit(Xb_train, yb_train)



print('Accuracy of XGBoost Classifier on training set: {:.2f}'

     .format(xgb_mm.score(Xm_train, ym_train)*100))

print('Accuracy of XGBoost Classifier on test set: {:.2f}'

     .format(xgb_mm.score(Xm_test, ym_test)*100))
Xfs = train_data4.drop('Survived', axis='columns')

yfs = train_data4['Survived']



Xfs_train, Xfs_test, yfs_train, yfs_test = train_test_split(Xfs, yfs, test_size=0.3, random_state=42)



from sklearn.preprocessing import StandardScaler

stdscalar = StandardScaler()

Xfs_train = stdscalar.fit_transform(Xfs_train)

Xfs_test = stdscalar.transform(Xfs_test)



xgb_fs = XGBClassifier()

xgb_fs.fit(Xfs_train, yfs_train)



print('Accuracy of XGBoost Classifier on training set: {:.2f}'

     .format(xgb_fs.score(Xfs_train, yfs_train)*100))

print('Accuracy of XGBoost Classifier on test set: {:.2f}'

     .format(xgb_fs.score(Xfs_test, yfs_test)*100))
Xf = train_data4.drop('Survived', axis='columns')

yf = train_data4['Survived']



#apply random over sampling

from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler()

Xf, yf = ros.fit_sample(Xf,yf)



Xf_train, Xf_test, yf_train, yf_test = train_test_split(Xf, yf, test_size=0.3, random_state=42)



from sklearn.preprocessing import StandardScaler

stdscalar = StandardScaler()

Xf_train = stdscalar.fit_transform(Xf_train)

Xf_test = stdscalar.transform(Xf_test)



xgb_f = XGBClassifier()

xgb_f.fit(Xf_train, yf_train)



print('Accuracy of XGBoost Classifier on training set: {:.2f}'

     .format(xgb_f.score(Xf_train, yf_train)*100))

print('Accuracy of XGBoost Classifier on test set: {:.2f}'

     .format(xgb_f.score(Xf_test, yf_test)*100))
###use test data to make prediction

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head(1)
test_data.info()
###feature engineering & feature selection

###fillna Age w/ mean, Embarked&Cabin w/ mode

test_data1 = test_data.copy()

test_data1['Age']=test_data1['Age'].fillna(test_data1['Age'].mean())



###create one hot encoding

test_data2 = test_data1.copy()

pclass = pd.get_dummies(test_data2.Pclass)

sex = pd.get_dummies(test_data2.Sex)

test_data2 = pd.concat([pclass, sex, test_data2], axis=1)

test_data2.head(1)
### apply model xgb_mm to test_data2

features = [1,2,3,'female','male','Age','SibSp','Parch','Fare']

X_test = test_data2[features].values

prediction = xgb_mm.predict(X_test)



print('Accuracy of XGBoost Classifier on test set: {:.2f}'

     .format(xgb_mm.score(X_test, prediction)*100))
output = pd.DataFrame({'PassengerId': test_data2.PassengerId, 'Survived': prediction})

output.to_csv('mytitanic_submission.csv', index=False)

print("Your submission was successfully saved!")