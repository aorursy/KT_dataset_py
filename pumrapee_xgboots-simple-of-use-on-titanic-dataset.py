# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy import stats



# Visualization

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Prediction model

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold, cross_val_score

from sklearn.linear_model import LogisticRegression



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/train.csv')

train_data.head()
test_data = pd.read_csv('../input/test.csv')

test_data.head()
print(train_data.shape)

print(test_data.shape)
missing_train = train_data.isnull().sum()

missing_train
missing_train.sum() / train_data.shape[0] * 100
missing_test = test_data.isnull().sum()

missing_test
missing_test.sum() / test_data.shape[0] * 100
# Drop Embarked

train_data = train_data[train_data.Embarked.notnull()]

train_data.isnull().sum()
# Fill Age

train_data.Age.describe()
train_data.Age.median()
train_data.Age = train_data.Age.fillna(train_data.Age.median())

train_data.isnull().sum()
test_data.Age = test_data.Age.fillna(train_data.Age.median())

test_data.isnull().sum()
# Drop Cabin

train_data.Cabin.describe()
train_data = train_data.drop(columns=['Cabin'], axis=1)

train_data.head()
test_data = test_data.drop(columns=['Cabin'], axis=1)

test_data.head()
train_data.isnull().sum()
test_data.isnull().sum()
# Fill Fare

test_data.Fare.describe()
test_data.Fare = test_data.Fare.fillna(test_data.Fare.mean())

test_data.isnull().sum()
train_data.Ticket.describe()
train_feature = train_data.drop(columns=['PassengerId', 'Survived', 'Name', 'Ticket'], axis=1)

train_feature.head()
train_target = train_data['Survived']

train_target.head()
Sex_le = LabelEncoder()

Embarked_le = LabelEncoder()



#Sex_le = Sex_le.fit(train_feature['Sex'])

#Embarked_le = Embarked_le.fit(train_feature['Embarked'])



Sex_le_result = pd.DataFrame(Sex_le.fit_transform(train_feature['Sex']))

Embarked_le_result = pd.DataFrame(Embarked_le.fit_transform(train_feature['Embarked']))



train_feature['Sex'] = Sex_le_result

train_feature['Embarked'] = Embarked_le_result



train_feature.head()
train_feature.info()
train_feature.head()
sns.pairplot(train_data, hue="Survived")
model = xgb.XGBClassifier(base_score=0.8)



kfold = KFold(n_splits=10, random_state=0)



model.fit(train_feature, train_target)
results = cross_val_score(model, train_feature, train_target, cv=kfold)

print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
# Prepare test set