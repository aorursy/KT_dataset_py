# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv', index_col='PassengerId')

test = pd.read_csv('../input/test.csv', index_col='PassengerId')

print(train.shape)

print(test.shape)

train.head()
import seaborn as sns

%matplotlib inline

sns.countplot(data=train, x='Pclass', hue='Survived')
sns.countplot(data=train, x='Embarked', hue='Survived')
sns.barplot(data=train, x='Pclass', y='Fare', hue='Survived')
sns.pointplot(data=train, x='Pclass', y='Fare', hue='Survived')
sns.countplot(data=train, x='Sex', hue='Survived')
sns.countplot(data=train, x='Cabin', hue='Survived')
sns.lmplot(data=train, x='Age', y='Fare', hue='Survived', fit_reg=False)
sns.lmplot(data=train[train['Fare'] < 300], x='Age', y='Fare', hue='Survived', fit_reg=False)
import matplotlib.pyplot as plt



figure, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)

figure.set_size_inches(18, 8)



sns.countplot(data=train, x='Embarked', hue='Survived', ax=ax1)

sns.countplot(data=train, x='Sex', hue='Survived', ax=ax2)

sns.countplot(data=train, x='Pclass', hue='Survived', ax=ax3)
train_data = train



# 'Sex' : str to int(true or false)

train_data.loc[train_data['Sex']=='male', 'Sex']=0

train_data.loc[train_data['Sex']=='female', 'Sex']=1



# 'Embarked' : str to boolean

train_data["Embarked_C"] = train_data["Embarked"] == "C"

train_data['Embarked_S'] = train_data['Embarked'] == 'S'

train_data['Embarked_Q'] = train_data['Embarked'] == 'Q'



# 'Age' : NaN -> mean value

age_mean = train_data['Age'].mean()

train_data.loc[pd.isnull(train_data['Age']), 'Age'] = age_mean



# feature selection and split data into x and y

train_x = train_data[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked_C', 'Embarked_S', 'Embarked_Q']]

train_y = train_data['Survived']



print(train_x.shape)

print(train_y.shape)
train_x.head()
train_y.head()
# train_x[pd.isnull(train_x['Age'])].head()
test_data = test



# 'Sex' : str to int(true or false)

test_data.loc[test_data['Sex']=='male', 'Sex']=0

test_data.loc[test_data['Sex']=='female', 'Sex']=1



# 'Embarked' : str to boolean

test_data["Embarked_C"] = test_data["Embarked"] == "C"

test_data['Embarked_S'] = test_data['Embarked'] == 'S'

test_data['Embarked_Q'] = test_data['Embarked'] == 'Q'



# 'Age' : NaN -> mean value

age_test_mean = test_data['Age'].mean()

test_data.loc[pd.isnull(test_data['Age']), 'Age'] = age_test_mean



# 'Fare' : NaN -> mean value

fare_test_mean = test_data['Fare'].mean()

test_data.loc[pd.isnull(test_data['Fare']), 'Fare'] = fare_test_mean



# feature selection and split data into x and y

test_x = test_data[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked_C', 'Embarked_S', 'Embarked_Q']]

# test_y = test_data['Survived']



print(test_x.shape)

# print(test_y.shape)



test_x.head()
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

from sklearn.model_selection import train_test_split



# make ml model

rfc = RandomForestClassifier(max_depth=3, random_state=0)



# training

rfc.fit(train_x, train_y)



# prediction

pred = rfc.predict(test_x)



print(pred.shape)
# make result dataframe and write a csv file

submission = pd.DataFrame({

    "PassengerId": test.index,

    "Survived": pred

})

submission.to_csv('titanic_prediction.csv', index=False)