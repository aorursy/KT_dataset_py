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
test = pd.read_csv("/kaggle/input/titanic/test.csv")
train= pd.read_csv("/kaggle/input/titanic/train.csv")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')
round(np.mean(train['Survived']),2)
train.isnull().sum().sort_values(ascending=False)
train.isnull().mean().sort_values(ascending=False)
sns.heatmap(train.isnull(), yticklabels=False,cbar=False, cmap='plasma')
print(train.describe(include='all'))
sns.countplot(train['Pclass'])
train.Name.value_counts()
sns.countplot(x= 'Survived', hue='Sex', data=train)

train['Age'].hist(bins=50, color='blue')
sns.countplot(train['SibSp'])

sns.countplot(train.Parch)
train['Ticket'].value_counts()
train['Fare'].hist(bins=50, color='green')
train['Cabin'].value_counts()
sns.countplot(train.Embarked)
sns.heatmap(train.corr(), annot=True)

sns.countplot(x='Survived', hue='Pclass', data=train)
age_group=train.groupby('Pclass')['Age']
print(age_group.median())
train.loc[train.Age.isnull(),'Age']= train.groupby("Pclass").Age.transform('median')

train.drop('Cabin', axis=1, inplace=True)
plt.figure(figsize=(16,8))
sns.distplot(train["Age"])
plt.title("Age Histogram")
plt.xlabel("Age")
plt.show()
from statistics import mode
train['Embarked']=train['Embarked'].fillna(mode(train['Embarked']))
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1

train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2
train.drop(['Name', 'Ticket'], axis = 1, inplace = True)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,  y_test = train_test_split(train.drop(['Survived'], axis=1),
                                        train['Survived'], test_size=0.2, random_state=2)
    
from sklearn.linear_model import LogisticRegression
logisticRegression = LogisticRegression()
logisticRegression.fit(X_train, y_train)
predictions = logisticRegression.predict(X_test)
print(predictions)
round(np.mean(predictions), 2)


from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, predictions))


accuracy = (90 + 49) / (90 + 10 + 30 + 49)
print('accuracy is: ' + str(round(accuracy, 2)))
y_test.to_csv('kaggle_sub2.csv', index=False)