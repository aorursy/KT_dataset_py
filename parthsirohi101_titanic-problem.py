# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'): 

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#read train data

train = pd.read_csv('/kaggle/input/titanic/train.csv')

print("Shape of Training Data:",train.shape)

#look at first 5 data obs

train.head()
#look at last 5 obs

train.tail()
#look at missing values

train.isnull().sum()
#sort in descending order

train.isnull().sum().sort_values(ascending=False)
sns.countplot(train['Survived'])
#load test data

test = pd.read_csv('/kaggle/input/titanic/test.csv')

print('Test Shape:',test.shape)
#look at test and train columns

print('Train Columns:',train.columns.tolist())

print('Test Columns:',test.columns.tolist())
#checking sample submission

sample_submission=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

print(sample_submission.head())
print(sample_submission.tail())
sns.countplot(x='Survived',hue='Sex',data=train)
#mean of target

round(np.mean(train['Survived']),2)

#finding percentages

train.isnull().mean().sort_values(ascending=False)
#plot of the above

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='plasma')
train.describe(include='all')
sns.countplot(train['Pclass'])
train.Name.value_counts()
train['Age'].hist(bins=50,color='Blue')
sns.countplot(train['SibSp'])
sns.countplot(train['Parch'])
train.Ticket.value_counts()
train['Fare'].hist(bins=50,color='purple')
train.Cabin.value_counts()
sns.countplot(train['Embarked'])
train.info()
#A clever idea at this point is to ask our friend seaborn to provide a features correlation matrix!

sns.heatmap(train.corr(),annot=True)
sns.countplot(x='Survived',hue='Pclass',data=train)
age_group=train.groupby("Pclass")['Age']

print(age_group.median())
train.loc[train.Age.isnull(),'Age']=train.groupby('Pclass').Age.transform('median')

print(train['Age'].isnull().sum())
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='plasma')
train.drop('Cabin',axis=1,inplace=True)
#age distribution

plt.figure(figsize=(16,8))

sns.distplot(train['Age'])

plt.title('Age Histogram')

plt.xlabel('Age')

plt.show()
train.isnull().sum().sort_values(ascending=False)
# Let's impute 'Embarked' missing values with the mode, which happens to be "S"!

from statistics import mode

train["Embarked"] = train["Embarked"].fillna(mode(train["Embarked"]))
#We have to transform our categorical to numeric to feed them in our models!

# Convert 'Sex' variable to integer form!

train["Sex"][train["Sex"] == "male"] = 0

train["Sex"][train["Sex"] == "female"] = 1



# Convert 'Embarked' variable to integer form!

train["Embarked"][train["Embarked"] == "S"] = 0

train["Embarked"][train["Embarked"] == "C"] = 1

train["Embarked"][train["Embarked"] == "Q"] = 2
train["Sex"]
# We'll drop the following features for now, but more to follow...

train.drop(['Name', 'Ticket'], axis = 1, inplace = True)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

# Fit a logistic regression model to our train data, 

#by converting 'Sex' to a dummy variable, to feed it into the model.

X_train, X_test, y_train, y_test = train_test_split(train.drop(['Survived'], axis = 1), 

                                                    train['Survived'], test_size = 0.2, 

                                                    random_state = 2)

logisticRegressor = LogisticRegression(max_iter=10000)

logisticRegressor.fit(X_train,y_train)
predictions = logisticRegressor.predict(X_test)
print(predictions)
round(np.mean(predictions),2)
from sklearn.metrics import classification_report, confusion_matrix



print(confusion_matrix(y_test, predictions))




accuracy = (91 + 50) / (91 + 9 + 29 + 50)

print('accuracy is: ' + str(round(accuracy, 2)))



output = pd.DataFrame({'PassengerId': X_test.PassengerId,

                       'Survived': predictions})

output.to_csv('my_submission.csv', index=False)