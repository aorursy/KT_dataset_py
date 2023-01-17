# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv("../input/titanic/train.csv")

train_data.head()
survived = train_data[train_data['Survived']==1]

nonsurvived = train_data[train_data['Survived']==0]
print("Total =", len(train_data))

print("Total Survived in training set = ",len(survived),

      "\nPercent of Survived in training set = ",1.*(len(survived)/len(train_data))*100)

plt.figure(figsize=(6,12))

plt.subplot(211)

sns.countplot(x='Pclass',data=train_data)

plt.subplot(212)

sns.countplot(x='Pclass',hue='Survived', data=train_data)
plt.figure(figsize=(6,20))

plt.subplot(311)

sns.countplot(x='Sex',data=train_data)

plt.subplot(312)

sns.countplot(x='Sex',hue='Pclass', data=train_data)

plt.subplot(313)

sns.countplot(x='Sex',hue='Survived', data=train_data)
plt.figure(figsize=(10,20))

plt.subplot(411)

sns.countplot(x='Embarked',data=train_data)

plt.subplot(412)

sns.countplot(x='Embarked',hue='Survived', data=train_data)

plt.subplot(413)

sns.countplot(x='Embarked',hue='Pclass', data=train_data)

plt.subplot(414)

sns.countplot(x='Embarked',hue='Sex', data=train_data)
train_data['Age'].hist(bins = 40)
sns.heatmap(train_data.isnull(), cbar=False)
train_data.drop(['Name', 'Cabin','Ticket','Embarked'], axis=1, inplace =True)
train_data.head()
sns.heatmap(train_data.isnull(), cbar=False)
plt.figure(figsize = (15,10))

sns.boxplot(x='Sex', y='Age', data= train_data)
def fill_age(data):

    age = data[0]

    sex = data[1]

    

    if pd.isnull(age):

        if sex is 'male':

            return 29

        else: 

            return 25

    else:

        return age
train_data['Age'] = train_data[['Age', 'Sex']].apply(fill_age,axis=1)
sns.heatmap(train_data.isnull(), cbar=False)
train_data['Age'].hist(bins=20)
male = pd.get_dummies(train_data['Sex'])
train_data.drop(['female'], axis=1, inplace =True)
train_data.head()
X = train_data.drop('Survived',axis=1).values

y = train_data['Survived'].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(X_train, y_train)
y_predict_test = classifier.predict(X_test)

y_predict_test
cm = confusion_matrix(y_test, y_predict_test)

sns.heatmap(cm, annot=True, fmt="d")
from sklearn.metrics import classification_report

print(classification_report(y_test, y_predict_test))