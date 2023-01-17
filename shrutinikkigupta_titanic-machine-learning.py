# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
sns.set(color_codes=True)

sns.set_style('whitegrid')
#Reading Data

train = pd.read_csv('../input/train.csv')

train.head()
test = pd.read_csv('../input/test.csv')

test.head()
sns.countplot(x='Survived', data=train)

plt.title("Survived")
sns.lmplot(x = 'Age', y = 'Fare', hue='Pclass', col='Survived', data = train)

plt.ylim(0,300)
train.info()

test.info()
train.isnull().any()
test.isnull().any()
df=pd.concat([train,test])

df['Age'] = df['Age'].fillna(df['Age'].median())

df["Sex"] = df["Sex"].map({"male": 0, "female":1})

df.drop(['Ticket', 'Name'], axis = 1, inplace = True)

df.head()



df['Fare'] = df['Fare'].fillna(df['Fare'].median())

df['Cabin'].isnull().sum()

df['Cabin'] = np.where(df['Cabin'].notnull(), 1, 0)

df['Embarked'].value_counts()

df['Embarked'] = df['Embarked'].fillna('S')

df['Embarked'] = df['Embarked'].map({'S': 0, 'C' : 1, 'Q' : 2})

df.head()
X = df.iloc[:, [0, 2,3,4,5,6,7,8]].values

y = df.iloc[:, 1].values



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)





#Fitting logistic regression to training set



classifier = LogisticRegression(random_state = 0)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)





#Confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)



from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

accuracy
decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, y_train)

y_pred = decision_tree.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

accuracy
from sklearn.ensemble import ExtraTreesClassifier

ExtraTreesClassifier = ExtraTreesClassifier()

ExtraTreesClassifier.fit(X_train, y_train)

y_pred = ExtraTreesClassifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

accuracy
category_column =['Survived'] 

for x in category_column:

    print (x)

    print (df[x].value_counts())





for col in category_column:

    b, c = np.unique(df[col], return_inverse=True) 

    df[col] = c



df.head()     
df.drop(['Pclass','Sex','SibSp','Parch','Fare','Embarked','Age','Fare','Cabin'],axis=1,inplace=True)

df.to_csv('submission.csv', index=False)