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
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline
train = pd.read_csv('../input/titanic/train.csv')

train.head()
plt.figure(figsize = (12,6))

sns.heatmap(train.isnull(), yticklabels = False, cbar = False, cmap = 'inferno')
columns = ['PassengerId','Name','Ticket','Cabin']

train.drop(columns, axis=1, inplace = True)
train.head()
plt.figure(figsize = (12,6))

sns.heatmap(train.isnull(), yticklabels = False, cbar = False, cmap = 'inferno')
sns.boxplot('Pclass','Age',data = train, palette = 'inferno')
def Class_Mean_Age(column):

    age = column[0]

    Class = column[1]

    

    if pd.isnull(age):

        if Class ==1:

            return 37

        elif Class == 2:

            return 29

        else:

            return 24

    else:

        return age
train['Age'] = train[['Age','Pclass']].apply(Class_Mean_Age, axis=1)
train.dropna(inplace = True)
plt.figure(figsize = (12,6))

sns.heatmap(train.isnull(), yticklabels = False, cbar = False, cmap = 'inferno')
sns.countplot('Survived', data = train, hue = 'Sex', palette = 'magma')



# Count of survivers by sex
sns.countplot('Pclass', data = train, hue = 'Survived', palette = 'inferno')



# Count of Survived by Passenger Class
Sex = pd.get_dummies(train['Sex'], drop_first = True) 

Embarked = pd.get_dummies(train['Embarked'], drop_first = True)
train.drop(['Sex','Embarked'], axis = 1, inplace = True)
train = pd.concat([train, Sex, Embarked], axis = 1)

train.head()
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(train.drop('Survived', axis = 1), train['Survived'], test_size = 0.3)
from sklearn.linear_model import LogisticRegression

logmodel =  LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,

          intercept_scaling=1, max_iter=1000, multi_class='auto',

          n_jobs=None, penalty='l2', random_state=None, solver='lbfgs',

          tol=0.0001, verbose=0, warm_start=False)

logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
print(classification_report(y_test,predictions))
test = pd.read_csv('../input/titanic/test.csv')

test.head()
columns = ['PassengerId','Name','Ticket','Cabin']

test.drop(columns, axis=1, inplace = True)
test['Age'] = test[['Age','Pclass']].apply(Class_Mean_Age, axis=1)
plt.figure(figsize = (12,6))

sns.heatmap(test.isnull(), yticklabels = False, cbar = False, cmap = 'inferno')
group = test.groupby('Pclass')

group.Fare.mean()
def meanFareClass(column):

    fare = column[0]

    Class = column[1]

    

    if pd.isnull(fare):

        if Class==1:

            return 94.28

        elif Class==2:

            return 22.2

        else:

            return 12.46

    else:

        return fare
test['Fare'] = test[['Fare','Pclass']].apply(meanFareClass, axis=1)
plt.figure(figsize = (12,6))

sns.heatmap(test.isnull(), yticklabels = False, cbar = False, cmap = 'inferno')
Sex = pd.get_dummies(test['Sex'], drop_first = True) 

Embarked = pd.get_dummies(test['Embarked'], drop_first = True)

test.drop(['Sex','Embarked'], axis = 1, inplace = True)
test = pd.concat([test, Sex, Embarked], axis = 1)

test.head()
predictions = logmodel.predict(test)
Id = pd.read_csv('../input/titanic/test.csv')

Id = Id.PassengerId
submission = pd.DataFrame({"PassengerId": Id,"Survived": predictions})

submission.head()
submission.to_csv('submission.csv',index=False)