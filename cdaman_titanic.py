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
import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
import seaborn as sns
sns.heatmap(train.isnull() ,cmap = 'viridis')
train.drop('Cabin',axis = 1,inplace = True)
sns.boxplot(x = 'Pclass',y = 'Age', data = train,)
def age(inp):

    ag = inp[0]

    pc = inp[1]

    if pd.isnull(ag):

        if pc == 1:

            return 37

        elif pc == 2:

            return 29

        else:

            return 24

    else:

        return ag

    
train['Age'] = train[['Age','Pclass']].apply(age,axis = 1

                                            )
train.head()
train.isnull()
train.dropna(inplace = True)
train.columns
train.drop(['PassengerId','Name','Ticket'],axis = 1,inplace = True)
sex = pd.get_dummies(train['Sex'],drop_first=True)

embark = pd.get_dummies(train['Embarked'],drop_first=True)
sex.head()
embark.head()
train = pd.concat([train,sex,embark],axis = 1)
train.head()
train.drop(['Sex','Embarked'],axis = 1, inplace = True)
X = train.drop('Survived',axis = 1)

y = train['Survived']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25,random_state=101)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)
from sklearn.metrics import classification_report

print(classification_report(y_test,model.predict(X_test)))
sns.heatmap(test.isnull() ,cmap = 'viridis')
test.drop('Cabin',axis = 1,inplace = True)
sns.boxplot(x = 'Pclass', y = 'Age', data = test)
def age(inp):

    ag = inp[0]

    pc = inp[1]

    if pd.isnull(ag):

        if pc == 1:

            return 42

        elif pc == 2:

            return 37

        else:

            return 34

    else:

        return ag

    
test['Age'] = test[['Age','Pclass']].apply(age,axis = 1

                                            )
test
test.drop(['PassengerId','Name','Ticket'],axis = 1, inplace  = True)
test.info()
test['Age'][test['Fare'].isnull() == True] = 30
sns.boxplot(y = 'Fare', x = 'Pclass', data = test)
def age(inp):

    ag = inp[0]

    pc = inp[1]

    if pd.isnull(ag):

        if pc == 3:

            return 30

    else:

        return ag

    
test['Fare'] = test[['Fare','Pclass']].apply(age,axis = 1

                                            )
test.dropna()
sex = pd.get_dummies(test['Sex'],drop_first=True)

embark = pd.get_dummies(test['Embarked'],drop_first=True)
test = pd.concat([test,sex,embark],axis = 1)
test.drop(['Sex','Embarked'],axis = 1, inplace = True)
test.head()
predict = model.predict(test)
predict
gender_submission.head()
gender_submission['Survived'] = predict
gender_submission.to_csv('output.csv')