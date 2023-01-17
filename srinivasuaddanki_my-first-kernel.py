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
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
titanic_train = pd.read_csv('../input/titanic/train.csv')

titanic_test =  pd.read_csv('../input/titanic/test.csv')
titanic_train.head()
sns.heatmap(titanic_train.isnull(),yticklabels = False, cbar = False, cmap = 'viridis')

sns.heatmap(titanic_test.isnull(),yticklabels = False, cbar = False, cmap = 'viridis')
sns.set_style('whitegrid')

sns.countplot(x='Survived', data = titanic_train, palette = 'RdBu_r')
sns.set_style("whitegrid")

sns.countplot(x='Survived',data=titanic_train,hue = 'Sex',palette = 'RdBu_r')
sns.set_style("whitegrid")

sns.countplot(x='Survived',data=titanic_train,hue = 'Pclass',palette = 'rainbow')
sns.set_style('whitegrid')

sns.distplot(titanic_train['Age'].dropna(),kde = False, color = 'darkred',bins = 30)
sns.set_style('whitegrid')

sns.distplot(titanic_train[titanic_train['Survived'] == 1]['Age'].dropna(),kde = False, color = 'darkred',bins = 30)
plt.figure(figsize = (12,7))

sns.boxplot(x='Pclass',y='Age',data = titanic_train,palette = 'winter')
def calc_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    if pd.isnull(Age):    

        if Pclass == 1:

            return 38

        elif Pclass == 2:

            return 30

        else:

            return 24

    else:

        return Age   

    
titanic_train['Age'] = titanic_train[['Age','Pclass']].apply(calc_age,axis = 1)

titanic_test['Age'] = titanic_train[['Age','Pclass']].apply(calc_age,axis = 1)
sns.heatmap(titanic_train.isnull(),yticklabels = False, cbar = False, cmap = 'viridis')
titanic_train.drop('Cabin',axis = 1 ,inplace = True)

titanic_test.drop('Cabin',axis = 1 ,inplace = True)
sns.heatmap(titanic_train.isnull(),yticklabels = False, cbar = False, cmap = 'viridis')
sns.heatmap(titanic_test.isnull(),yticklabels = False, cbar = False, cmap = 'viridis')
titanic_train.dropna(inplace=True)
titanic_train.info()
sex= pd.get_dummies(titanic_train['Sex'],drop_first = True)

embark = pd.get_dummies(titanic_train['Embarked'],drop_first = True)

sex1= pd.get_dummies(titanic_test['Sex'],drop_first = True)

embark1 = pd.get_dummies(titanic_test['Embarked'],drop_first = True)
titanic_train.drop(['Sex','Embarked','Name','Ticket','Fare'],axis = 1,inplace = True)

titanic_test.drop(['Sex','Embarked','Name','Ticket','Fare'],axis = 1,inplace = True)
titanic_train = pd.concat([titanic_train,sex,embark],axis =1)

titanic_test = pd.concat([titanic_test,sex1,embark1],axis =1)
titanic_train.head()
titanic_test.tail()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(titanic_train.drop('Survived',axis=1), 

                                                    titanic_train['Survived'], test_size=0.30, 

                                                    random_state=200)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
X_train = titanic_train.drop(["Survived","PassengerId"], axis=1)

Y_train = titanic_train["Survived"]

X_test  = titanic_test.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log
submission = pd.DataFrame({

        "PassengerId": titanic_test["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('submission.csv', index=False)

print("Your submission was successfully saved!")