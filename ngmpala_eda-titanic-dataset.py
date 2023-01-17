# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas_profiling

import matplotlib.pyplot as plt# Plotting library for Python programming language and it's numerical mathematics extension NumPy

import seaborn as sns# Provides a high level interface for drawing attractive and informative statistical graphics

%matplotlib inline

sns.set()



from subprocess import check_output



# Any results you write to the current directory are saved as output.
titanic_data = pd.read_csv('https://raw.githubusercontent.com/gampa123/AI/master/Machine%20Learning/titanic_train.csv')

titanic_data.head(2)
titanic_data.info()
#Replace all the missing values in Embarked with mode

titanic_data.Embarked = titanic_data.Embarked.fillna(titanic_data['Embarked'].mode()[0])

#Replace all the missing valeus in Age with Median

median_age = titanic_data.Age.median()

titanic_data.Age.fillna(median_age, inplace = True)

#Drop the column Cabin. There are so manny values are missing

titanic_data.drop('Cabin',axis=1,inplace = True)
titanic_data.info()
titanic = titanic_data.drop(['Name','Ticket','Sex','SibSp','Parch','Embarked'], axis = 1)

titanic.head(2)
titanic_data.head(2)
X = titanic.loc[:,titanic.columns != 'Survived']

y = titanic.Survived

X.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X_train,y_train)
y_pred_train = logreg.predict(X_train) 

y_pred_test = logreg.predict(X_test)
from sklearn.metrics import accuracy_score

print('Accuracy score for test data is:', accuracy_score(y_test,y_pred_test))
titanic_data['FamilySize'] = titanic_data['SibSp']+titanic_data['Parch']+1

titanic_data['GenderClass'] = titanic_data.apply(lambda x: 'child' if x['Age']<15 else x['Sex'],axis=1)
titanic_data.head(2)
titanic_data = pd.get_dummies(titanic_data,columns=['GenderClass','Embarked'],drop_first=True)
titanic_revisit = titanic_data.drop(['Name','Ticket','Sex','SibSp','Parch'], axis = 1)
sns.pairplot(titanic_data[["Fare","Age","Pclass","Survived"]],vars = ["Fare","Age","Pclass"],hue="Survived", dropna=True,markers=["o", "s"])

plt.title('Pair Plot')
X_revisit = titanic_revisit.loc[:,titanic_revisit.columns != 'Survived']

y_revisit = titanic_revisit.Survived

X_revisit_train, X_revisit_test, y_revisit_train, y_revisit_test = train_test_split(X_revisit, y_revisit, test_size=0.20, random_state=1)

logreg.fit(X_revisit_train,y_revisit_train)
y_pred_revisit_train = logreg.predict(X_revisit_train) 

y_pred_revisit_test = logreg.predict(X_revisit_test)

print('Accuracy score for test data is:', accuracy_score(y_revisit_test,y_pred_revisit_test))