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
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
train_data.isna().sum()
train_data.Age.describe()
train_data.Age.hist()
mean_age=train_data.Age.mean()
train_data.Age.fillna(mean_age,inplace=True)
train_data.Cabin.describe()
train_data.Cabin.value_counts()
train_data["Cabin_Class"]= train_data.apply(lambda x : str(x["Cabin"])[0],axis=1)
train_data["Cabin_Class"].value_counts()
train_data["Cabin_Class"].replace('n','C',inplace=True)
train_data["Cabin_Class"].value_counts()
train_data.drop("Cabin",axis=1,inplace=True)
train_data.isna().sum()
train_data.Embarked.value_counts()
train_data.loc[train_data.Embarked.isna()==True]
train_data.Embarked.fillna("S",inplace=True)
train_data.isna().sum()
train_data.drop("Name",axis=1,inplace=True)
for i in ["Pclass",'Sex','Embarked','Cabin_Class']:

     train_data=pd.concat([train_data,pd.get_dummies(train_data[i], prefix=i,dummy_na=True)],axis=1).drop([i],axis=1)
train_data.head()
train_data.drop(["PassengerId","Ticket"],axis=1,inplace=True)
train_data.drop(["Pclass_nan",'Sex_nan','Embarked_nan','Cabin_Class_nan'],axis=1,inplace=True)
train_data.columns
train_data.isna().sum()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_data.drop('Survived',axis=1), 

                                                    train_data['Survived'], test_size=0.30, 

                                                    random_state=101)
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report

print(classification_report(y_test,predictions))
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
test_data.isna().sum()
test_data.Age.hist()
mean_age=test_data.Age.mean()

print(mean_age)
test_data.Age.fillna(mean_age,inplace=True)

test_data["Cabin_Class"]= test_data.apply(lambda x : str(x["Cabin"])[0],axis=1)

test_data["Cabin_Class"].replace('n','C',inplace=True)

test_data.Cabin_Class.value_counts()
test_data.drop("Cabin",axis=1,inplace=True)

test_data.Embarked.value_counts()

test_data.Fare.describe()
test_data.Fare.fillna(35.6,inplace=True)

test_data.isna().sum()
for i in ["Pclass",'Sex','Embarked','Cabin_Class']:

     test_data=pd.concat([test_data,pd.get_dummies(test_data[i], prefix=i,dummy_na=False)],axis=1).drop([i],axis=1)
test_data.drop(["Ticket","Name"],axis=1,inplace=True)
test_data.columns
test_data["Cabin_Class_T"]=0
test_data.columns
featureset=[i for i in test_data.columns if i != "PassengerId" ]
featureset
test_pred = logmodel.predict(test_data[featureset])
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': test_pred})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")