# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

display(train_data.head())

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()

test_data["Fare"]=test_data["Fare"].fillna(value=test_data["Fare"].mean())

test_data.isnull().sum()











train_data.fillna({"Age":train_data["Age"].mean()},inplace= True)

test_data.fillna({"Age":train_data["Age"].mean()},inplace= True)

display(train_data.info())

test_data.info()





train_data["Cabin"].fillna(value=0,inplace= True)

test_data["Cabin"].fillna(value=0,inplace= True)

train_data["Cabin"].where(train_data["Cabin"]==0,1,inplace= True)

test_data["Cabin"].where(test_data["Cabin"]==0,1,inplace= True)





total = train_data.groupby("Sex").count()[["Survived"]]

survived = train_data.groupby("Sex").sum()[["Survived"]]



survival_rate = survived/total

survival_rate.columns = ["Survival rate"]

display(survival_rate)



ax = survival_rate.plot(kind="bar" )

ax.get_children()[1].set_color("r")

y = train_data["Survived"]



features = ["Pclass","Age","SibSp","Fare","Cabin"]

X= train_data[features]

X_test = test_data[features]



sex = pd.get_dummies(train_data["Sex"],drop_first=True)

sex_test = pd.get_dummies(test_data["Sex"],drop_first=True)



embarked = pd.get_dummies(train_data["Embarked"],drop_first=True)

embarked_test = pd.get_dummies(test_data["Embarked"],drop_first=True)



X = pd.concat([X,sex,embarked],axis=1)

X_test = pd.concat([X_test,sex_test,embarked_test],axis=1)



display(X.head())

display(X_test.head())
#Random forest model



from sklearn import model_selection

from sklearn.ensemble import RandomForestClassifier



from sklearn.model_selection import cross_val_score











model = RandomForestClassifier()

model.fit(X, y)

predictions = model.predict(X_test)



scores = model_selection.cross_val_score(model,X,y,cv=5)

print(scores.mean())





output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

display(output.head())

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")



from sklearn import metrics

a = model.predict(X)

print(metrics.classification_report(y,a))

print(metrics.mean_squared_error(y,a))