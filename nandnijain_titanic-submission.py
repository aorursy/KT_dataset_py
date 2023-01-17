# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns #visualization 

import matplotlib.pyplot as plt #visualization

%matplotlib inline

from sklearn.model_selection import train_test_split #data split

from sklearn.preprocessing import OneHotEncoder #forCatogoricalVariable

from sklearn.linear_model import LogisticRegression #for training

from sklearn.metrics import accuracy_score #for accuracy





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
total_train = pd.read_csv("../input/titanic/train.csv")

total_test = pd.read_csv("../input/titanic/test.csv")
total_train.dropna(axis=0, subset=['Survived'], inplace=True)

y = total_train.Survived

total_train.drop(['Survived'], axis=1, inplace=True)



X_train, X_test, y_train, y_test = train_test_split(total_train, y, random_state = 0)

X_train.head()
X_train["Name"].nunique()
X_train.shape
X_train.drop(["Cabin","Ticket","Name"], axis = 1, inplace = True)
X_train.head()
X_train.isnull().sum()
plt.figure(figsize = (5,5))

sns.barplot(x = X_train["Sex"], y = y_train);

plt.title("SEX survived")
X_train.head()
xx_train = pd.get_dummies(X_train, columns = ["Sex"])

xx_train.head()
plt.figure(figsize = (5,5))

sns.barplot(x = xx_train["Embarked"], y = y_train);

plt.title("Embarked survived")
xx_train.isnull().sum()
xx_train["Embarked"].fillna( "C", inplace = True)
xx_train.isnull().sum()
xx_train = pd.get_dummies(xx_train, columns = ["Embarked"]) 

xx_train.head()
sns.boxplot(x = xx_train["Pclass"], y = xx_train["Age"]);

plt.show()

means = xx_train.groupby(["Pclass"])["Age"].mean()
means
xx_train.loc[(xx_train["Pclass"]==1) & xx_train["Age"].isnull(),"Age"] = means.get(1)

xx_train.loc[(xx_train["Pclass"]==2) & xx_train["Age"].isnull(),"Age"] = means.get(2)

xx_train.loc[(xx_train["Pclass"]==3) & xx_train["Age"].isnull(),"Age"] = means.get(3)
xx_train.head()
xx_train.isnull().sum()
X_test.head()
X_test.drop(["Ticket","Cabin","Name"], axis = 1, inplace = True)

X_test = pd.get_dummies(X_test, columns = ["Sex"])

X_test.head()
X_test = pd.get_dummies(X_test, columns = ["Embarked"])

X_test.head()
mean2 = X_test.groupby(["Pclass"])["Age"].mean()
X_test.loc[(X_test["Pclass"]==1) & X_test["Age"].isnull(),"Age"] = mean2.get(1)

X_test.loc[(X_test["Pclass"]==2) & X_test["Age"].isnull(),"Age"] = mean2.get(2)

X_test.loc[(X_test["Pclass"]==3) & X_test["Age"].isnull(),"Age"] = mean2.get(3)
columns = xx_train.columns

columns
col = ['Pclass', 'Age','Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Parch']
logreg = LogisticRegression( random_state = 0)

logreg.fit(xx_train[col],y_train)

y_pred = logreg.predict(X_test[col])
score = accuracy_score(y_pred, y_test)
score
total_test.drop(["Cabin","Ticket","Name"],axis = 1, inplace = True)
total_train.drop(["Cabin","Ticket","Name"],axis = 1, inplace = True)
total_test = pd.get_dummies(total_test, columns = ["Sex"])
total_train = pd.get_dummies(total_train, columns = ["Sex"])
total_test = pd.get_dummies(total_test, columns = ["Embarked"])
total_train = pd.get_dummies(total_train, columns = ["Embarked"])
meanf = total_test.groupby(["Pclass"])["Age"].mean()
meanH = total_train.groupby(["Pclass"])["Age"].mean()
total_test.loc[(total_test["Pclass"]==1) & total_test["Age"].isnull(),"Age"] = meanf.get(1)

total_test.loc[(total_test["Pclass"]==2) & total_test["Age"].isnull(),"Age"] = meanf.get(2)

total_test.loc[(total_test["Pclass"]==3) & total_test["Age"].isnull(),"Age"] = meanf.get(3)
total_train.loc[(total_train["Pclass"]==1) & total_train["Age"].isnull(),"Age"] = meanH.get(1)

total_train.loc[(total_train["Pclass"]==2) & total_train["Age"].isnull(),"Age"] = meanH.get(2)

total_train.loc[(total_train["Pclass"]==3) & total_train["Age"].isnull(),"Age"] = meanH.get(3)
total_test.isnull().sum()
total_train.isnull().sum()
total_test["Fare"].fillna(value = total_test["Fare"].mean(), inplace = True)
total_test.isnull().any()
logreg.fit(total_train[col],y)
final_pred = logreg.predict(total_test[col])
total_test.head()
final_given = pd.read_csv("../input/titanic/gender_submission.csv")
final_given
final_pred
f_score = accuracy_score(final_pred,final_given["Survived"])
f_score
submission = pd.DataFrame({'PassengerId' : total_test.PassengerId, 'Survived':final_pred})

submission.to_csv("submission.csv", index = False)