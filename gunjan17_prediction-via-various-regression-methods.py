# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#selecting the various regression method



from sklearn import linear_model

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import BaggingRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.tree import DecisionTreeRegressor

#importing the data & liberary

import pandas as pd

import numpy as np



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
# importing the liberary for plotting the data

import seaborn as sns

import matplotlib.pyplot as plt
#checking the data

train.head()

#useful colums are

#pclass ,sex,age,fare,embarked



#checking the useful data

train.describe()
#cleaning the data

#pclass ,sex,age,fare,embarked

train["Age"].isnull().sum()

train["Age"] = train["Age"].fillna(train["Age"].median())

train["Age"].isnull().sum()





#cleaning the data

#pclass ,sex,age,fare,embarked

test["Age"].isnull().sum()

test["Age"] = test["Age"].fillna(test["Age"].median())

test["Age"].isnull().sum()
train["Embarked"].isnull().sum()

train["Embarked"]=train["Embarked"].fillna("S")



test["Embarked"].isnull().sum()

test["Embarked"]=test["Embarked"].fillna("S")
#ploting the data for embarked

sns.factorplot('Embarked','Survived', data=train,size=4,aspect=3)

fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))



sns.countplot(x='Embarked', data=train, ax=axis1)

sns.countplot(x='Survived', hue="Embarked", data=train, order=[1,0], ax=axis2)





# group by embarked, and get the mean for survived passengers for each value in Embarked

embark_perc = train[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()

sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q'],ax=axis3)



plt.show()
test["Fare"] = test["Fare"].fillna(test["Fare"].median())
test.loc[test["Sex"] == "male", "Sex"] = 0

test.loc[test["Sex"]=="female","Sex"]=1



test.loc[test["Embarked"]=="S","Embarked"] =0

test.loc[test["Embarked"]=="C","Embarked"] =1

test.loc[test["Embarked"]=="Q","Embarked"] =2





train.loc[train["Sex"] == "male", "Sex"] = 0

train.loc[train["Sex"]=="female","Sex"]=1



train.loc[train["Embarked"]=="S","Embarked"] =0

train.loc[train["Embarked"]=="C","Embarked"] =1

train.loc[train["Embarked"]=="Q","Embarked"] =2
new_col = ["Age","Sex","Embarked","Pclass","Fare","SibSp","Parch"]
x_train = train[new_col]

x_test = test[new_col]

y_train = train["Survived"]
#apply the linear regression

reg = linear_model.LinearRegression(fit_intercept=True,normalize =True)

reg.fit(x_train,y_train)
reg.score(x_train,y_train)
rf  = RandomForestRegressor(criterion='mse',n_estimators=2000,max_leaf_nodes=3000,oob_score=True)
rf.fit(x_train,y_train)
rf.score(x_train,y_train)
br = BaggingRegressor( n_estimators=2000,oob_score = True)
br.fit(x_train,y_train)
br.score(x_train,y_train)
ab = AdaBoostRegressor(n_estimators=5000)
ab.fit(x_train,y_train)
ab.score(x_train,y_train)
et = ExtraTreesRegressor(n_estimators=3000,bootstrap=True,oob_score=True,)
et.fit(x_train,y_train)
et.score(x_train,y_train)
gb = GradientBoostingRegressor(n_estimators=3000 )
gb.fit(x_train,y_train)
gb.score(x_train,y_train)
dt = DecisionTreeRegressor(max_depth=30)
dt.fit(x_train,y_train)
dt.score(x_train,y_train)
#so we apply the various method so the we can find the maximum performence and 

#also performence can tuned by the changing or adding extra argument in the various regression method



#so we can see the scores of various regression methods