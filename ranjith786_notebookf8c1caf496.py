# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns #plotting

sns.set_style('whitegrid')

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



from sklearn.linear_model import LogisticRegression

data_train  = pd.read_csv("../input/train.csv",sep=",")

data_test   = pd.read_csv("../input/test.csv",sep=",")
data_train.dropna(subset=["Embarked"],inplace=True)

data_train.drop(labels=["PassengerId","Name"],inplace=True,axis=1)

data_train.Embarked.value_counts()
data_train.Embarked.describe()
data_train.info()
data_train.Embarked.value_counts()
data_train.Embarked.value_counts().plot(kind='bar')
data_train[data_train["Survived"]==0].Embarked.value_counts().plot(kind='bar',color="Red",alpha=0.5,width=0.4)

data_train[data_train["Survived"]==1].Embarked.value_counts().plot(kind='bar',color="Green",alpha=0.5,width=0.4)
data_train.groupby("Survived").Embarked.value_counts().plot(kind='bar',color="r")
data_train[data_train["Survived"]==0].Age.plot(kind='hist',color="Red",alpha=0.5,width=1)

data_train[data_train["Survived"]==1].Age.plot(kind='hist',color="Green",alpha=0.5,width=1)
#data_train[["Embarked","Survived"]].groupby("Embarked").mean().plot(kind='bar')

embarked_percentage = data_train[["Embarked","Survived"]].groupby("Embarked",as_index=False).mean()

embarked_percentage
sns.barplot(x="Embarked",y="Survived",data=embarked_percentage,order=["S","C","Q"])
data  = data_train.groupby("Survived")
sns.countplot(x="Embarked",data=data.get_group(0),order=["S","C","Q"])
sns.countplot(x="Embarked",data=data.get_group(1),order=["S","C","Q"])
sns.countplot(x="Survived",hue="Embarked",data=data_train[["Survived","Embarked"]])

#data_train[["Survived","Embarked"]]
data_train.Sex.value_counts()

data_train.Sex.describe()
survived_sex = data_train[["Sex","Survived"]].groupby("Sex",as_index=False).mean()
sns.barplot(x="Sex",y="Survived",data=survived_sex)
data_train.Age.describe()
average_age = data_train.Age.mean()

std_age = data_train.Age.std()

age_null_sum = data_train.Age.isnull().sum()

n = data_train.Age.isnull()

average_age, std_age,age_null_sum
n.value_counts()
rand_age_train  = np.random.randint((average_age-std_age),(average_age+std_age),age_null_sum)
rand_age_train
data_train.Age.dropna().astype(int)

data_train.Age.describe()
data_train["Age"][np.isnan(data_train["Age"])] = rand_age_train
data_train.Age.describe()
data_train.Age.hist(bins=70)
from sklearn.linear_model import LogisticRegression

data_train.head()
data_train["Family"] = data_train["SibSp"] + data_train["Parch"]
data_train.head()
data_train["Family"].describe()
data_train["Family"].loc[data_train["Family"] > 1] = 1 

data_train["Family"].loc[data_train["Family"] ==0] = 0
data_train["Family"].describe()
data_train.drop(labels=["SibSp","Parch","Cabin","Ticket"],axis=1,inplace=True)

data_train
data_train["Sex"].describe()

data_train["Sex"].loc[data_train["Age"]<=16] = "Child"
data_train["Sex"].describe(), data_train["Sex"].value_counts()
Sex_dummies = pd.get_dummies(data_train["Sex"])



Embarked_dummies = pd.get_dummies(data_train["Embarked"])
Embarked_dummies.head(3)
data_train = pd.concat([data_train,Embarked_dummies,Sex_dummies])
data_train
data_train["Child"].loc[np.isnan(data_train["Child"])] = 0.0

data_train["female"].loc[np.isnan(data_train["female"])] = 0.0

data_train["male"].loc[np.isnan(data_train["male"])] = 0.0
data_train["C"].loc[np.isnan(data_train["C"])] = 0.0

data_train["S"].loc[np.isnan(data_train["S"])] = 0.0

data_train["Q"].loc[np.isnan(data_train["Q"])] = 0.0

data_train
data_train.loc[np.isnan(data_train)] = 0.0
model = LogisticRegression()
data_train.columns
x_train = data_train[["Age","C","S","Q","Child","female","Family","Pclass"]]

y_train = data_train[["Survived"]]

x_train

model.fit(x_train,y_train)