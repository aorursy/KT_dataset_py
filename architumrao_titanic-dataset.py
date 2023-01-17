import pandas as pd

import numpy as np

import scipy.stats

from scipy.stats import mode
train=pd.read_csv("train.csv")
train.drop(["Name","Ticket","Cabin"],axis=1,inplace=True)

train.info()

train.head()
med_age=train.Age.median()

train.Age=train.Age.fillna(med_age)

train.Embarked=train.Embarked.fillna(train.Embarked.mode())

train.info()
a = []

for i in range(1,len(train['Fare'])):

    a.append(train['Embarked'][i])
mode(train.Embarked.tolist())[0][0]
train.Embarked.value_counts()

train.Embarked=train.Embarked.fillna("S")

train.info()
train["Gender"]=train.Sex.map({"male":1,"female":0})

train["Port"]=train.Embarked.map({"S":1,"C":2,"Q":3})

train.info()
train.drop(["Sex","Embarked"],axis=1,inplace=True)
cols=train.columns.tolist()

cols = cols[1:2] + cols[0:1] + cols[2:]

train=train[cols]
train_data=train.values
from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier(n_estimators=100)
model=model.fit(train_data[:,2:],train_data[:,0])
test=pd.read_csv("test.csv")
test.drop(["Name","Ticket","Cabin"],axis=1,inplace=True)
test.Age=test.Age.fillna(med_age)
test.info()
mean_fare0=test.pivot_table(index="Pclass",values="Fare")

mean_fare0
test.Fare=test[["Fare","Pclass"]].apply(lambda row: mean_fare[row["Pclass"]] if pd.isnull(row["Fare"]) else row["Fare"],axis=1)
test["Gender"]=test.Sex.map({"male":1,"female":0})

test["Port"]=test.Embarked.map({"S":1,"C":2,"Q":3})

test.info()
test.drop(["Sex","Embarked"],axis=1,inplace=True)
test_data=test.values
output=model.predict(test_data[:,1:])
result=np.c_[test_data[:,0].astype(int),output.astype(int)]
result_df=pd.DataFrame(result[:,0:2],columns=["Passenger_id","Survived"])
result_df.to_csv("result1.csv")
result_df.shape