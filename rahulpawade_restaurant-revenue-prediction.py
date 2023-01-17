import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

l = LabelEncoder()
train = pd.read_csv("../input/restaurant-revenue-prediction/train.csv.zip")
train.head()
train.describe()
train.shape
train["City"].value_counts()
train["City"] = (train["City"]=="İstanbul").astype(np.int)
train["City"].value_counts()
train["City Group"].value_counts()
train["City Group"] = l.fit_transform(train["City Group"])
train["City Group"].value_counts()
train["Type"].value_counts()
train["Type"] = l.fit_transform(train["Type"])
train["Type"].value_counts()
train["year"]=0

for i in range(len(train["Open Date"])):

       a=train["Open Date"][i].split("/")

       train["year"][i]=a[2]



train["month"]=0

for i in range(len(train["Open Date"])):

       a=train["Open Date"][i].split("/")

       train["month"][i]=a[0]

    

train["day_No"]=0

for i in range(len(train["Open Date"])):

       a=train["Open Date"][i].split("/")

       train["day_No"][i]=a[1]

train.head()
plt.figure(figsize=(50,50))

sns.heatmap(train.corr(),annot=True)
test = pd.read_csv("../input/restaurant-revenue-prediction/test.csv.zip")
test.head()
test.shape
test["City"] = (test["City"]=="İstanbul").astype(np.int)

test["Type"] = l.fit_transform(test["Type"])

test["City Group"] = l.fit_transform(test["City Group"])
test["year"]=0

for i in range(len(test["Open Date"])):

       a=test["Open Date"][i].split("/")

       test["year"][i]=a[2]



test["month"]=0

for i in range(len(test["Open Date"])):

       a=test["Open Date"][i].split("/")

       test["month"][i]=a[0]

    

test["day_No"]=0

for i in range(len(test["Open Date"])):

       a=test["Open Date"][i].split("/")

       test["day_No"][i]=a[0]

test.head()
x_train = train.drop(columns=["Id","revenue","Open Date"],axis=1)

y_train = train["revenue"]

x_test = test.drop(columns=["Id","Open Date"],axis=1)

x_train.shape,x_test.shape,y_train.shape
from sklearn.preprocessing import StandardScaler

s = StandardScaler()
xtrain = s.fit_transform(x_train)

x_train = pd.DataFrame(x_train,columns=x_train.columns)

xtest = s.fit_transform(x_test)

x_test = pd.DataFrame(x_test,columns=x_test.columns)
x_train.shape,x_test.shape,y_train.shape
from xgboost import XGBRegressor

model = XGBRegressor()

model.fit(x_train,y_train)

y_pred = model.predict(x_test)
s = pd.read_csv("../input/restaurant-revenue-prediction/sampleSubmission.csv")
s.head()
f = {"Id":test["Id"],"Prediction":y_pred}
f = pd.DataFrame(f)
f.to_csv("submission.csv",index=False)
f.head()