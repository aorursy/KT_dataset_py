import pandas as pd

from math import *

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
data=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")

submit=pd.read_csv("../input/submit.csv")

submit['User_ID']=test['User_ID'] 

submit['Product_ID']=test['Product_ID']



data.head(2)

#data.shape

#$submit.Product_ID

Index=[]

for i in range(550068):

    Index.append(i+1)

    i+=1
data["Index"]=Index

data.head()
p_id=data.groupby('Product_ID')

ans=p_id.Index.count()



pid=[]

for i in data.Product_ID:

    pid.append(ans[i])



#ans["P00000142"]
#data.Product_ID.value_counts()
#data.pid_count[data.Product_ID=="P00114942"]
data["pid_count"]=pid

data.head()
data["check"]=1

test["check"]=0
data.head(4)
test["Purchase"]=np.nan

test.head()
#test.info()

Index=[]

for i in range(233599):

    Index.append(i+1)

    i+=1

test['Index']=Index
p_id=test.groupby('Product_ID')

ans=p_id.Index.count()



pid=[]

for i in test.Product_ID:

    pid.append(ans[i])
test['pid_count']=pid
test.head(2)
combined=pd.concat([data,test])
combined.info()
combined.fillna(0,inplace=True)

#age_cr=data.groupby(["Age","Marital_Status"])



#age_cr['Purchase'].mean()
#print (len(data.Product_ID.value_counts()))

#data.Product_ID.unique()
from sklearn.preprocessing import LabelEncoder 

le = LabelEncoder()

#combined['Gender']=le.fit_transform(combined['Gender'])

#combined['Age']=le.fit_transform(combined['Age'])

combined.pop("Index")
df=pd.get_dummies(combined['Gender'])
combined=pd.concat([combined,df],axis=1)
combined.head()
combined.pop('Gender')

combined.pop('F')
df1=pd.get_dummies(combined['City_Category'])

combined=pd.concat([combined,df1],axis=1)



df1=pd.get_dummies(combined['Stay_In_Current_City_Years'])

combined=pd.concat([combined,df1],axis=1)
combined['Product_ID']=le.fit_transform(combined['Product_ID'])
combined.head()
df1=pd.get_dummies(combined['Age'])

combined=pd.concat([combined,df1],axis=1)
combined.pop("C")

combined.pop("4+")

combined.pop("55+")

combined.pop("Age")

combined.pop("City_Category")

combined.pop("Stay_In_Current_City_Years")

combined.head()
train = combined[combined.check==1]

test = combined[combined.check==0]
from sklearn.metrics import mean_squared_error
train.pop("check")
x=train.drop(labels='Purchase',axis=1)

y=train['Purchase']

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
'''from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators=100,random_state = 42)

rfr.fit(x_train, y_train)

y_pred=rfr.predict(x_test)

print(sqrt(mean_squared_error(y_test, y_pred)))'''
params = {}

params["objective"] = "reg:linear"

params["eta"] = 0.03

params["min_child_weight"] = 10

params["subsample"] = 0.8

params["colsample_bytree"] = 0.7

params["silent"] = 1

params["max_depth"] = 18

#params["max_delta_step"]=2

params["seed"] = 0

 #params['eval_metric'] = "auc"

plst1 = list(params.items())

num_rounds1 = 1100
import xgboost as xgb

xgdmat=xgb.DMatrix(x_train,y_train)



final_gb1=xgb.train(plst1,xgdmat,num_rounds1)



tesdmat=xgb.DMatrix(x_test)

y_pred=final_gb1.predict(tesdmat)
print(np.sqrt(mean_squared_error(y_test, y_pred)))
'''#from sklearn.ensemble import

from sklearn.metrics import mean_squared_error

import xgboost as xgb

xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.08,

                max_depth = 9, alpha = 10, n_estimators = 100)

xg_reg.fit(x_train,y_train)

preds = xg_reg.predict(x_test)

print(np.sqrt(mean_squared_error(y_test, preds)))'''
test.head()
test.pop("check")

#test.pop("Index")

test.pop("Purchase")
tesdmat=xgb.DMatrix(test)

answer=final_gb1.predict(tesdmat)
submit.Purchase.mean()
submit['Purchase']=answer
submit.Purchase[submit.Purchase < 0]=submit.Purchase.mean()
submit.describe()
submit.head()
submit.to_csv("submit.csv",index=False)

