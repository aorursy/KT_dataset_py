import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
train=pd.read_csv("/home/nihal/Desktop/lab1_sem2/train.csv")
train

j=0

for i in train.columns:

    if(train[i].isna().sum()!=0):

        j=j+1

print(j)    
train.describe()
train1=train.drop(['id'],axis=1)
plt.matshow(train1.corr())

plt.show()
import seaborn as sns
corr=train1.corr()
corr
import math

for i in corr.columns:

    if(math.isnan(getattr(corr,i)[0])):

        print(i)

# corr.b2[0]
train1=train1.drop(['b10','b12','b26','b61','b81'],axis=1)
train1
corr=train1.corr()
corr
label=train1.label
train1=train1.drop(['label'],axis=1)
corr=train1.corr()
corr
corr.columns.size
u2={'a0','a1','a2','a3','a4','a5','a6'}
u1=[]

train3=train1

for i in corr.columns:

    if(i not in u1):

        if(i not in u2 ):

            for j in range(corr.columns.size):

                if(getattr(corr,i)[j]!=1 and (getattr(corr,i)[j]>=0.9 or getattr(corr,i)[j]<=-0.9)):

        #             u1.append(corr.columns[j])

        #             print(i)

                    print(i,", ",corr.columns[j]," : ",getattr(corr,i)[j])

        #             print(getattr(corr,i)[j]

                    if(corr.columns[j] not in u2 ):

                        u1.append(corr.columns[j])

                        train3=train3.drop([corr.columns[j]],axis=1)

            corr=train3.corr()
# train3['a0']=train1.a0

# train3['a2']=train1.a2

# train3['a3']=train1.a3

train3
corr=train3.corr()
corr
from sklearn import ensemble

a1=ensemble.RandomForestRegressor(n_estimators=100,random_state=4)
from sklearn.preprocessing import MinMaxScaler

scaler_x = MinMaxScaler()

# scaler_y = MinMaxScaler()

train4=scaler_x.fit_transform(train3)

# label

train4
test=scaler_x.fit_transform(test)
test
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(train4, label, test_size=0.2, random_state=0)
X_train
X_val
a1.fit(X_train,y_train)
ans=a1.predict(test)
ans
a=a1.predict(X_val)
from sklearn.metrics import mean_squared_error

from math import sqrt

rms = sqrt(mean_squared_error(y_val, a))
rms
pd.DataFrame(ans).to_csv("/home/nihal/Desktop/lab1_sem2/ans1.csv")
df=pd.read_csv("/home/nihal/Desktop/lab1_sem2/ans1.csv")
df['id']=test1.id

df
pd.DataFrame(df).to_csv("/home/nihal/Desktop/lab1_sem2/ans2.csv")
a1.score(X_val,y_val)
from sklearn.metrics import mean_squared_error

from math import sqrt



rms = sqrt(mean_squared_error(y_val, ans))
format(rms)
test1=pd.read_csv("/home/nihal/Desktop/lab1_sem2/test.csv")
test=test1
u1
test.drop(columns=u1,inplace=True)

test=test.drop(['b10','b12','b26','b61','b81','id'],axis=1)

test
X_train1, X_val1, y_train1, y_val1 = train_test_split(train1, label, test_size=0.2, random_state=1)
train1
a1.fit(X_train1,y_train1)
ans1=a1.predict(X_val1)
ans1
a1.score(X_val1,y_val1)
rms = sqrt(mean_squared_error(y_val1, ans1))
rms