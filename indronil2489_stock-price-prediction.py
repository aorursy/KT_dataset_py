# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 22})



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/tesla-stock-price/Tesla.csv - Tesla.csv.csv', header=None)

df=df.iloc[1:]



df[0] = pd.to_datetime(df[0])

df[4] = pd.to_numeric(df[4])

df[3] = pd.to_numeric(df[3])

df[2] = pd.to_numeric(df[2])

df[1] = pd.to_numeric(df[1])

df.tail()
train = df[0:1200]

valid = df[1200:]



x_train = train.iloc[:,[1,2]]

y_train = train.iloc[:,4]

date_train=train.iloc[:,0]

x_valid = valid.iloc[:,[1,2]]

y_valid = valid.iloc[:,4]

date_valid=valid.iloc[:,0]

date_train[1200]

plt.rcParams.update({'font.size': 12})

fig, ax = plt.subplots(figsize = (8,3))

ax.plot(date_train, y_train, color = 'red', marker='', linewidth='0.75')

ax.plot(date_valid, y_valid, color = 'blue', marker='', linewidth='0.75')

plt.setp(ax.get_xticklabels(), rotation=45)

plt.legend(['Training Data', 'Testing Data'], loc='upper left')

ax.set(xlabel="Date",

       ylabel="Close Price",

       title="Stock Prices");

plt.show()
from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(x_train,y_train)

pred = reg.predict(x_valid)
import math

err=[]

a=0

SUM=0

for i in range(1201,1693):

    a=y_valid[i]-pred[i-1201]

    err.append(a)

    SUM = SUM + pow(a,2)

(SUM/492)
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

s_train = scaler.fit_transform(x_train)

s_valid = scaler.transform(x_valid)
from sklearn.neighbors import KNeighborsRegressor



knn = KNeighborsRegressor(algorithm='auto', leaf_size=10)

knn.fit(s_train, y_train)

pred2 = knn.predict(s_valid)
import math

err2=[]

a=0

SUM=0

for i in range(1201,1693):

    a=y_valid[i]-pred2[i-1201]

    err2.append(a)

    SUM = SUM + pow(a,2)

(SUM/492)
plt.rcParams.update({'font.size': 13})

fig, ax = plt.subplots(figsize = (12,6))

ax.plot(date_valid[100:300], y_valid[100:300], color = 'red', marker='.', linewidth='0.75')

ax.plot(date_valid[100:300], pred[100:300], color = 'blue', marker='.', linewidth='0.75')

ax.plot(date_valid[100:300], pred[100:300], color = 'yellow', marker='.', linewidth='0.75')

ax.plot(date_valid[100:300], pred2[100:300], color = 'green', marker='.', linewidth='0.75')

plt.xticks(rotation='45')

plt.legend(['Actual Data', 'Linear Regression', 'K-Nearest Neighbor'], loc='lower left')

ax.set(xlabel="Date",

       ylabel="Close Price",

       title="Stock Prices");

plt.show()
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=10)

lasso.fit(x_train,y_train)

pred3 = lasso.predict(x_valid)
import math

err3=[]

a=0

SUM=0

for i in range(1201,1693):

    a=y_valid[i]-pred3[i-1201]

    err3.append(a)

    SUM = SUM + pow(a,2)

SUM/492
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=0.5)

ridge.fit(x_train,y_train)

pred4 = ridge.predict(x_valid)
import math

err4=[]

a=0

SUM=0

for i in range(1201,1693):

    a=y_valid[i]-pred4[i-1201]

    err4.append(a)

    SUM = SUM + pow(a,2)

SUM/492
import math

mape=[]

a=0

SUM=0

for i in range(1201,1693):

    a=(abs((y_valid[i]-pred4[i-1201])/y_valid[i]))

    mape.append(a)

    SUM = SUM + a

(SUM/492)*100
plt.rcParams.update({'font.size': 11})

fig, ax = plt.subplots(figsize = (12,3))

ax.plot(date_valid[150:250], y_valid[150:250], color = 'red', marker='.', linewidth='0.75')

ax.plot(date_valid[150:250], pred[150:250], color = 'blue', marker='.', linewidth='0.75')

ax.plot(date_valid[150:250], pred3[150:250], color = 'green', marker='.', linewidth='0.75')

ax.plot(date_valid[150:250], pred4[150:250], color = 'yellow', marker='.', linewidth='0.75')

plt.xticks(rotation='45')

plt.legend(['Actual Data', 'Simple Linear Regression', 'Lasso Regression', 'Ridge Regression'], loc='best')

ax.set(xlabel="(a) Tesla Dataset",

       ylabel="Close Price",

       title="Stock Prices");

plt.show()
from sklearn.ensemble import RandomForestRegressor 



rforest = RandomForestRegressor(n_estimators = 100, random_state = 1) 

rforest.fit(x_train,y_train)

pred51 = rforest.predict(x_valid)

rforest = RandomForestRegressor(n_estimators = 250, random_state = 0) 

rforest.fit(x_train,y_train)

pred52 = rforest.predict(x_valid)

rforest = RandomForestRegressor(n_estimators = 500, random_state = 1) 

rforest.fit(x_train,y_train)

pred53 = rforest.predict(x_valid)

rforest = RandomForestRegressor(n_estimators = 1000, random_state = 0) 

rforest.fit(x_train,y_train)

pred54 = rforest.predict(x_valid)
plt.rcParams.update({'font.size': 11})

fig, ax = plt.subplots(figsize = (12,3))

ax.plot(date_valid[150:250], y_valid[150:250], color = 'red', marker='.', linewidth='0.75')



ax.plot(date_valid[150:250], pred53[150:250], color = 'orange', marker='o', linewidth='2.75')

ax.plot(date_valid[150:250], pred52[150:250], color = 'blue', marker='.', linewidth='0.95')

ax.plot(date_valid[150:250], pred51[150:250], color = 'yellow', marker='.', linewidth='0.75')

ax.plot(date_valid[150:250], pred54[150:250], color = 'green', marker='.', linewidth='0.75')

plt.xticks(rotation='45')



plt.legend(['Actual Data', 'Random Forest (n=1000)', 'Random Forest (n=500)', 'Random Forest (n=250)', 'Random Forest (n=100)'], loc='lower left')

ax.set(xlabel="Date",

       ylabel="Close Price",

       title="Stock Prices");

plt.show()
import math

err5=[]

a=0

SUM=0

for i in range(1201,1693):

    a=y_valid[i]-pred54[i-1201]

    err5.append(a)

    SUM = SUM + pow(a,2)

SUM/492
import math

mape=[]

a=0

SUM=0

for i in range(1201,1693):

    a=(abs((y_valid[i]-pred51[i-1201])/y_valid[i]))

    mape.append(a)

    SUM = SUM + a

(SUM/492)*100
from sklearn import svm

svmm = svm.SVR(gamma='auto_deprecated', kernel='linear', max_iter=-1).fit(s_train,y_train)

pred1 = svmm.predict(s_valid)
import math

err1=[]

a=0

SUM=0

for i in range(1201,1693):

    a=y_valid[i]-pred1[i-1201]

    err1.append(a)

    SUM = SUM + pow(a,2)

SUM/492
import math

mape=[]

a=0

SUM=0

for i in range(1201,1693):

    a=(abs((y_valid[i]-pred1[i-1201])/y_valid[i]))

    mape.append(a)

    SUM = SUM + a

(SUM/492)*100
plt.rcParams.update({'font.size': 11})

fig, ax = plt.subplots(figsize = (12,3))

ax.plot(date_valid[150:250], y_valid[150:250], color = 'red', marker='.', linewidth='0.75')

ax.plot(date_valid[150:250], pred2[150:250], color = 'yellow', marker='.', linewidth='0.75')

ax.plot(date_valid[150:250], pred1[150:250], color = 'green', marker='.', linewidth='0.75')

plt.xticks(rotation='45')

plt.legend(['Actual Data', 'K-Nearest Neighbor', 'Support Vector Machine'], loc='lower left')

ax.set(xlabel="Date",

       ylabel="Close Price",

       title="Stock Prices");

plt.show()
from keras.layers import Dense, Activation, Dropout

from keras.models import Sequential

ann = Sequential()



ann.add(Dense(20, activation = 'relu', input_dim = 3))

ann.add(Dense(units = 75, activation = 'relu'))

ann.add(Dense(units = 75, activation = 'relu'))

ann.add(Dense(units = 75, activation = 'relu'))

ann.add(Dense(units = 1))
ann.compile(optimizer = 'adam',loss = 'mean_squared_error')

ann.fit(s_train, y_train, batch_size = 70, epochs = 300)

pred6 = ann.predict(s_valid)
import math

err6=[]

a=0

SUM=0

for i in range(1201,1693):

    a=y_valid[i]-pred6[i-1201]

    err6.append(a)

    SUM = SUM + pow(a,2)

SUM/492
import math

mape=[]

a=0

SUM=0

for i in range(1201,1693):

    a=(abs((y_valid[i]-pred6[i-1201])/y_valid[i]))

    mape.append(a)

    SUM = SUM + a

(SUM/492)*100
slp = Sequential()



# Adding the input layer and the first hidden layer

slp.add(Dense(16, input_dim = 3))



# Adding the output layer

slp.add(Dense(units = 1))



slp.compile(optimizer = 'adam',loss = 'mse')

slp.fit(s_train, y_train, batch_size =50, epochs = 300)

pred7 = slp.predict(s_valid)
import math

err7=[]

a=0

SUM=0

for i in range(1201,1693):

    a=y_valid[i]-pred7[i-1201]

    err7.append(a)

    SUM = SUM + pow(a,2)

SUM/492
import math

mape=[]

a=0

SUM=0

for i in range(1201,1693):

    a=(abs((y_valid[i]-pred7[i-1201])/y_valid[i]))

    mape.append(a)

    SUM = SUM + a

(SUM/492)*100
plt.rcParams.update({'font.size': 11})

fig, ax = plt.subplots(figsize = (12,3))

ax.plot(date_valid[150:250], y_valid[150:250], color = 'red', marker='.', linewidth='0.75')

ax.plot(date_valid[150:250], pred7[150:250], color = 'yellow', marker='.', linewidth='0.75')

ax.plot(date_valid[150:250], pred6[150:250], color = 'green', marker='.', linewidth='0.75')

plt.xticks(rotation='45')

plt.legend(['Actual Data', 'Single Layer Perceptron', 'Multilayer Perceptron'], loc='lower left')

ax.set(xlabel="Date",

       ylabel="Close Price",

       title="Stock Prices");

plt.show()
df[4] = pd.to_numeric(df[4])

sma = df.iloc[1191:,4]



pred101=[]

for j in range(1201,1693):

    sum=0

    for i in range(0,10):

        sum=sum+sma[j-i]

    pred101.append(sum/10)

    

df[4] = pd.to_numeric(df[4])

sma = df.iloc[1186:,4]



pred102=[]

for j in range(1201,1693):

    sum=0

    for i in range(0,15):

        sum=sum+sma[j-i]

    pred102.append(sum/15)

    

df[4] = pd.to_numeric(df[4])

sma = df.iloc[1171:,4]



pred103=[]

for j in range(1201,1693):

    sum=0

    for i in range(0,30):

        sum=sum+sma[j-i]

    pred103.append(sum/30)
import math

err10=[]

a=0

SUM=0

for i in range(1201,1693):

    a=y_valid[i]-pred101[i-1201]

    err10.append(a)

    SUM = SUM + pow(a,2)

SUM/492
import math

mape=[]

a=0

SUM=0

for i in range(1201,1693):

    a=(abs((y_valid[i]-pred103[i-1201])/y_valid[i]))

    mape.append(a)

    SUM = SUM + a

(SUM/492)*100
df[4] = pd.to_numeric(df[4])

sma = df.iloc[1191:,4]

pred121=[]

for j in range(1201,1693):

    sum=0

    for i in range(0,10):

        sum=sum+(sma[j-i]*((100/55)*(10-i)))

    pred121.append(sum/100)

    

df[4] = pd.to_numeric(df[4])

sma = df.iloc[1186:,4]

pred122=[]

for j in range(1201,1693):

    sum=0

    for i in range(0,15):

        sum=sum+(sma[j-i]*((100/120)*(15-i)))

    pred122.append(sum/100)

    

df[4] = pd.to_numeric(df[4])

sma = df.iloc[1171:,4]

pred123=[]

for j in range(1201,1693):

    sum=0

    for i in range(0,30):

        sum=sum+(sma[j-i]*((100/465)*(30-i)))

    pred123.append(sum/100)
a=0

w=[]

d=[]

for i in range(0,15):

        a=(100/120)*(15-i)

        d.append(14-i+1)

        w.append(a)

    

plt.rcParams.update({'font.size': 10})

fig, ax = plt.subplots(figsize = (7,4))



plt.bar(w, w, width=0.35)

plt.xticks(w, [15,14,13,12,11,10,9,8,7,6,5,4,3,2,1])

plt.show()
import math

err12=[]

a=0

SUM=0

for i in range(1201,1693):

    a=y_valid[i]-pred121[i-1201]

    err12.append(a)

    SUM = SUM + pow(a,2)

SUM/492
import math

mape=[]

a=0

SUM=0

for i in range(1201,1693):

    a=(abs((y_valid[i]-pred121[i-1201])/y_valid[i]))

    mape.append(a)

    SUM = SUM + a

(SUM/492)*100
plt.rcParams.update({'font.size': 11})

fig, ax = plt.subplots(figsize = (12,3))

ax.plot(date_valid[150:250], y_valid[150:250], color = 'red', marker='.', linewidth='0.75')

ax.plot(date_valid[150:250], pred101[150:250], color = 'yellow', marker='.', linewidth='0.75')

ax.plot(date_valid[150:250], pred102[150:250], color = 'green', marker='.', linewidth='0.75')

ax.plot(date_valid[150:250], pred103[150:250], color = 'blue', marker='.', linewidth='0.75')

plt.xticks(rotation='45')



plt.legend(['Actual Data', '10-day Simple Moving Average', '15-day Simple Moving Average', '30-day Simple Moving Average'], loc='lower left')

ax.set(xlabel="(a) Tesla Dataset",

       ylabel="Close Price",

       title="Stock Prices");

plt.show()
plt.rcParams.update({'font.size': 11})

fig, ax = plt.subplots(figsize = (12,3))

ax.plot(date_valid[150:250], y_valid[150:250], color = 'red', marker='.', linewidth='0.75')

ax.plot(date_valid[150:250], pred121[150:250], color = 'yellow', marker='.', linewidth='0.75')

ax.plot(date_valid[150:250], pred122[150:250], color = 'green', marker='.', linewidth='0.75')

ax.plot(date_valid[150:250], pred123[150:250], color = 'blue', marker='.', linewidth='0.75')

plt.xticks(rotation='45')



plt.legend(['Actual Data', '10-day Weighted Moving Average', '15-day Weighted Moving Average', '30-day Weighted Moving Average'], loc='lower left')

ax.set(xlabel="(a) Tesla Dataset",

       ylabel="Close Price",

       title="Stock Prices");

plt.show()
plt.rcParams.update({'font.size': 16})

fig, ax = plt.subplots(figsize = (16,8))

ax.plot(date_valid[100:250], y_valid[100:250], color = 'red', marker='.', linewidth='0.75')

ax.plot(date_valid[100:250], pred101[100:250], color = 'yellow', marker='.', linewidth='0.75')

ax.plot(date_valid[100:250], pred121[100:250], color = 'green', marker='.', linewidth='0.75')

plt.xticks(rotation='45')



plt.legend(['Actual Data', 'Simple Moving Average', 'Weighted Moving Average'], loc='lower left')

ax.set(xlabel="Date",

       ylabel="Close Price",

       title="Stock Prices");

plt.show()
a=0

pred201=[]

pred202=[]

pred203=[]

pred201.append(y_train[1200])

pred202.append(y_train[1200])

pred203.append(y_train[1200])

for i in range(1201,1693):

    a=pred201[i-1201]+(0.75)*(y_valid[i]-pred201[i-1201])

    pred201.append(a) 

    

for i in range(1201,1693):

    a=pred202[i-1201]+(0.5)*(y_valid[i]-pred202[i-1201])

    pred202.append(a) 

    

for i in range(1201,1693):

    a=pred203[i-1201]+(0.3)*(y_valid[i]-pred203[i-1201])

    pred203.append(a) 

#pred20[1200:]
plt.rcParams.update({'font.size': 11})

fig, ax = plt.subplots(figsize = (12,3))

ax.plot(date_valid[150:250], y_valid[150:250], color = 'red', marker='.', linewidth='0.75')

ax.plot(date_valid[150:250], pred201[150:250], color = 'yellow', marker='.', linewidth='0.75')

ax.plot(date_valid[150:250], pred202[150:250], color = 'green', marker='.', linewidth='0.75')

ax.plot(date_valid[150:250], pred203[150:250], color = 'blue', marker='.', linewidth='0.75')

plt.xticks(rotation='45')



plt.legend(['Actual Data', 'Exponential Smoothing (α =0.75)', 'Exponential Smoothing (α =0.5)', 'Exponential Smoothing (α =0.3)'], loc='lower left')

ax.set(xlabel="(a) Tesla Dataset",

       ylabel="Close Price",

       title="Stock Prices");

plt.show()
import math

err20=[]

a=0

SUM=0

for i in range(1201,1693):

    a=y_valid[i]-pred201[i-1201]

    err20.append(a)

    SUM = SUM + pow(a,2)

SUM/492
import math

mape=[]

a=0

SUM=0

for i in range(1201,1693):

    a=(abs((y_valid[i]-pred203[i-1201])/y_valid[i]))

    mape.append(a)

    SUM = SUM + a

(SUM/492)*100
pred13=[]

pred13.append(y_train[1200])

for i in range(1201,1693):

    a=y_valid[i]

    pred13.append(a) 
import math

err13=[]

a=0

SUM=0

for i in range(1201,1693):

    a=y_valid[i]-pred13[i-1201]

    err13.append(a)

    SUM = SUM + pow(a,2)

SUM/492
import math

mape=[]

a=0

SUM=0

for i in range(1201,1693):

    a=(abs((y_valid[i]-pred13[i-1201])/y_valid[i]))

    mape.append(a)

    SUM = SUM + a

(SUM/492)*100
plt.rcParams.update({'font.size': 13})

fig, ax = plt.subplots(figsize = (12,6))

ax.plot(date_valid[100:300], y_valid[100:300], color = 'red', marker='.', linewidth='0.75')

#ax.plot(date_valid[100:250], pred20[101:251], color = 'green', marker='.', linewidth='0.75')

ax.plot(date_valid[100:300], pred13[100:300], color = 'blue', marker='.', linewidth='0.75')

plt.xticks(rotation='45')

plt.legend(['Actual Data', 'Naive Approach'], loc='lower left')

ax.set(xlabel="Date",

       ylabel="Close Price",

       title="Stock Prices");

plt.show()
plt.rcParams.update({'font.size': 14})

fig, ax = plt.subplots(figsize = (16,6))

ax.plot(date_valid[100:490], err20[100:490], color = 'red', marker='.', linewidth='0.25')

ax.plot(date_valid[100:490], err13[100:490], color = 'green', marker='.', linewidth='0.25')

ax.plot(date_valid[100:490], err1[100:490], color = 'blue', marker='.', linewidth='0.25')

ax.plot(date_valid[100:490], err5[100:490], color = 'yellow', marker='.', linewidth='0.25')

ax.plot(date_valid[100:490], err7[100:490], color = 'orange', marker='.', linewidth='0.25')

plt.xticks(rotation='45')

ax.set(xlabel="Date",

       ylabel="Close Price",

       title="Stock Prices");

plt.show()
plt.rcParams.update({'font.size': 14})

fig, ax = plt.subplots(figsize = (12,6))

ax.plot(date_valid[100:250], y_valid[100:250], color = 'red', marker='.', linewidth='0.75')

ax.plot(date_valid[100:250], pred101[100:250], color = 'green', marker='.', linewidth='0.75')

#ax.plot(date_valid[0:250], pred6[0:250], color = 'yellow', marker='.', linewidth='1.75')

plt.xticks(rotation='45')

ax.set(xlabel="Date",

       ylabel="Close Price",

       title="Stock Prices");

plt.show()