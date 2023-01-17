# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))
data_1= pd.read_csv('../input/Admission_Predict.csv')

data_2= pd.read_csv('../input/Admission_Predict.csv')
cleancolumn = []

for i in range(len(data_1.columns)):

    cleancolumn.append(data_1.columns[i].replace(' ','').lower())

data_1.columns = cleancolumn
#Considering not much of data cleaning it required

data_1.columns

#Check of null or na values

#data_1.toeflscore.isna()==True

#grescore and toeflscore columns to be scaled

from sklearn.preprocessing import MinMaxScaler
X = data_1.drop('chanceofadmit', axis=1)

y = data_1.chanceofadmit
scaler = MinMaxScaler()

scl_X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

#scl_y = pd.DataFrame(scaler.fit_transform(y), columns=y.columns)
#importing algo

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
rdm_reg = RandomForestRegressor()

rdm_reg1 = RandomForestRegressor()



X_train,X_test,y_train,y_test = train_test_split(X,y)

X_train1,X_test1,y_train1,y_test1 = train_test_split(scl_X,y)
rdm_reg.fit(X_train,y_train)

rdm_reg1.fit(X_train1,y_train1)
y_predict = rdm_reg.predict(X_test)

y_predict1 = rdm_reg1.predict(X_test1)
from sklearn.metrics import r2_score,mean_squared_error,roc_auc_score

print (r2_score(y_test,y_predict))

print (mean_squared_error(y_test,y_predict))

#print (roc_auc_score(y_test,y_predict))

print ("-----------scaled results below")

print (r2_score(y_test1,y_predict1))

print (mean_squared_error(y_test1,y_predict1))
import matplotlib.pyplot as plt

%matplotlib inline
plt.scatter(y_test,y_predict)

plt.scatter(y_test1,y_predict1)
from sklearn.linear_model import LinearRegression
log_reg = LinearRegression()

log_reg1 = LinearRegression()
log_reg.fit(X_train,y_train)

log_reg1.fit(X_train1,y_train1)
y_log_predict = log_reg.predict(X_test)

y_log_predict1 = log_reg.predict(X_test1)
print (r2_score(y_test,y_log_predict))

print (mean_squared_error(y_test,y_log_predict))

print ("--------Scaled result below-------------")

print (r2_score(y_test1,y_log_predict1))

print (mean_squared_error(y_test1,y_log_predict1))
plt.scatter(y_test,y_log_predict)

plt.scatter(y_test1,y_predict1)