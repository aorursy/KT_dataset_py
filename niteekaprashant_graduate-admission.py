# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import statistics

import seaborn as sns

import pandas_profiling

from sklearn.model_selection import train_test_split



print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/Admission_Predict.csv")
#Correcting the field

data=data.rename(columns = {'Chance of Admit ':'Chance of Admit'})
#Starting the EDA with pandas profiler

pandas_profiling.ProfileReport(data)
import matplotlib.pyplot as plt

%matplotlib inline



#EDA Check the occurence of each of the fields

for var in data[['CGPA','GRE_Score','TOEFL_Score','LOR_','Research','SOP','University_Rating']]:

    plt.hist(data[var],bins=200)

    plt.xlabel(var)

    plt.ylabel("Occurrence")

    plt.show()
#Check the relation between the columns variables

for var in data[['CGPA','GRE_Score','TOEFL_Score','LOR_','Research','SOP','University_Rating']]:

    sns.relplot(x=var, y='Chance_of_Admit', data=data)

       
#Finding Outliers

for var in data[['CGPA','GRE_Score','TOEFL_Score','LOR_','Research','SOP','University_Rating']]:

    plt.boxplot(data[var])

    plt.xlabel(var)

    plt.ylabel("Occurrence")

    plt.show()
#Find the discrete and continuous varaibles

dist_vars = {}

for var in data[['CGPA','GRE_Score','TOEFL_Score','LOR_','Research','SOP','TOEFL_Score']]:

    counts = len(data[var].unique())

    dist_vars[var] = counts

    #print('No of Unique values of {} are :{}'.format(var,counts))

discrete = [item for item,key in dist_vars.items() if key < 10]

continuous = [item for item,key in dist_vars.items() if key > 10]

print("Discrete variables:",discrete)

print("Continuous variables:",continuous)
# Performing Test Train Split

X = data.iloc[:,1:8]

Y = data.iloc[:,-1].values

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.30,random_state=0)





#feature Scaling

from sklearn.preprocessing import MinMaxScaler

scale = MinMaxScaler()

X_train = scale.fit_transform(X_train)

X_test = scale.transform(X_test)





#Machine learning Algorithm

from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression()

linear_reg.fit(X_train,y_train)

y_pred_lr = linear_reg.predict(X_test)

print("MSE For LR is : {}".format(mean_squared_error(y_test,y_pred_lr)))

print("R squared value for LR is : {}".format(r2_score(y_test,y_pred_lr)))





from sklearn.ensemble import RandomForestRegressor

random_reg = RandomForestRegressor(max_depth=4,random_state = 42)

random_reg.fit(X_train,y_train)

y_pred_rf = random_reg.predict(X_test)

print("MSE For RF is : {}".format(mean_squared_error(y_test,y_pred_rf)))

print("R squared value for RF is : {}".format(r2_score(y_test,y_pred_rf)))



from sklearn.svm import SVR

regressor = SVR(kernel = 'rbf')

regressor.fit(X_train,y_train)

y_pred_svr = regressor.predict(X_test)

print("MSE For SVR is : {}".format(mean_squared_error(y_test,y_pred_svr)))

print("R squared value for SVR is : {}".format(r2_score(y_test,y_pred_svr)))
