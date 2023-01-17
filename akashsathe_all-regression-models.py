#import all required libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

#import data
raw_data=pd.read_csv("../input/student-alcohol-consumption/student-mat.csv")
data=raw_data.copy()
raw_data.head()

#get description of data
data.describe(include='all')
# get info about variables in data
data.info()
#get number of nan values
data.isnull().sum()
corr=data.corr()
corr.style.background_gradient(cmap='coolwarm')
corr
#iters = range(len(corr.columns) - 1)
#drop_cols = []

#Iterates through Correlation Matrix Table to find correlated columns
#for i in iters:
    #for j in range(i):
        #item = corr.iloc[j:(j+1), (i+1):(i+2)]
        #col = item.columns
        #row = item.index
        #val = item.values
        #if abs(val) <= 0.1:
            #Prints the correlated feature set and the corr val
            #print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
            #drop_cols.append(i)
# check covariance of all input variables with output variable
# remove columns having less correlation with output

x=min(abs(corr.iloc[:,-1]))
x   
plt.scatter(x=data['freetime'],y=data['G3'])
import seaborn as sns
sns.distplot(data['G3'])

q_low = data["G3"].quantile(0.01)

data2 = data[(data["G3"] > q_low)]
sns.distplot(data2['G3'])
plt.hist(data2['G3'],bins=20)
#remove column 'freetime' having minimum correlation with output
data3=data2.drop('freetime',axis=1)
data3

#get numerical_variables of data 
num_data=data3.select_dtypes(include=[np.number])
num_data.head()
#get categorical_variables of data
cat_data=data3.select_dtypes(exclude=[np.number])
cat_data.head()
# check counts of classes of different variables
freq_list=[]
for i in cat_data.columns:
    freq=cat_data[i].value_counts()
    freq_list.append(freq)
freq_list    
# convert categrical variables into dummy variables
cat_data_dummies=pd.get_dummies(cat_data,drop_first=True)
cat_data_dummies
total_data=pd.concat([num_data,cat_data_dummies],axis=1,join='inner')
total_data
# seperate data as features and targets 
x=total_data.drop(['G3'],axis=1)
y=total_data['G3']
# standardization of features
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_scaled=sc.fit_transform(x)
# split data into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.2,random_state=0)
# Simple Linear Regression
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred_lin=reg.predict(x_test)
plt.scatter(y_test,y_pred_lin)
from sklearn.metrics import r2_score
r2_score(y_test, y_pred_lin)
#polynomial regression
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(x_train)
regressor = LinearRegression()
regressor.fit(X_poly, y_train)
poly_reg = PolynomialFeatures(degree=4)
X_poly_test = poly_reg.fit_transform(x_test)
y_pred=regressor.predict(X_poly_test)
plt.scatter(y_test,y_pred)
regressor.score(X_poly_test,y_test)
# support vector regression
from sklearn.svm import SVR
regressor_svr = SVR(kernel = 'rbf')
regressor_svr.fit(x_train, y_train)
y_pred_svr = regressor_svr.predict(x_test)
plt.scatter(y_test,y_pred_svr)
from sklearn.metrics import r2_score
r2_score(y_test, y_pred_svr)
# decision tree regression
x_train_dt,x_test_dt,y_train_dt,y_test_dt=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.tree import DecisionTreeRegressor
regressor_dt = DecisionTreeRegressor(random_state = 0)
regressor_dt.fit(x_train_dt, y_train_dt)
y_pred_dt = regressor_dt.predict(x_test_dt)
plt.scatter(y_test_dt,y_pred_dt)
from sklearn.metrics import r2_score
r2_score(y_test_dt, y_pred_dt)
# random forest regression
from sklearn.ensemble import RandomForestRegressor
regressor_rf = RandomForestRegressor(n_estimators = 25, random_state = 0)
regressor_rf.fit(x_train_dt, y_train_dt)
y_pred_rf = regressor_rf.predict(x_test_dt)
plt.scatter(y_test_dt,y_pred_rf)
from sklearn.metrics import r2_score
r2_score(y_test_dt, y_pred_rf)







