# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Reading the CSV
admission = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')
admission.head()
admission.columns
#Dropping Serial No. column as it adds no value
admission.drop('Serial No.', axis = 1, inplace = True)
admission.head()
#-----Lets do some EDA--------
#Check the correlation between the various features
plt.figure(figsize=(10,6), dpi =100)
sns.heatmap(admission.corr(), cmap="YlGnBu", annot=True)

#CGPA Seems to be most correlated with Chance of Admit

plt.figure(figsize=(100,6), dpi =100)
for col in admission.columns:
    plt.figure(figsize=(10,6), dpi =100)
    sns.jointplot(x=col, y='Chance of Admit ', data = admission, height=7, kind='scatter')

#-------------Machine learning-----------------
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

#--Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
#Scaling the dataframe
scaler = MinMaxScaler()
scaler.fit(admission.drop(['Chance of Admit '], axis = 1))
admission_scaler =scaler.transform(admission.drop(['Chance of Admit '], axis = 1))
#Train-test-split
X = admission_scaler
y = admission['Chance of Admit ']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 101)
#Linear Regression
lm = LinearRegression()
lm.fit(X_train,y_train)
predict_linear_regression = lm.predict(X_test)

#Plots
fig = plt.figure(figsize=(8,4), dpi=100)
#Distplot of the variations between actual and predicted "Chance of Admit"
sns.distplot((y_test-predict_linear_regression), bins = 165, color = 'Red')
#Figure shows variation between "actual Chance of Admit" in "red" vs "predicted Chance of Admit" in "blue"
#It shows for 35 rows
fig = plt.figure(figsize=(15,4), dpi=100)
plt.xlabel('Serial ID of Record')
plt.ylabel('Chance of Admit')
plt.scatter(np.arange(0,165,3),y_test[0:165:3],color = "black")
plt.scatter(np.arange(0,165,3),predict_linear_regression[0:165:3],color = "blue")
#Random Forest Regression
rf = RandomForestRegressor()
rf.fit(X_train,y_train)
predict_random_forest_regression = rf.predict(X_test)

#Plots
fig = plt.figure(figsize=(8,4), dpi=100)
#Distplot of the variations between actual and predicted "Chance of Admit"
sns.distplot((y_test-predict_random_forest_regression), bins = 165, color = 'Green')
#Figure shows variation between "actual Chance of Admit" in "red" vs "predicted Chance of Admit" in "blue"
#It shows for 35 rows
fig = plt.figure(figsize=(15,4), dpi=100)
plt.xlabel('Serial ID of Record')
plt.ylabel('Chance of Admit')
plt.scatter(np.arange(0,165,3),y_test[0:165:3],color = "black")
plt.scatter(np.arange(0,165,3),predict_random_forest_regression[0:165:3],color = "blue")
#XGBoost
xgb = XGBRegressor()
xgb.fit(X_train,y_train)
predict_xgb = xgb.predict(X_test)

#Plots
fig = plt.figure(figsize=(8,4), dpi=100)
#Distplot of the variations between actual and predicted "Chance of Admit"
sns.distplot((y_test-predict_xgb), bins = 165, color = 'Blue')
fig = plt.figure(figsize=(8,4), dpi=100)
#Figure shows variation between "actual Chance of Admit" in "red" vs "predicted Chance of Admit" in "blue"
#It shows for 35 rows
fig = plt.figure(figsize=(15,4), dpi=100)
plt.xlabel('Serial ID of Record')
plt.ylabel('Chance of Admit')
plt.scatter(np.arange(0,165,3),y_test[0:165:3],color = "black")
plt.scatter(np.arange(0,165,3),predict_xgb[0:165:3],color = "blue")
#Comparison of Regression models
fig,axes = plt.subplots(1,3,figsize=(15,5))
x = ['LinearReg','RandomForrReg','XGBoostReg']

#Mean Absolute Error
axes[0].set_title("Mean Absolute Error")
y_mae = np.array([mean_absolute_error(y_test,predict_linear_regression),mean_absolute_error(y_test,predict_random_forest_regression),mean_absolute_error(y_test,predict_xgb)])
axes[0].bar(x,y_mae)

#Mean Squared Error
axes[1].set_title("Mean Squared Error")
y_mse = np.array([mean_squared_error(y_test,predict_linear_regression),mean_squared_error(y_test,predict_random_forest_regression),mean_squared_error(y_test,predict_xgb)])
axes[1].bar(x,y_mse)

#Root Mean Squared Error
axes[2].set_title("Root Mean Squared Error")
y_rmse = np.array([np.sqrt(mean_squared_error(y_test,predict_linear_regression)),np.sqrt(mean_squared_error(y_test,predict_random_forest_regression)),np.sqrt(mean_squared_error(y_test,predict_xgb))])
axes[2].bar(x,y_rmse)

#Figure shows variation between "actual Chance of Admit" in "red" vs "predicted Chance of Admit" in "blue"
#It shows for 35 rows
fig = plt.figure(figsize=(15,4), dpi=100)
plt.title('Comparsion of various Regression models Chance of Admit vs Actual Chance of Admit')
plt.xlabel('Serial ID of Record')
plt.ylabel('Chance of Admit')
plt.scatter(np.arange(0,165,3),y_test[0:165:3],color = "black")
plt.scatter(np.arange(0,165,3),predict_linear_regression[0:165:3],color = "red")
plt.scatter(np.arange(0,165,3),predict_random_forest_regression[0:165:3],color = "green")
plt.scatter(np.arange(0,165,3),predict_xgb[0:165:3],color = "blue")

##############################
# By looking at the plots, its clear that LinearRegression model best suits the data set!!
##############################
