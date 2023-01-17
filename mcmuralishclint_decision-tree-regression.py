# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/carpriceprediction/data.csv')
data.head()
data.apply(lambda x: sum(x.isnull()))
data = pd.read_csv('../input/cleaning-data-microsoft-azure/Numerical Cleaned Data.csv') 
#I cleaned the missing numericial data using MICE algorithm on Microsoft Azure Machine Learning Studio
data.apply(lambda x: sum(x.isnull()))
from scipy.stats import mode
Engine_Fuel_Type  = data.pivot_table(values='Engine Fuel Type',
                                   columns='Model',
                                   aggfunc=lambda x: x.mode().iat[0])
miss_bool_Fuel_Type = data['Engine Fuel Type'].isnull() 
data.loc[miss_bool_Fuel_Type,'Engine Fuel Type'] = data.loc[miss_bool_Fuel_Type,'Model'].apply(lambda x: Engine_Fuel_Type[x])
from scipy.stats import mode
market_category  = data.pivot_table(values='Market Category',
                                   columns='Make',
                                   aggfunc=lambda x: x.mode().iat[0])
miss_bool_market = data['Market Category'].isnull() 
data.loc[miss_bool_market,'Market Category'] = data.loc[miss_bool_market,'Make'].apply(lambda x: market_category[x])
df = data['Market Category'].str.get_dummies(sep=',')
data = pd.concat([data, df], axis=1)
data=data.drop('Market Category',axis=1)
data.head()
#Split the data to x and y
y=data['MSRP']
x=data.drop('MSRP',axis=1)
set(x.Make)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x['Make'] = le.fit_transform(x['Make'])
category = []
category.append('Make')
x.Make.value_counts()
x['Model'] = le.fit_transform(x['Model'])
category.append('Model')
import matplotlib.pyplot as plt
%matplotlib inline
plt.scatter(x.Make,np.log(y+1))
y=np.log(y+1)
x.Year.min(),x.Year.max()
x['Year'] = pd.qcut(x['Year'], 5, labels=False)
category.append('Year')
x['Engine Fuel Type'].value_counts()
category.append('Engine Fuel Type')
x.loc[x['Engine Fuel Type'] == 'regular unleaded', 'Engine Fuel Type'] = 'regular'
x.loc[x['Engine Fuel Type'] == 'premium unleaded (required)', 'Engine Fuel Type'] = 'premium'
x.loc[x['Engine Fuel Type'] == 'premium unleaded (recommended)', 'Engine Fuel Type'] = 'premium'
x.loc[x['Engine Fuel Type'] == 'flex-fuel (unleaded/E85)', 'Engine Fuel Type'] = 'flex-fuel'
x.loc[x['Engine Fuel Type'] == 'flex-fuel (premium unleaded required/E85)', 'Engine Fuel Type'] = 'flex-fuel'
x.loc[x['Engine Fuel Type'] == 'flex-fuel (premium unleaded recommended/E85)', 'Engine Fuel Type'] = 'flex-fuel'
x.loc[x['Engine Fuel Type'] == 'flex-fuel (unleaded/natural gas)', 'Engine Fuel Type'] = 'flex-fuel'
x['Engine Fuel Type'] = le.fit_transform(x['Engine Fuel Type'])
x['Transmission Type'] = le.fit_transform(x['Transmission Type'])
category.append('Transmission Type')
x['Driven_Wheels'].value_counts()
x['Driven_Wheels'] = le.fit_transform(x['Driven_Wheels'])
category.append('Driven_Wheels')
x['Vehicle Size'].value_counts()
x['Vehicle Size'] = le.fit_transform(x['Vehicle Size'])
category.append('Vehicle Size')
x['Vehicle Style'].value_counts()
x['Vehicle Style'] = le.fit_transform(x['Vehicle Style'])
category.append('Vehicle Style')
x = pd.get_dummies(x, columns=category)
for count in category:
     x=x.drop(count + '_0',axis=1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33)
from sklearn.tree import DecisionTreeRegressor
dtr= DecisionTreeRegressor()
dtr.fit(x_train,y_train)
y_pred = dtr.predict(x_test)
from sklearn.metrics import mean_squared_error 
mean_squared_error(y_test, y_pred)
y_pred_train = dtr.predict(x_train)
mean_squared_error(y_train, y_pred_train)
from sklearn.linear_model import Ridge
from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

rr = Ridge(alpha=0.01) # higher the alpha value, more restriction on the coefficients; low alpha > more generalization, coefficients are barely
# restricted and in this case linear and ridge regression resembles
rr.fit(x_train, y_train)
rr100 = Ridge(alpha=100) #  comparison with alpha value
rr100.fit(x_train, y_train)
train_score=lr.score(x_train, y_train)
test_score=lr.score(x_test, y_test)
Ridge_train_score = rr.score(x_train,y_train)
Ridge_test_score = rr.score(x_test, y_test)
Ridge_train_score100 = rr100.score(x_train,y_train)
Ridge_test_score100 = rr100.score(x_test, y_test)
print ("linear regression train score:", train_score)
print ("linear regression test score:", test_score)
print ("ridge regression train score low alpha:", Ridge_train_score)
print ("ridge regression test score low alpha:", Ridge_test_score)
print ("ridge regression train score high alpha:", Ridge_train_score100)
print ("ridge regression test score high alpha:", Ridge_test_score100)
plt.plot(rr.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Ridge; $\alpha = 0.01$',zorder=7) # zorder for ordering the markers
plt.plot(rr100.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Ridge; $\alpha = 100$') # alpha here is for transparency
plt.plot(lr.coef_,alpha=0.4,linestyle='none',marker='o',markersize=7,color='green',label='Linear Regression')
plt.xlabel('Coefficient Index',fontsize=16)
plt.ylabel('Coefficient Magnitude',fontsize=16)
plt.legend(fontsize=13,loc=4)
plt.show()
from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.fit(x_train,y_train)
train_score=lasso.score(x_train,y_train)
test_score=lasso.score(x_test,y_test)
coeff_used = np.sum(lasso.coef_!=0)
print ("training score:", train_score )
print ("test score: ", test_score)
print ("number of features used: ", coeff_used)
lasso001 = Lasso(alpha=0.01, max_iter=10e5)
lasso001.fit(x_train,y_train)
train_score001=lasso001.score(x_train,y_train)
test_score001=lasso001.score(x_test,y_test)
coeff_used001 = np.sum(lasso001.coef_!=0)
print ("training score for alpha=0.01:", train_score001 )
print ("test score for alpha =0.01: ", test_score001)
print ("number of features used: for alpha =0.01:", coeff_used001)
lasso00001 = Lasso(alpha=0.0001, max_iter=10e5)
lasso00001.fit(x_train,y_train)
train_score00001=lasso00001.score(x_train,y_train)
test_score00001=lasso00001.score(x_test,y_test)
coeff_used00001 = np.sum(lasso00001.coef_!=0)
print ("training score for alpha=0.0001:", train_score00001 )
print ("test score for alpha =0.0001: ", test_score00001)
print ("number of features used: for alpha =0.0001:", coeff_used00001)
lr = LinearRegression()
lr.fit(x_train,y_train)
lr_train_score=lr.score(x_train,y_train)
lr_test_score=lr.score(x_test,y_test)
print ("LR training score:", lr_train_score )
print ("LR test score: ", lr_test_score)
plt.subplot(1,2,1)
plt.plot(lasso.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Lasso; $\alpha = 1$',zorder=7) # alpha here is for transparency
plt.plot(lasso001.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Lasso; $\alpha = 0.01$') # alpha here is for transparency

plt.xlabel('Coefficient Index',fontsize=16)
plt.ylabel('Coefficient Magnitude',fontsize=16)
plt.legend(fontsize=13,loc=4)
plt.subplot(1,2,2)
plt.plot(lasso.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Lasso; $\alpha = 1$',zorder=7) # alpha here is for transparency
plt.plot(lasso001.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Lasso; $\alpha = 0.01$') # alpha here is for transparency
plt.plot(lasso00001.coef_,alpha=0.8,linestyle='none',marker='v',markersize=6,color='black',label=r'Lasso; $\alpha = 0.00001$') # alpha here is for transparency
plt.plot(lr.coef_,alpha=0.7,linestyle='none',marker='o',markersize=5,color='green',label='Linear Regression',zorder=2)
plt.xlabel('Coefficient Index',fontsize=16)
plt.ylabel('Coefficient Magnitude',fontsize=16)
plt.legend(fontsize=13,loc=4)
plt.tight_layout()
plt.show()
#Ridge Regression Low Alpha can be chosen