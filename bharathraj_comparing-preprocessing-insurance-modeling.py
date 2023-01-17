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
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
df = pd.read_csv('../input/insurance.csv')
x = df.copy()
print(df.columns)
#Checking for null values

l = list(x.isnull().sum())
print(x.columns)
#Data pre- processing process for Type 1 Data

temp = pd.get_dummies(x['sex'],prefix = 'sex')
del x['sex']
x = pd.concat([x,temp], axis = 1)

temp = pd.get_dummies(x['smoker'],prefix = 'smoker')
del x['smoker']
x = pd.concat([x,temp], axis = 1)

temp = pd.get_dummies(x['region'],prefix = 'region')
del x['region']
x = pd.concat([x,temp], axis = 1)

t = x['charges'].copy()
del x['charges']

#Deleting the columns which having 95% of zeros which will not effect the model accuracy

v = x.copy()
var = list(v.columns)

for i in range(0,12):
    count = 0
    for j in range(0,1339):
        l = list(x[var[i]])
        count = l.count(0)
    if(count > 1275 or (j - count) > 63):
        del v[var[i]]
        break
# checking correlation for data preprocessed using get_dummies method  for Type 1 Data

corr = v.corr()
sm.graphics.plot_corr(corr)
plt.show()
#Removing highly correlated variable from the dataset for Type 1 Data

X = v.copy()
var2 = list(X.columns)
thresh = 5
for i in np.arange(0,len(var2)):
    vif = [variance_inflation_factor(X[var2].values, ix) for ix in range(x[var2].shape[1])]
    maxloc = vif.index(max(vif))
    if max(vif) > thresh:
        x1 = var2[maxloc]
        del X[x1]
        var2 = list(X.columns)
    else:
        break
        
# checking correlation for Type 1 Data after removing unwanted columns
corr = X.corr()
sm.graphics.plot_corr(corr)
plt.show()
print(var2)
#Data pre- processing process for Type 2 Data using Labelencoder

f = df.copy()
del f['charges']
lb = LabelEncoder()
f['sex'] = lb.fit_transform(f['sex'])
f['smoker'] = lb.fit_transform(f['smoker'])
f['region'] = lb.fit_transform(f['region'])
print(f.columns)
#checking corrleation for data pre-processed using Type 2 Data Label encoder

corr = f.corr()
sm.graphics.plot_corr(corr)
plt.show()
#Splitting data in to Train and test for modeling purpose using type 1 data

x_train1,x_test1,y_train1,y_test1 = train_test_split(v,t, test_size = 0.2,random_state = 1)
print(np.shape(x_train1),np.shape(x_test1),np.shape(y_train1),np.shape(y_test1))
#Splitting data in to Train and test for modeling purpose using type 2 data

x_train2,x_test2,y_train2,y_test2 = train_test_split(f,t, test_size = 0.2,random_state = 1)
print(np.shape(x_train2),np.shape(x_test2),np.shape(y_train2),np.shape(y_test2))
#Model creation using type 1 data

lr1 = LinearRegression()
lr1.fit(x_train1,y_train1)

de1 = DecisionTreeRegressor()
de1.fit(x_train1,y_train1)

re1 = RandomForestRegressor(max_leaf_nodes=20,n_estimators = 100)
re1.fit(x_train1,y_train1)

sr1 = SVR(kernel = 'linear', C = 100000, degree = 2, gamma = 'auto')
sr1.fit(x_train1,y_train1)

#Model creation using type 2 data

lr2 = LinearRegression()
lr2.fit(x_train2,y_train2)

de2 = DecisionTreeRegressor( max_depth = 15)
de2.fit(x_train2,y_train2)

re2 = RandomForestRegressor(max_leaf_nodes=20,n_estimators = 100)
re2.fit(x_train2,y_train2)

sr2 = SVR(kernel = 'linear', C = 100000, degree = 2, gamma = 'auto')
sr2.fit(x_train2,y_train2)
print('------Linear Regression--Type 1 Data---')
print('Training score : ', r2_score(lr1.predict(x_train1),y_train1))
print('Testing score : ', r2_score(lr1.predict(x_test1),y_test1))

print('------Linear Regression--Type 2 Data---')
print('Training score : ', r2_score(lr2.predict(x_train2),y_train2))
print('Testing score : ', r2_score(lr2.predict(x_test2),y_test2))
print('------Decision Tree Regression--Type 1 Data---')
print('Training score : ', r2_score(de1.predict(x_train1),y_train1))
print('Testing score : ', r2_score(de1.predict(x_test1),y_test1))

print('------Decision Tree Regression--Type 2 Data---')
print('Training score : ', r2_score(de2.predict(x_train2),y_train2))
print('Testing score : ', r2_score(de2.predict(x_test2),y_test2))
print('------Random Forest Regression--Type 1 Data---')
print('Training score : ', r2_score(re1.predict(x_train1),y_train1))
print('Testing score : ', r2_score(re1.predict(x_test1),y_test1))

print('------Random Forest Regression--Type 2 Data---')
print('Training score : ', r2_score(re2.predict(x_train2),y_train2))
print('Testing score : ', r2_score(re2.predict(x_test2),y_test2))
print('------Support vector Regression---Type 1 Data--')
print('Training score : ', r2_score(sr1.predict(x_train1),y_train1))
print('Testing score : ', r2_score(sr1.predict(x_test1),y_test1))

print('------Support vector Regression---Type 2 Data--')
print('Training score : ', r2_score(sr2.predict(x_train2),y_train2))
print('Testing score : ', r2_score(sr2.predict(x_test2),y_test2))