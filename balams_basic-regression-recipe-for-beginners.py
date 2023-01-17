#Basic Libraries
import numpy as np
import pandas as pd

#Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as mnso

#Data PreProcessing
from sklearn.preprocessing import MinMaxScaler

#Train Test Split
from sklearn.model_selection import KFold, cross_val_score, train_test_split

#Model
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor

#metrics
from sklearn.metrics import mean_squared_error
from math import sqrt

import os
print(os.listdir("../input"))
student_data = pd.read_csv('../input/StudentsPerformance.csv')
student_data.head()
student_data.shape, student_data.columns
student_data.info()
student_data.nunique()
mnso.matrix(student_data)
pd.set_option('precision', 2)
student_data['Over_all_score'] = (student_data['math score']+student_data['reading score']+student_data['writing score'])/3
student_data.head()
#Separate Categorical and Continous Variables
categ = ['gender','race/ethnicity','parental level of education','lunch','test preparation course']
conti = ['math score','reading score','writing score']
plt.figure(figsize = (10,30))
i=0
for cate in categ:
    plt.subplot(5,2,i+1)
    sns.set_style('whitegrid')
    sns.countplot(x = cate, data = student_data);
    plt.tight_layout()
    plt.xticks(rotation =90)
    i +=1
for cont in conti:
    plt.subplot(5,2,i+1)
    sns.distplot(student_data[cont])
    i+=1
plt.show()
plt.figure(figsize = (15,30))
i = 0
for cat in categ:
    plt.subplot(5,2,i+1)
    sns.set_style('whitegrid')
    sns.barplot(x = cat, y = 'Over_all_score', data = student_data, hue = 'gender');
    plt.tight_layout()
    i+=1
plt.show()
ind_var = student_data.iloc[:,0:8]
target_var = student_data['Over_all_score']
pre_data = ind_var
scale = MinMaxScaler()
pre_data[['math score','reading score','writing score']] = scale.fit_transform(pre_data[['math score','reading score','writing score']])
pre_data.head()
pre_data = pd.get_dummies(pre_data, drop_first = True)
pre_data.head()
x_train, x_test, y_train, y_test = train_test_split(pre_data, target_var, test_size = 0.20, random_state = 10)

print('x_train Shape is :', x_train.shape)
print('y_train Shape is :', y_train.shape)
print('x_test Shape is :', x_test.shape)
print('y_test Shape is :', y_test.shape)
#Model
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
models = []
models.append(['LR', LinearRegression()])
models.append(['Lasso', Lasso()])
models.append(['tree', DecisionTreeRegressor()])
models.append(['knn', KNeighborsRegressor()])
models.append(['GBM', GradientBoostingRegressor()])
models.append(['ada', AdaBoostRegressor()])
results = []
names =[]

for name, model in models:    
    kfold = KFold(n_splits = 10, random_state = 7)
    cv_result = cross_val_score(model, x_train, y_train, cv =kfold, scoring = 'r2')
    results.append(cv_result)
    names.append(name)
    msg = '%s: %f (%f)' % (name, cv_result.mean(), cv_result.std())
    print(msg)

plt.figure(figsize = (10,5))
sns.boxplot(x = names, y = results)
plt.show()
lr = LinearRegression()
lr.fit(x_train, y_train)
pred = lr.predict(x_test)
rms = sqrt(mean_squared_error(y_test, pred))
print("Root Mean Squre Error is : %.20f" % rms)
plt.figure(figsize = (7,7))
sns.regplot(y_test, pred)
plt.show()