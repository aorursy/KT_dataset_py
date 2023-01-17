import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



adm = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict.csv')

adm.head()
adm = adm.drop('Serial No.',axis = 1)

adm.shape
adm.info()
adm.columns
adm.columns = adm.columns.str.strip()

adm.columns = adm.columns.str.replace(' ', '_')

adm.columns
adm[['University_Rating', 'SOP', 'LOR','Research']] = adm[['University_Rating',

                                                                    'SOP','LOR','Research']].astype('category')
column1 =  ['GRE_Score', 'TOEFL_Score', 'University_Rating', 'SOP', 'LOR', 'CGPA','Research']

column2 =  ['GRE_Score','TOEFL_Score','CGPA']
adm.describe()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



plt.figure(figsize=(6,6))

sns.heatmap(adm.corr(), annot = True)
plt.figure(figsize= (10,6))

sns.pairplot(data= adm, x_vars=['GRE_Score', 'TOEFL_Score','CGPA'], 

             y_vars= 'Chance_of_Admit',kind = 'reg',aspect=0.9, height=5)
j=0

plt.figure(figsize=(12,10))



for i in ['University_Rating','SOP','LOR', 'Research']:

    j=j+1

    plt.subplot(2,2,j)

    plt.title('variation of %s' %i)

    sns.countplot(i, data = adm)
from sklearn.linear_model import LinearRegression, Lasso, Ridge

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn import metrics



y = adm['Chance_of_Admit']

x = adm[column1]



x_train, x_test, y_train, y_test =  train_test_split(x,y, test_size = 0.2,random_state = 1)
lr = LinearRegression().fit(x_train,y_train)

lr_res  = lr.predict(x_test)



print ('train score  : %.3f' %lr.score(x_train,y_train))

print ('test score : {:.3f}'.format(lr.score(x_test,y_test)))



print ('MSE : {:.3f}'.format(metrics.mean_squared_error(y_test,lr_res)))

print ('SMSE: {:.3f}'.format(np.sqrt(metrics.mean_squared_error(y_test,lr_res))))
scores = cross_val_score(lr,x_train,y_train, cv=10)

print(scores)

print(scores.mean())
from sklearn.ensemble import RandomForestRegressor



rf = RandomForestRegressor(n_estimators=10,max_depth=3).fit(x_train, y_train)

rf_res = rf.predict(x_test)



print ('train score  : %.3f' %rf.score(x_train,y_train))

print ('test score : {:.3f}'.format(rf.score(x_test,y_test)))

print ('MSE : {:.3f}'.format(metrics.mean_squared_error(y_test,rf_res)))

print ('SMSE: {:.3f}'.format(np.sqrt(metrics.mean_squared_error(y_test,rf_res))))
rf.feature_importances_
from sklearn.tree import DecisionTreeRegressor



dt = DecisionTreeRegressor(max_depth=3).fit(x_train,y_train)

dt_res = dt.predict(x_test)



print ('train score  : %.3f' %dt.score(x_train,y_train))

print ('test score : {:.3f}'.format(dt.score(x_test,y_test)))

print ('MSE : {:.3f}'.format(metrics.mean_squared_error(y_test,dt_res)))

print ('SMSE: {:.3f}'.format(np.sqrt(metrics.mean_squared_error(y_test,dt_res))))