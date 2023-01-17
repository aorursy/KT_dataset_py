import os

import pandas as pd



data = pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv')

data.info()
data.describe().round(2)
data.isnull().sum()
print(data.columns)
data.columns = (['serial_no','GRE','TOEFL','university_rating','SOP','LOR','CGPA','research','COA'])

data = data.drop('serial_no',axis=1)

data.head()
import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style='darkgrid')



plt.figure(figsize=(20,20))



plt.subplot(4,2,1)

sns.distplot(data['GRE'])

plt.subplot(4,2,2)

sns.distplot(data['TOEFL'])

plt.subplot(4,2,3)

sns.distplot(data['university_rating'], kde=False)

plt.subplot(4,2,4)

sns.distplot(data['SOP'], kde=False)

plt.subplot(4,2,5)

sns.distplot(data['LOR'], kde=False)

plt.subplot(4,2,6)

sns.distplot(data['CGPA'])

plt.subplot(4,2,7)

sns.distplot(data['research'], kde=False)

plt.subplot(4,2,8)

sns.distplot(data['COA'])

plt.show()
con_feat = ['GRE','TOEFL','CGPA', 'COA']

ax = sns.pairplot(data[con_feat])
correlation_matrix = data.corr()

f, x = plt.subplots(figsize=(10,10))

sns.heatmap(correlation_matrix, vmax=1, annot=True)
from statsmodels.stats.outliers_influence import variance_inflation_factor

features = ['GRE','TOEFL','university_rating','SOP','LOR','CGPA','research']

vif = pd.DataFrame()

vif['Features'] = data[features].columns

vif['VIF'] = [variance_inflation_factor(data[features].values, i) for i in range(data[features].shape[1])] 

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = 'VIF', ascending = False)

vif
import numpy as np

from sklearn.model_selection import train_test_split



np.random.seed(0)

data_train, data_test = train_test_split(data, train_size=.8, test_size=.2, random_state=100)

data_train.head()
X_train = data_train[features] 

y_train = data_train['COA']

X_test = data_test[features] 

y_test = data_test['COA']
from sklearn import linear_model

from sklearn import metrics

from sklearn.model_selection import cross_val_score



evaluation = pd.DataFrame({'Model': [],

                          'RMSE': [],

                          '5-Fold Cross Validation': []})
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR



model1 = LinearRegression()

model1.fit(X_train,y_train)



predict1 = model1.predict(X_test)

rmse1 = float(format(np.sqrt(metrics.mean_squared_error(y_test,predict1)),

                   '.3f'))

cv1 = float(format(cross_val_score(model1, data[features], data['COA'],

                                 cv=5).mean(), '.3f'))



model2 = RandomForestRegressor()

model2.fit(X_train,y_train)



predict2 = model2.predict(X_test)

rmse2 = float(format(np.sqrt(metrics.mean_squared_error(y_test,predict2)),

                   '.3f'))

cv2 = float(format(cross_val_score(model2, data[features], data['COA'],

                                 cv=5).mean(), '.3f'))



model3 = RandomForestRegressor()

model3.fit(X_train,y_train)



predict3 = model3.predict(X_test)

rmse3 = float(format(np.sqrt(metrics.mean_squared_error(y_test,predict3)),

                   '.3f'))

cv3 = float(format(cross_val_score(model3, data[features], data['COA'],

                                 cv=5).mean(), '.3f'))



model4 = RandomForestRegressor()

model4.fit(X_train,y_train)



predict4 = model4.predict(X_test)

rmse4 = float(format(np.sqrt(metrics.mean_squared_error(y_test,predict4)),

                   '.3f'))

cv4 = float(format(cross_val_score(model4, data[features], data['COA'],

                                 cv=5).mean(), '.3f'))



r = evaluation.shape[0]

evaluation.loc[r] = ['Multiple Regression', rmse1, cv1]

evaluation.loc[r+1] = ['Random Forest', rmse2, cv2]

evaluation.loc[r+2] = ['K-Neighbours', rmse3, cv3]

evaluation.loc[r+3] = ['SVR', rmse4, cv4]

evaluation.sort_values(by = '5-Fold Cross Validation', ascending = False)