import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv("../input/graduate-admissions/Admission_Predict_Ver1.1.csv")
df.shape
df.isnull().sum()
df.head()
df.drop(['Serial No.'], axis = 1, inplace = True)
df.head()
plt.figure(figsize=(25,10))

sns.heatmap(df.corr(), annot=True, linewidth=0.5, cmap='coolwarm')
sns.pairplot(df)
x = df['Chance of Admit ']

sns.distplot(x , kde= True,rug = False, bins = 30)
x = df['GRE Score']

sns.distplot(x , kde= True,rug = False, bins = 30)
x = df['TOEFL Score']

sns.distplot(x , kde= True,rug = False, bins = 30)
x = df['CGPA']

sns.distplot(x , kde= True,rug = False, bins = 30)
sns.lineplot(x = 'GRE Score', y = 'CGPA', data = df)
sns.lineplot(x = 'TOEFL Score', y = 'CGPA', data = df)
sns.jointplot(x = 'GRE Score', y = 'CGPA', data=df)
sns.jointplot(x = 'TOEFL Score', y = 'CGPA', data=df)
sns.jointplot(x = 'TOEFL Score', y = 'University Rating', data=df)
sns.jointplot(x = 'GRE Score', y = 'University Rating', data=df)
sns.relplot(x ='SOP', y ='Chance of Admit ', col = 'University Rating', data = df, estimator = None,palette = 'ch:r = -0.8, l = 0.95')
sns.relplot(x ='LOR ', y ='Chance of Admit ', col = 'University Rating', data = df, estimator = None,palette = 'ch:r = -0.8, l = 0.95')
sns.relplot(x ='Research', y ='Chance of Admit ', col = 'University Rating', data = df, estimator = None,palette = 'ch:r = -0.8, l = 0.95')
from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn import metrics

from sklearn import linear_model

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet

from sklearn.linear_model import Ridge

from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error





from sklearn_pandas import DataFrameMapper

from numpy import asarray

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler
X = df.drop(['Chance of Admit '], axis = 1)

X
y = df['Chance of Admit ']

y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,shuffle = False)
model1 = LinearRegression()

model1.fit(X_train, y_train)



accuracy1 = model1.score(X_test,y_test)

print(accuracy1*100,'%')
y_pred1 = model1.predict(X_test)



val = mean_squared_error(y_test, y_pred1, squared=False)

val1 = str(round(val, 4))



print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred1)))

model2 = DecisionTreeRegressor()

model2.fit(X_train, y_train)



accuracy2 = model2.score(X_test,y_test)

print(accuracy2*100,'%')
y_pred2 = model2.predict(X_test)



val = mean_squared_error(y_test, y_pred2, squared=False)

val2 = str(round(val, 4))





print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred2)))

n_estimators = [10, 40, 70, 100, 130, 160, 190, 220, 250, 270]



RF = RandomForestRegressor()



parameters = {'n_estimators': [10, 40, 70, 100, 130, 160, 190, 220, 250, 270]}



RFR = GridSearchCV(RF, parameters,scoring='neg_mean_squared_error', cv=5)



RFR.fit(X_train, y_train)



RFR.best_params_

model3 = RandomForestRegressor(n_estimators = 190)

model3.fit(X_train, y_train)



accuracy3 = model3.score(X_test,y_test)

print(accuracy3*100,'%')
y_pred3 = model3.predict(X_test)



val = mean_squared_error(y_test, y_pred3, squared=False)

val3 = str(round(val, 4))





print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred3)))

lasso = Lasso()



parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}



lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv = 100)



lasso_regressor.fit(X_train, y_train)



lasso_regressor.best_params_
model4 = linear_model.Lasso(alpha=.001)

model4.fit(X_train,y_train)



accuracy4 = model4.score(X_test,y_test)

print(accuracy4*100,'%')
y_pred4 = model4.predict(X_test)



val= mean_squared_error(y_test, y_pred4, squared=False)

val4 = str(round(val, 4))





print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred4)))

alpha = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]



ridge = Ridge()



parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}



ridge_regressor = GridSearchCV(ridge, parameters,scoring='neg_mean_squared_error', cv=100)



ridge_regressor.fit(X_train, y_train)

ridge_regressor.best_params_
model5 = linear_model.Ridge(alpha=1)

model5.fit(X_train,y_train)



accuracy5 = model5.score(X_test,y_test)

print(accuracy5*100,'%')
y_pred5 = model5.predict(X_test)



val = mean_squared_error(y_test, y_pred5, squared=False)

val5 = str(round(val, 4))





print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred5)))

Elasticnet = ElasticNet()



parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}



en_regressor = GridSearchCV(Elasticnet, parameters, scoring='neg_mean_squared_error', cv = 100)



en_regressor.fit(X_train, y_train)

en_regressor.best_params_
model6 = linear_model.ElasticNet(alpha=0.001)

model6.fit(X_train,y_train)



accuracy6 = model6.score(X_test,y_test)

print(accuracy6*100,'%')
y_pred6 = model6.predict(X_test)



val = mean_squared_error(y_test, y_pred6, squared=False)

val6 = str(round(val, 4))





print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred6)))

data1 = [['Linear Regression ',val1],['Decision Tree',val2],['Random Forest',val3],['Lasso Regression',val4],['Ridge Regression',val5],['ElasticNet Regression',val6]]

d1 = pd.DataFrame(data1,columns = ['Without Scaling Models ','RMSE Error'])

Half1RMSE = d1.copy()

Half1RMSE

mapper = DataFrameMapper([(df.columns, StandardScaler())])

scaled_features = mapper.fit_transform(df.copy(), 4)

data = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)
data.head()
x = data.drop(['Chance of Admit '], axis = 1)

x
Y = data['Chance of Admit ']

Y
x_train, x_test, Y_train, Y_test = train_test_split(x, Y, test_size = 0.20,shuffle = False)
model7 = LinearRegression()

model7.fit(x_train, Y_train)



accuracy7 = model7.score(x_test,Y_test)

print(accuracy7*100,'%')
y_pred7 = model7.predict(x_test)



val = mean_squared_error(Y_test, y_pred7, squared=False)

val7 = str(round(val, 4))





print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred7)))

model8 = DecisionTreeRegressor()

model8.fit(x_train, Y_train)



accuracy8 = model8.score(x_test,Y_test)

print(accuracy8*100,'%')
y_pred8 = model8.predict(x_test)



val = mean_squared_error(Y_test, y_pred8, squared=False)

val8 = str(round(val, 4))



print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred8)))

n_estimators = [10, 40, 70, 100, 130, 160, 190, 220, 250, 270]



rf = RandomForestRegressor()



parameters = {'n_estimators': [10, 40, 70, 100, 130, 160, 190, 220, 250, 270]}



rfr = GridSearchCV(rf, parameters,scoring='neg_mean_squared_error', cv=10)



rfr.fit(x_train, Y_train)



rfr.best_params_
model9 = RandomForestRegressor(n_estimators = 220)

model9.fit(x_train, Y_train)



accuracy9 = model9.score(x_test,Y_test)

print(accuracy9*100,'%')
y_pred9 = model9.predict(x_test)



val = mean_squared_error(Y_test, y_pred9, squared=False)

val9 = str(round(val, 4))



print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred9)))

L = Lasso()



parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}



LR = GridSearchCV(L, parameters, scoring='neg_mean_squared_error', cv = 100)



LR.fit(x_train, Y_train)

LR.best_params_
model10 = linear_model.Lasso(alpha=.01)

model10.fit(x_train,Y_train)



accuracy10 = model10.score(x_test,Y_test)

print(accuracy10*100,'%')
y_pred10 = model10.predict(x_test)



val = mean_squared_error(Y_test, y_pred10, squared=False)

val10 = str(round(val, 4))



print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred10)))

EN = ElasticNet()



parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}



ENR = GridSearchCV(Elasticnet, parameters, scoring='neg_mean_squared_error', cv = 100)



ENR.fit(x_train, Y_train)

ENR.best_params_
model11 = linear_model.Lasso(alpha=.01)

model11.fit(x_train,Y_train)



accuracy11 = model11.score(x_test,Y_test)

print(accuracy11*100,'%')
y_pred11 = model11.predict(x_test)



val = mean_squared_error(Y_test, y_pred11, squared=False)

val11 = str(round(val, 4))





print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred11)))

SVR = SVR()



parameters = {'C':[.0001 ,.001 ,0.1, 1, 10, 100, 1000],

              'epsilon':[0.001, 0.01, 0.1, 0.5, 1, 2, 4]

             }



ENR = GridSearchCV(SVR, parameters, scoring='neg_mean_squared_error', cv = 10)



ENR.fit(x_train, Y_train)

ENR.best_params_
from sklearn.svm import SVR

model12 = SVR(C=1, epsilon=0.1)

model12.fit(x_train,Y_train)



model12 = model12.score(x_test,Y_test)

print(model12*100,'%')
R = Ridge()



parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}



R = GridSearchCV(R, parameters, scoring='neg_mean_squared_error', cv = 100)



R.fit(x_train, Y_train)

R.best_params_
model13 = linear_model.Ridge(alpha=10)

model13.fit(x_train,Y_train)



accuracy13 = model13.score(x_test,Y_test)

print(accuracy13*100,'%')
y_pred13 = model13.predict(x_test)



val = mean_squared_error(Y_test, y_pred13, squared=False)

val13 = str(round(val, 4))



print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred13)))

data2 = [['Scaled Linear Regression',val7],['Scaled Decision Tree',val8],['Scaled Random Forest',val9],['Scaled Lasso Regression',val10],['Scaled Ridge Regression',val13],['Scaled ElasticNet Regression',val11]]

d2 = pd.DataFrame(data2,columns = ['Standard Scaler - Model ','RMSE Error'])

Half2RMSE = d2.copy()

Half2RMSE
from pandas import DataFrame

trans = MinMaxScaler()

dat = trans.fit_transform(df)

dataset = DataFrame(dat)



df.head()
dataset.columns = ['GRE Score','TOEFL Score','University Rating','SOP','LOR','CGPA','Research','Chance of Admit']

ex = dataset.drop(['Chance of Admit'], axis = 1)

ex
ey = dataset['Chance of Admit']

ey
x_t, x_es, Y_t, Y_es = train_test_split(ex, ey, test_size = 0.20,shuffle = False)
model14 = LinearRegression()

model14.fit(x_t, Y_t)



accuracy14 = model14.score(x_es,Y_es)

print(accuracy14*100,'%')
y_pred14 = model14.predict(x_es)



val = mean_squared_error(Y_es, y_pred14, squared=False)

val14 = str(round(val, 4))





print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_es, y_pred14)))

l = Lasso()



parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}



lr = GridSearchCV(l, parameters, scoring='neg_mean_squared_error', cv = 100)



lr.fit(x_t, Y_t)

lr.best_params_
model15 = linear_model.Lasso(alpha=.001)

model15.fit(x_t,Y_t)



accuracy15 = model15.score(x_es,Y_es)

print(accuracy15*100,'%')
y_pred15 = model15.predict(x_es)



val = mean_squared_error(Y_es, y_pred15, squared=False)

val15 = str(round(val, 4))



print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_es, y_pred15)))

r = Ridge()



parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}



r = GridSearchCV(r, parameters, scoring='neg_mean_squared_error', cv = 100)



r.fit(x_t, Y_t)

r.best_params_
model16 = linear_model.Ridge(alpha=0.01)

model16.fit(x_t,Y_t)



accuracy16 = model16.score(x_es,Y_es)

print(accuracy16*100,'%')
y_pred16 = model16.predict(x_es)



val = mean_squared_error(Y_es, y_pred16, squared=False)

val16 = str(round(val, 4))





print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_es, y_pred16)))

en = ElasticNet()



parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}



enr = GridSearchCV(en, parameters, scoring='neg_mean_squared_error', cv = 100)



enr.fit(x_t, Y_t)

enr.best_params_
model17 = linear_model.Lasso(alpha=.001)

model17.fit(x_t,Y_t)



accuracy17 = model17.score(x_es,Y_es)

print(accuracy17*100,'%')
y_pred17 = model17.predict(x_es)



val = mean_squared_error(Y_es, y_pred17, squared=False)

val17 = str(round(val, 4))





print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_es, y_pred17)))

model18 = DecisionTreeRegressor()

model18.fit(x_t, Y_t)



accuracy18 = model18.score(x_es,Y_es)

print(accuracy18*100,'%')
y_pred18 = model18.predict(x_es)



val = mean_squared_error(Y_es, y_pred8, squared=False)

val18 = str(round(val, 4))



print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_es, y_pred18)))

n_estimators = [10, 40, 70, 100, 130, 160, 190, 220, 250, 270]



Rf = RandomForestRegressor()



parameters = {'n_estimators': [10, 40, 70, 100, 130, 160, 190, 220, 250, 270]}



Rfr = GridSearchCV(Rf, parameters,scoring='neg_mean_squared_error', cv=10)



Rfr.fit(x_t, Y_t)



Rfr.best_params_
model19 = RandomForestRegressor(n_estimators = 100)

model19.fit(x_t, Y_t)



accuracy19 = model19.score(x_es,Y_es)

print(accuracy19*100,'%')
y_pred19 = model19.predict(x_es)



val = mean_squared_error(Y_es, y_pred19, squared=False)

val19 = str(round(val, 4))



print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_es, y_pred19)))

from sklearn.svm import SVR



Svr = SVR()



parameters = {'C':[.0001 ,.001 ,0.1, 1, 10, 100, 1000],

              'epsilon':[0.001, 0.01, 0.1, 0.5, 1, 2, 4]

             }



Enr = GridSearchCV(Svr, parameters, scoring='neg_mean_squared_error', cv = 10)



Enr.fit(x_t, Y_t)

Enr.best_params_
from sklearn.svm import SVR

model20 = SVR(C=1, epsilon=0.1)

model20.fit(x_t,Y_t)



model20 = model20.score(x_es,Y_es)

print(model20*100,'%')
data3 = [['Scaled Linear Regression',val14],['Scaled Lasso Regression',val15],['Scaled Ridge Regression',val16],['Scaled ElasticNet Regression',val17],['Scaled Decision Tree',val18],['Scaled Random Forest',val19]]

d3 = pd.DataFrame(data3,columns = ['Min Max Scaler - Model ','RMSE Error'])

Half3RMSE = d3.copy()

Half3RMSE
frames = [Half1RMSE,Half2RMSE,Half3RMSE] 

FullRMSE = pd.concat(frames, axis = 1)
FullRMSE