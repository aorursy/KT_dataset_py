import numpy as np

import pandas as pd

import seaborn as sns

import random

import scipy.stats as stt

import warnings

import sklearn

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_curve,roc_auc_score

from sklearn.metrics import classification_report

from sklearn.model_selection import cross_val_score

from sklearn.metrics import r2_score,mean_absolute_error

from sklearn import linear_model

import statsmodels.regression.linear_model as sm

from sklearn.model_selection import GridSearchCV

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt 

%matplotlib inline

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score,mean_absolute_error

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor
raw = pd.read_csv("../input/Admission_Predict.csv", index_col = 'Serial No.')
raw.head(10)
raw.shape
raw.describe()
raw.info()
raw.isnull().sum()
sns.pairplot(raw[['GRE Score', 'TOEFL Score','CGPA','SOP','University Rating','Chance of Admit ']])
plt.figure(figsize=(20,8))

sns.heatmap(raw.corr(),annot=True)
target = raw.iloc[:,-1:]

inputs = raw.iloc[:,:-1]



X_train,X_test,Y_train,Y_test = train_test_split(inputs,target,test_size=0.2,random_state=0)





X_train = np.append(arr = np.ones((320,1)).astype(int), values = X_train, axis = 1)

X_train_opt = X_train[:,[0,1,2,3,4,5,6,7]]

regressor_OLS = sm.OLS(endog = Y_train, exog = X_train_opt).fit()

regressor_OLS.summary()
X_train_opt = X_train[:,[0,1,2,3,5,6,7]]

regressor_OLS = sm.OLS(endog = Y_train, exog = X_train_opt).fit()

regressor_OLS.summary()
X_train_opt = X_train[:,[0,1,2,5,6,7]]

regressor_OLS = sm.OLS(endog = Y_train, exog = X_train_opt).fit()

regressor_OLS.summary()
X_train_opt = X_train[:,[0,1,5,6,7]]

regressor_OLS = sm.OLS(endog = Y_train, exog = X_train_opt).fit()

regressor_OLS.summary()
X_train = X_train[:,1:]

X_train
lr = LinearRegression()

lr.fit(X_train,Y_train)

lr_predict = lr.predict(X_test)

print("R2 score : {}".format(r2_score(Y_test, lr_predict)))

print("MAE : {}".format(mean_absolute_error(Y_test, lr_predict)))
pl = PolynomialFeatures(degree=2)

X_train_poly = pl.fit_transform(X_train)

lr_poly = LinearRegression()

lr_poly.fit(X_train_poly, Y_train)

lr_poly_predict = lr_poly.predict(pl.fit_transform(X_test))

print("R2 score : {}".format(r2_score(Y_test, lr_poly_predict)))

print("MAE : {}".format(mean_absolute_error(Y_test, lr_poly_predict)))
sv = SVR(kernel = 'rbf')

sv.fit(X_train, Y_train)

sv_predict = sv.predict(X_test)

print("R2 score : {}".format(r2_score(Y_test, sv_predict)))

print("MAE : {}".format(mean_absolute_error(Y_test, sv_predict)))
sv_linear = SVR(kernel = 'linear')

sv_linear.fit(X_train, Y_train)

sv_linear_predict = sv_linear.predict(X_test)

print("R2 score : {}".format(r2_score(Y_test, sv_linear_predict)))

print("MAE : {}".format(mean_absolute_error(Y_test, sv_linear_predict)))
dc = DecisionTreeRegressor()

dc.fit(X_train,Y_train)

dc_predict = dc.predict(X_test)

print("R2 score : {}".format(r2_score(Y_test, dc_predict)))

print("MAE : {}".format(mean_absolute_error(Y_test, dc_predict)))
rf = RandomForestRegressor()

rf.fit(X_train,Y_train)

rf_predict = rf.predict(X_test)

print("R2 score : {}".format(r2_score(Y_test, rf_predict)))

print("MAE : {}".format(mean_absolute_error(Y_test, rf_predict)))
lr_predict
lr_predict_df = pd.DataFrame(lr_predict, columns =['Chance of Admit'])

lr_predict_df
f,ax =plt.subplots(1,2, figsize =(18,8))

sns.scatterplot(x=Y_test.index, y=Y_test['Chance of Admit '], data=Y_test, ax=ax[0])

sns.scatterplot(x=lr_predict_df.index, y=lr_predict_df['Chance of Admit'], data=lr_predict_df, ax=ax[1])