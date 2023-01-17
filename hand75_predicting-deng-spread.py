import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



import seaborn as sns

import missingno as msno

import statsmodels.api as sm

from sklearn.metrics import r2_score

from sklearn.metrics import mean_absolute_error

from sklearn import linear_model as lm

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score 

from datetime import datetime, timedelta

import os

print(os.listdir("../input"))
features_train=pd.read_csv('../input/DengAI_Predicting_Disease_Spread_-_Training_Data_Features.csv')
labels_train=pd.read_csv('../input/DengAI_Predicting_Disease_Spread_-_Training_Data_Labels.csv')
df=pd.merge(features_train, labels_train, on=["city","year","weekofyear"])
df.head()
df.shape
df.isna().sum()
df.describe()
# Baseline error for predictions



baseline_error=round(sum(abs(df["total_cases"]-df["total_cases"].mean()))/len(df["total_cases"]))

print("Limit basic error on number of disease case : ", baseline_error)
# Drop all NaN



df=df.dropna()
df.isna().sum()
df.head()
df.tail()
df.shape
df.describe()
# Baseline error for predictions



baseline_error=round(sum(abs(df["total_cases"]-df["total_cases"].mean()))/len(df["total_cases"]))

print("Limit basic error on number of disease case : ", baseline_error)
# Correlation



corr = df.iloc[:,-21:].corr()

corr
f, ax = plt.subplots(figsize=(18, 10))

sns.heatmap(corr, annot=True, ax=ax)
# Visualisation
f, ax = plt.subplots(figsize=(14, 9))

sns.relplot(x="year", y="total_cases", kind="line", data=df, ax=ax)
sns.relplot(x="weekofyear", y="total_cases", kind="line", data=df)
f, ax = plt.subplots(figsize=(14, 9))

sns.relplot(x="year", y="total_cases", hue="city", kind="line", data=df, ax=ax)
f, ax = plt.subplots(figsize=(14, 7))

sns.relplot(x="year", y="total_cases", hue="city", kind="line", data=df[df['city'].str.match('sj')], ax=ax)
sns.relplot(x="year", y="total_cases", hue="city", kind="line", data=df[df['city'].str.match('iq')])
sns.relplot(x="year", y="precipitation_amt_mm", hue="city", kind="line", data=df)
sns.relplot(x="year", y="reanalysis_air_temp_k", hue="city", kind="line", data=df)
# Features and target 
X=df.iloc[:,4:-1]

y=df.iloc[:,-1]
X.head()
X.tail()
y.head()
y.tail()
# Split dataset X nn train and test set
from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split( X, y, test_size=0.3, random_state=42)
# Normalization
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

train_X = sc.fit_transform(train_X) 

test_X = sc.fit_transform(test_X) 
train_X
test_X
# Modelization
# Linear Regression
reg_l = lm.LinearRegression()

reg_l_fit = reg_l.fit(train_X, train_y)

reg_l_pred=reg_l_fit.predict(test_X)

MAE_l=mean_absolute_error(test_y, reg_l_pred)

print ("MAE :", MAE_l)
# Ridge : alpha = 0.01
ridge001 = lm.Ridge(alpha = 0.01)

ridge001_fit = ridge001.fit(train_X, train_y)

ridge001_pred=ridge001_fit.predict(test_X)

MAE_r001=mean_absolute_error(test_y, ridge001_pred)

print ("MAE :", MAE_r001)
# Ridge : alpha = 100
ridge1OO = lm.Ridge(alpha = 100)

ridge100_fit = ridge1OO.fit(train_X, train_y)

ridge100_pred=ridge100_fit.predict(test_X)

MAE_r1OO=mean_absolute_error(test_y, ridge100_pred)

print ("MAE :", MAE_r1OO)
ridge1OO.coef_
# Lasso : alpha = 0.01
from sklearn.linear_model import Lasso

lassoreg_001 = Lasso(alpha=0.01)

lasso_fit_001=lassoreg_001.fit(train_X, train_y)

lasso_pred_001 = lasso_fit_001.predict(test_X)

MAE_ls001=mean_absolute_error(test_y, lasso_pred_001)

print ("MAE :", MAE_ls001)
lassoreg_001.coef_
# Lasso : alpha = 0.1
from sklearn.linear_model import Lasso

lassoreg_01 = Lasso(alpha=0.1)

lasso_fit_01=lassoreg_01.fit(train_X, train_y)

lasso_pred_01 = lasso_fit_01.predict(test_X)

MAE_ls01=mean_absolute_error(test_y, lasso_pred_01)

print ("MAE :", MAE_ls01)
lassoreg_01.coef_
coef_lasso01 = pd.DataFrame()

coef_lasso01['features'] = X.iloc[:,:].columns

coef_lasso01['coef'] = lassoreg_01.coef_

coef_lasso01['coef_abs'] = np.abs(lassoreg_01.coef_)

coef_lasso01
coef_lasso01 = coef_lasso01.sort_values(by = 'coef_abs', ascending= False)

fig, ax = plt.subplots(figsize=(12,7))

h=sns.barplot(x = "coef_abs", y = "features", data = coef_lasso01, ax=ax)

h.set_title("Coef Lasso")

plt.show()

plt.savefig('coef_lasso.png')
# Random Forest Regressor
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)

rf.fit(train_X, train_y)

rf_pred = rf.predict(test_X)

MAE_rf=mean_absolute_error(test_y, rf_pred)

print ("MAE :", MAE_rf)
# Random Forest Regressor Hyperprameters
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
from sklearn.model_selection import GridSearchCV

grid_param = {  

    'n_estimators': [10, 20, 30, 60, 100],

    'max_depth': range(3,7),

    'bootstrap': [True, False]

}
gsrf = GridSearchCV(estimator=rf,     

                     param_grid=grid_param,    

                     scoring='neg_mean_absolute_error',       

                     cv=5,  

                     verbose=0,

                     n_jobs=-1) 
gsrf_fit=gsrf.fit(train_X, train_y)
gsrf_fit
best_parameters_rf = gsrf.best_params_  

print("Best parameters :",best_parameters_rf) 
gsrf_pred=gsrf_fit.predict(test_X)

MAE_gsrf=mean_absolute_error(test_y, gsrf_pred)

print ("MAE :", MAE_gsrf)
# Support Vector Regressor SVR
from sklearn.svm import SVR

model_svr=SVR(gamma='scale', C=1.0, epsilon=0.2)

svr_fit=model_svr.fit(train_X, train_y)

svr_pred=svr_fit.predict(test_X)

MAE_svr=mean_absolute_error(test_y, svr_pred)

print ("MAE :", MAE_svr)
# Support Vector Regressor SVR Hyperprameters
grid_param_svr = {

    'kernel':["linear","rbf"], 

    'C':np.logspace(np.log10(0.001), np.log10(200), num=20), 

    'gamma':np.logspace(np.log10(0.00001), np.log10(2), num=30),

}

svr =SVR()

gsvr = GridSearchCV(svr, grid_param_svr, n_jobs=8, verbose=2)
gsvr_fit=gsvr.fit(train_X, train_y)
gsvr_fit
best_parameters_svr = gsvr.best_params_ 

print("Best parameters :",best_parameters_svr)  
svr_pred=gsvr_fit.predict(test_X)

MAE_gsvr=mean_absolute_error(test_y, svr_pred)

print ("MAE :", MAE_gsvr)
# Performance metric : MAE
mae_score=pd.DataFrame(columns=["Type_reg","Algo_name","Algo_ref","MAE_Score"])

mae_score["Type_reg"]=["Reg Linear", "Reg Linear", "Reg Linear","Reg Linear","Reg Linear", "Reg Linear","Reg Linear", "Reg Linear", "Reg Linear" ]

mae_score["Algo_ref"]=["MAE_l", "MAE_r001", "MAE_r1OO","MAE_ls001", "MAE_ls01", "MAE_rf","MAE_gsrf", "MAE_svr", "MAE_gsvr" ]

mae_score["Algo_name"]=["sklearn", "Ridge_001", "Ridge_100", "lasso001", "lasso01","R_Forest_Reg","R_Forest Reg Hyp", "SVR", "SVR Hyp" ]

mae_score["MAE_Score"]=MAE_l, MAE_r001, MAE_r1OO, MAE_ls001, MAE_ls01, MAE_rf, MAE_gsrf, MAE_svr, MAE_gsvr
mae_score
mae_score.describe()
fig, ax = plt.subplots(figsize=(12,7))

h=sns.barplot(x = "MAE_Score", y = "Algo_name", data = mae_score, ax=ax)

h.set_title("Performance metric")

plt.show()
# Prediction
data=pd.read_csv('../input/DengAI_Predicting_Disease_Spread_-_Test_Data_Features.csv')
data.head()
data.shape
test_X=data.iloc[:,4:]
test_X.head()
test_X=test_X.fillna(0)
test_X.shape
test_X.isna().sum()
# Normalization
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

test_X = sc.fit_transform(test_X) 
test_X
predict = gsvr_fit.predict(test_X)
# Submission_Deng
type(predict)
len(predict)
predict
y = np.array(np.round(predict), dtype=int)
y
pred_y = pd.DataFrame(y, columns=["total_cases"])
pred_y.head()
Submission_Deng_AI = pd.DataFrame()

Submission_Deng_AI["city"] = data["city"]

Submission_Deng_AI["year"]=data["year"]

Submission_Deng_AI["weekofyear"]=data["weekofyear"]

Submission_Deng_AI["total_cases"] =pred_y
Submission_Deng_AI
# Create csv file
Submission_Deng_AI.to_csv("Submission_Deng_AI.csv", index=False)