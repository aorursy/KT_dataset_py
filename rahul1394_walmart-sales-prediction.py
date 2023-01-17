import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt
train = pd.read_csv('../input/train.csv')

train.head()
train.shape
test = pd.read_csv('../input/test.csv')

test.head()
macro = pd.read_excel('../input/macro_economic.xlsx')

macro.head()
macro.shape
macro['Year-Month'] = macro['Year-Month'].str.split('-')
macro['Year'] = macro['Year-Month'].str[0]

macro['Month'] = macro['Year-Month'].str[1]
macro.drop('Year-Month',axis=1,inplace=True)

macro.head()
macro['Month'].unique()
macro['Year'].unique()
mn_num = list(range(1,13))

mn_alpha = macro['Month'].unique()
mn_map = dict(zip(mn_alpha,mn_num))

mn_map
macro['Month'] = macro['Month'].map(mn_map)
macro.head()
macro.info()
train.info()
macro['Year'] = macro['Year'].astype(int)
mergeddata = pd.merge(macro,train,on=['Year','Month'],how='right')

mergeddata.head()
merged_test = pd.merge(macro,test,on=['Year','Month'],how='right')

merged_test.head()
weather_a = pd.read_excel('../input/WeatherData.xlsx',sheet_name=0)

weather_b = pd.read_excel('../input/WeatherData.xlsx',sheet_name=1)

weather_c = pd.read_excel('../input/WeatherData.xlsx',sheet_name=2)

weather_d = pd.read_excel('../input/WeatherData.xlsx',sheet_name=3)

weather_e = pd.read_excel('../input/WeatherData.xlsx',sheet_name=4)

weather_f = pd.read_excel('../input/WeatherData.xlsx',sheet_name=5)

weather_g = pd.read_excel('../input/WeatherData.xlsx',sheet_name=6)

weather_h = pd.read_excel('../input/WeatherData.xlsx',sheet_name=7)
wlist = [weather_a,weather_b,weather_c,weather_d,weather_e,weather_f,weather_g,weather_h]

ilist = [0,1,2,3,4,5,6,7]

for w,i in zip(wlist,ilist):

    w['Year'] = w['Year'] + i
weather = pd.concat(wlist)

weather.tail()
weather.info()
mn_num = list(range(1,13))

mn_alpha = weather['Month'].unique()

mn_map = dict(zip(mn_alpha,mn_num))

mn_map
weather['Month'] = weather['Month'].map(mn_map)
mergeddata = pd.merge(weather,mergeddata,on=['Year','Month'],how='right')

mergeddata.head()
merged_test = pd.merge(weather,merged_test,on=['Year','Month'],how='right')

merged_test.head()
holiday = pd.read_excel('../input/Events_HolidaysData.xlsx')

holiday.head()
holiday.info()
holiday.Event.unique()
holiday['Month'] = holiday['MonthDate'].dt.month
holiday['Day'] = holiday['MonthDate'].dt.day
holiday['is_holiday'] = 1
mergeddata.dropna(inplace=True)
mergeddata = pd.merge(holiday,mergeddata,on=['Year','Month','Day'],how='right')

mergeddata.head()
mergeddata.info()
merged_test.info()
mergeddata.drop(['Year','MonthDate','Event','DayCategory','WeatherEvent'],axis=1,inplace=True)
mergeddata.is_holiday.fillna(0,inplace=True)
mergeddata.dropna(inplace=True)
test = pd.read_csv('../input/test.csv')

merged_test = pd.merge(macro,test,on=['Year','Month'],how='right')

merged_test = pd.merge(weather,merged_test,on=['Year','Month'],how='right')

merged_test = pd.merge(holiday,merged_test,on=['Year','Month','Day'],how='right')

merged_test.head()
merged_test.drop(['Year','MonthDate','Event','DayCategory','WeatherEvent'],axis=1,inplace=True)

merged_test.is_holiday.fillna(0,inplace=True)
mergeddata.head()
mergeddata.ProductCategory.unique()
from sklearn.preprocessing import LabelEncoder
lc = LabelEncoder()
mergeddata['AdvertisingExpenses (in Thousand Dollars)'].unique()
pd.set_option('display.max_columns', 500)
adv_mean = round(mergeddata[mergeddata['AdvertisingExpenses (in Thousand Dollars)']!='?']['AdvertisingExpenses (in Thousand Dollars)'].mean())
mergeddata['AdvertisingExpenses (in Thousand Dollars)'].replace({'?':adv_mean},inplace=True)
merged_test['AdvertisingExpenses (in Thousand Dollars)'].unique()
macro['AdvertisingExpenses (in Thousand Dollars)'].unique()
#merged_test['AdvertisingExpenses (in Thousand Dollars)'].replace({'?':np.nan})

#adv_mean_test = round(merged_test['AdvertisingExpenses (in Thousand Dollars)'].mean())

#merged_test['AdvertisingExpenses (in Thousand Dollars)'].fillna(adv_mean_test,inplace=True)
mergeddata.info()
mergeddata.head()
mergeddata.replace({'-':np.nan},inplace=True)
merged_test.replace({'-':np.nan},inplace=True)
original_columns = pd.Series(mergeddata.columns)

original_columns
mergeddata.columns = [i for i in range(0,mergeddata.shape[1])]
mergeddata[21].replace({'T':np.nan},inplace=True)
mergeddata.info()
for col in mergeddata.columns:

    if mergeddata[col].isnull().sum() > 0 :

        mergeddata[col].fillna(mergeddata[col].median(),inplace=True)
dummydata = pd.get_dummies(mergeddata)

dummydata.head()
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = dummydata.drop([30,40],axis=1)

y = mergeddata[40]
scaledData = pd.DataFrame(sc.fit_transform(x),columns=x.columns)

scaledData.head()
merged_test.info()
original_columns_test = pd.Series(merged_test.columns)

original_columns_test
merged_test.columns = [i for i in range(0,merged_test.shape[1])]

merged_test.head()
merged_test[21].replace({'T':np.nan},inplace=True)
merged_test.replace({'-':np.nan},inplace=True)
for col in merged_test.columns:

    if merged_test[col].isnull().sum() > 0 :

        merged_test[col].fillna(merged_test[col].median(),inplace=True)
testX = merged_test.drop(30,axis=1)

testX.head()
testX.info()
dummydata_test = pd.get_dummies(testX)

dummydata_test.head()
scaledData_test = pd.DataFrame(sc.fit_transform(dummydata_test),columns=dummydata_test.columns)

scaledData_test.head()
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.model_selection import train_test_split,cross_val_score
xtrain,xtest,ytrain,ytest = train_test_split(scaledData,y,random_state=2)
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import r2_score,mean_squared_error
import warnings

warnings.filterwarnings('ignore')
lr = LinearRegression()

r = Ridge(alpha=1)

l = Lasso(alpha=0.5)

e = ElasticNet(alpha=0.01)

rf = RandomForestRegressor()
mlist = [lr,r,l,e,rf]

mname = ['linear','ridge','lasso','elastic','forest']
for model,name in zip(mlist,mname):

    print('*****',name)

    acc_score = cross_val_score(model,x,y,cv=10)

    print('cross val score',acc_score.mean(),acc_score.var())

    model.fit(xtrain,ytrain)

    ypred = model.predict(xtest)

    print('r2 score',r2_score(ytest,ypred))

    print('rmse',mean_squared_error(ytest,ypred)**0.5)

    print('*****')
params_ridge = {"alpha":[0.01,0.5,0.1,1,2,10,50],

         "solver":["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],

         }

params_lasso = {"alpha":[0.01,0.5,0.1,1,2,10,50],

         }

params_elastic = {"alpha":[0.01,0.5,0.1,1,2,10,50],

         }



plist = [params_ridge,params_lasso,params_elastic]
from sklearn.model_selection import GridSearchCV
for model,param,name in zip(mlist,plist,mname):

    if (name != 'linear'):

        print('*******',name)

        grid = GridSearchCV(estimator=model,param_grid=param,cv=10,verbose=1)

        grid.fit(xtrain,ytrain)

        print(grid.best_params_)
x.shape
l.fit(scaledData,y)

ypredlasso = l.predict(scaledData_test)
# submissiondf = pd.DataFrame(sample['Year']).join(pd.DataFrame({'Sales(In ThousandDollars)':ypredlasso}))

# submissiondf.head()
# submissiondf.to_csv('submission_lr.csv',index=False)
param_rf = {

    'bootstrap': [True],

    'max_depth': [1,2,5,10,20],

    'max_features': [5,10,15],

    'min_samples_leaf': [3, 4, 5],

    'min_samples_split': [8, 10, 12],

    'n_estimators': [ 20, 30, 50,100]

}

# grid2 = GridSearchCV(estimator=rf,param_grid=param_rf,cv=5,verbose=1)

# grid2.fit(scaledData,y)

# grid2.best_params_
rf = RandomForestRegressor(bootstrap= True,

 max_depth=10,

 max_features=15,

 min_samples_leaf=3,

 min_samples_split= 8,

 n_estimators= 20)

rf.fit(scaledData,y)

ypredrf = rf.predict(scaledData_test)
# submissiondf = pd.DataFrame(sample['Year']).join(pd.DataFrame({'Sales(In ThousandDollars)':ypredrf}))

# submissiondf.to_csv('submission_rf.csv',index=False)
from sklearn.ensemble import AdaBoostRegressor
#ada = AdaBoostRegressor(base_estimator=rf)

ada = AdaBoostRegressor()
ada.fit(scaledData,y)

ypredada = ada.predict(scaledData_test)
# submissiondf = pd.DataFrame(sample['Year']).join(pd.DataFrame({'Sales(In ThousandDollars)':ypredada}))

# submissiondf.to_csv('submission_ada.csv',index=False)
from sklearn.ensemble import GradientBoostingRegressor
gbm = GradientBoostingRegressor()
gbm.fit(scaledData,y)

ypredgbm = gbm.predict(scaledData_test)
# submissiondf = pd.DataFrame(sample['Year']).join(pd.DataFrame({'Sales(In ThousandDollars)':ypredgbm}))

# submissiondf.to_csv('submission_gbm.csv',index=False)
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import xgboost as xgb
xgbreg = xgb.XGBRegressor()
xgbreg.fit(scaledData,y)
ypredxgb = xgbreg.predict(scaledData_test)
# submissiondf = pd.DataFrame(sample['Year']).join(pd.DataFrame({'Sales(In ThousandDollars)':ypredxgb}))

# submissiondf.to_csv('submission_xgb.csv',index=False)
from sklearn.ensemble import BaggingRegressor
bg = BaggingRegressor()
# bg.fit(scaledData,y)

# ypredbg = bg.predict(scaledData_test)

# submissiondf = pd.DataFrame(sample['Year']).join(pd.DataFrame({'Sales(In ThousandDollars)':ypredbg}))

# submissiondf.to_csv('submission_bg.csv',index=False)
ada.fit(xtrain,ytrain)

ypred = ada.predict(xtest)

print('r2 score',r2_score(ytest,ypred))

print('rmse',mean_squared_error(ytest,ypred)**0.5)
from sklearn.neighbors import KNeighborsRegressor
kn = KNeighborsRegressor()
kn.fit(xtrain,ytrain)

ypred = kn.predict(xtest)

print('r2 score',r2_score(ytest,ypred))

print('rmse',mean_squared_error(ytest,ypred)**0.5)
bg.fit(xtrain,ytrain)

ypred = bg.predict(xtest)

print('r2 score',r2_score(ytest,ypred))

print('rmse',mean_squared_error(ytest,ypred)**0.5)
rf.fit(xtrain,ytrain)

ypred = rf.predict(xtest)

print('r2 score',r2_score(ytest,ypred))

print('rmse',mean_squared_error(ytest,ypred)**0.5)
gbm.fit(xtrain,ytrain)

ypred = gbm.predict(xtest)

print('r2 score',r2_score(ytest,ypred))

print('rmse',mean_squared_error(ytest,ypred)**0.5)
from sklearn.svm import SVR
sv = SVR(kernel='linear')
sv.fit(xtrain,ytrain)

ypred = sv.predict(xtest)

print('r2 score',r2_score(ytest,ypred))

print('rmse',mean_squared_error(ytest,ypred)**0.5)
from sklearn.decomposition import PCA
pc = PCA(n_components=2)
pcax = pc.fit_transform(scaledData)
# xtrain,xtest,ytrain,ytest = train_test_split(pcax,y,random_state=2)

# rf = RandomForestRegressor()

# rf.fit(xtrain,ytrain)

# ypred = rf.predict(xtest)

# print('r2 score',r2_score(ytest,ypred))

# print('rmse',mean_squared_error(ytest,ypred)**0.5)
plt.figure(figsize=(25,20))

sns.heatmap(mergeddata.corr(),annot=True)
from statsmodels.api import OLS
model = OLS(y,scaledData.drop([19,36,4,21,2,29,23,8,15,12,'25_Democrats',1,7,31,18,10,13,14,6,11,3,16,17,9,22,20,35,24,28,27,32,34,38,37,33,5],axis=1)).fit()

model.summary()
#model.pvalues>0.05
scaledXtrain = scaledData.drop([19,34,37,21,20,18,16,14,12,11,10,7,4,3,2,1,\

                                8,15,'25_Democrats',13,6,17,9\

                                ,28,34,38,37,5],axis=1)
xtrain,xtest,ytrain,ytest = train_test_split(scaledXtrain,y,random_state=2)



rf= RandomForestRegressor(bootstrap= True,

 max_depth=10,

 min_samples_leaf=3,

 min_samples_split= 8,

 n_estimators= 20)

print('****rf')

rf.fit(xtrain,ytrain)

ypred = rf.predict(xtest)

print('r2 score',r2_score(ytest,ypred))

print('rmse',mean_squared_error(ytest,ypred)**0.5)



print('****ada')

ada=AdaBoostRegressor()

ada.fit(xtrain,ytrain)

ypred = ada.predict(xtest)

print('r2 score',r2_score(ytest,ypred))

print('rmse',mean_squared_error(ytest,ypred)**0.5)
plt.figure(figsize=(20,10))

pd.Series(rf.feature_importances_,scaledXtrain.columns).plot.barh()
modelcopy = model.summary()
scaledXtest = scaledData_test.drop([19,34,37,21,20,18,16,14,12,11,10,7,4,3,2,1,\

                                8,15,'25_Democrats',13,6,17,9\

                                ,28,34,38,37,5],axis=1)
# rf= RandomForestRegressor(bootstrap= True,max_depth=10,min_samples_leaf=3,min_samples_split= 8,n_estimators= 20)

# rf.fit(scaledXtrain,y)

# ypredrf = rf.predict(scaledXtest)

# submissiondf = pd.DataFrame(sample['Year']).join(pd.DataFrame({'Sales(In ThousandDollars)':ypredrf}))

# submissiondf.to_csv('submission_rf.csv',index=False)
# ada = AdaBoostRegressor()

# ada.fit(scaledXtrain,y)

# ypredada = ada.predict(scaledXtest)

# submissiondf = pd.DataFrame(sample['Year']).join(pd.DataFrame({'Sales(In ThousandDollars)':ypredada}))

# submissiondf.to_csv('submission_ada.csv',index=False)
# bg = BaggingRegressor()

# bg.fit(scaledData,y)

# ypredbg = bg.predict(scaledData_test)

# submissiondf = pd.DataFrame(sample['Year']).join(pd.DataFrame({'Sales(In ThousandDollars)':ypredbg}))

# submissiondf.to_csv('submission_bg.csv',index=False)
# kn.fit(scaledData,y)

# ypredkn = kn.predict(scaledData_test)

# submissiondf = pd.DataFrame(sample['Year']).join(pd.DataFrame({'Sales(In ThousandDollars)':ypredkn}))

# submissiondf.to_csv('submission_kn.csv',index=False)