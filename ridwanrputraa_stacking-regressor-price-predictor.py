import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv('../input/craigslist-carstrucks-data/vehicles.csv')

df.head(2)
df.columns
df.info()
df = df.drop(columns=['url',

                      'region_url', 

                      'image_url', 

                      'lat', 

                      'long',

                      'vin',

                      'description',

                      'id',

                      'paint_color'])
df.shape
df.isnull().sum(axis=1)
df.head(2)


dfmissing = df.isnull().sum().to_frame()

dfmissing.columns = ['null']

dfmissing['numberrow'] = df.count()

dfmissing['pct'] = dfmissing['null']/len(df)

dfmissing
#all country is null

df = df.drop(columns = ['county'],axis = 1)
#9 missing per row hapus

df = df[df.isnull().sum(axis=1) < 9]

df.shape
df.describe().T

dfmissing = df.isnull().sum().to_frame()

dfmissing.columns = ['null']

dfmissing['numberrow'] = df.count()

dfmissing['pct'] = dfmissing['null']/len(df)

dfmissing
#remove harga 0

df = df[df.price != 0]

df.shape
#cek tahun

plt.figure(figsize=(15,9))

ax = sns.countplot(x='year',data=df);

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right",fontsize=10);

#ambil dari 1995

df = df[df.year > 1995]

df.shape
#cek tahun

plt.figure(figsize=(15,9))

ax = sns.countplot(x='year',data=df);

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right",fontsize=10);

df.odometer.max(), df.odometer.min()
sns.kdeplot(data=df.odometer)
df = df[~(df.odometer > 300000)]

df.shape
sns.kdeplot(data=df.odometer)
#hapus <20

df = df[~(df.odometer < 20)]

df.shape
sns.kdeplot(data=df.odometer)
df.head(2)
dfmissing = df.isnull().sum().to_frame()

dfmissing.columns = ['null']

dfmissing['numberrow'] = df.count()

dfmissing['pct'] = dfmissing['null']/len(df)

dfmissing
df.describe().T

df.price.max()
sns.kdeplot(data=df.price)
sdf = df[~(df.price > 50000)]

sdf.shape
sns.kdeplot(data=sdf.price)
df =sdf
df.head()
dfmissing = df.isnull().sum().to_frame()

dfmissing.columns = ['null']

dfmissing['numberrow'] = df.count()

dfmissing['pct'] = dfmissing['null']/len(df)

dfmissing
df = df.drop(columns = ['size'],axis = 1)
dfmissing = df.isnull().sum().to_frame()

dfmissing.columns = ['null']

dfmissing['numberrow'] = df.count()

dfmissing['pct'] = dfmissing['null']/len(df)

dfmissing
df.manufacturer.value_counts()
## feature selection 

dflearn = df[['price','year','cylinders','fuel','odometer','title_status','transmission','condition']]

dflearn
dfmissing = dflearn.isnull().sum().to_frame()

dfmissing.columns = ['null']

dfmissing['numberrow'] = dflearn.count()

dfmissing['pct'] = dfmissing['null']/len(dflearn)

dfmissing
print(dflearn.condition.value_counts())
dflearn['condition'].fillna("unknown",inplace = True)
dfmissing = dflearn.isnull().sum().to_frame()

dfmissing.columns = ['null']

dfmissing['numberrow'] = dflearn.count()

dfmissing['pct'] = dfmissing['null']/len(dflearn)

dfmissing
dflearn = dflearn.dropna()
dflearn.condition.value_counts()
#map categorical string to int

condition_ = {'excellent' :0, 

         'good'      :1, 

         'like new'  :2, 

         'fair'      :3,

         'new'       :4,

         'salvage'   :5,

         'unknown'   :6}

cylinders_ = {'4 cylinders' :4, 

         '6 cylinders'      :6, 

         '8 cylinders'      :8, 

         '5 cylinders'      :5,

         '10 cylinders'     :10,

         '3 cylinders'      :3,

         '12 cylinders'     :12,

         'other'             :0}

status_ = {'clean'      :1, 

         'rebuilt'      :2, 

         'salvage'      :3, 

         'lien'         :4,

         'missing'      :5,

         'parts only'   :6}



transmission_ = {'automatic'  :1, 

                 'manual'     :2, 

                 'other'      :3}

fuel_ = {'gas' :1, 

         'diesel'      :2, 

         'hybrid'      :3,

         'electric'    :4,

         'other'       :5}
dflearn['condition'] = dflearn['condition'].map(condition_)

dflearn['title_status'] = dflearn['title_status'].map(status_)

dflearn['fuel'] = dflearn['fuel'].map(fuel_)

dflearn['transmission'] = dflearn['transmission'].map(transmission_)

dflearn['cylinders'] = dflearn['cylinders'].map(cylinders_)
dflearn.head()
#dflearn = dflearn.drop(["index"],axis =1)

dflearn.reset_index(inplace = True)
dflearn = dflearn.drop("index",axis = 1)
dflearn
corr = dflearn.corr()

corr.style.background_gradient(cmap='coolwarm')
from sklearn import linear_model

from sklearn.linear_model import Ridge

from sklearn.svm import SVR

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.neural_network import MLPRegressor

from xgboost import XGBRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import Ridge

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error,r2_score
y = dflearn[['price']]

X = dflearn.drop(columns=['price'])

Xtrain, Xtest, ytrain, ytest = train_test_split(X,y, test_size=0.25,random_state = 42)
regr= linear_model.LinearRegression()

rr  = Ridge(alpha=2)

rfr = RandomForestRegressor(max_depth=2, random_state=3,n_estimators = 1000)

#svr = SVR(epsilon=0.2)

gbr = GradientBoostingRegressor(random_state=0)

xgb = XGBRegressor(random_state=0)

mlp = MLPRegressor()

dtr = DecisionTreeRegressor()
regr.fit(Xtrain, ytrain)

rr.fit(Xtrain, ytrain)

rfr.fit(Xtrain, ytrain)

#svr.fit(Xtrain, ytrain)

xgb.fit(Xtrain, ytrain)

mlp.fit(Xtrain, ytrain)

dtr.fit(Xtrain, ytrain)
gbr.fit(Xtrain, ytrain)
ypred0 = regr.predict(Xtest)

ypred1 = rr.predict(Xtest)

ypred2 = rfr.predict(Xtest)

#ypred3 = svr.predict(Xtest)

ypred4 = xgb.predict(Xtest)

ypred5 = gbr.predict(Xtest)

ypred6 = mlp.predict(Xtest)

ypred7 = dtr.predict(Xtest)
rmse0 = mean_squared_error(ytest, ypred0)

rmse1 = mean_squared_error(ytest, ypred1)

rmse2 = mean_squared_error(ytest, ypred2)

#rmse3 = mean_squared_error(ytest, ypred3)

rmse4 = mean_squared_error(ytest, ypred4)

rmse5 = mean_squared_error(ytest, ypred5)

rmse6 = mean_squared_error(ytest, ypred6)

rmse7 = mean_squared_error(ytest, ypred7)

rmse0 ,rmse1, rmse2,rmse4, rmse5,rmse6,rmse7

r20 = r2_score(ytest, ypred0)

r21 = r2_score(ytest, ypred1)

r22 = r2_score(ytest, ypred2)

#r23 = r2_score(ytest, ypred3)

r24 = r2_score(ytest, ypred4)

r25 = r2_score(ytest, ypred5)

r26 = r2_score(ytest, ypred6)

r27 = r2_score(ytest, ypred7)



r20 ,r21 ,r22 ,r24,r25,r26,r27

#Stacking Regressi

y = dflearn[['price']]

X = dflearn.drop(columns=['price'])

Xtrain, Xtest, ytrain, ytest = train_test_split(X,y, test_size=0.25,random_state = 42)

Xtrain, xval, ytrain, yval = train_test_split(Xtrain,ytrain, test_size=0.25,random_state = 42)
regr.fit(Xtrain, ytrain)

rr.fit(Xtrain, ytrain)

rfr.fit(Xtrain, ytrain)

#svr.fit(Xtrain, ytrain)

gbr.fit(Xtrain, ytrain)

xgb.fit(Xtrain, ytrain)

mlp.fit(Xtrain, ytrain)

dtr.fit(Xtrain, ytrain)
ypred0val = regr.predict(xval)

ypred1val = rr.predict(xval)

ypred2val = rfr.predict(xval)

#ypred3val = svr.predict(xval)

ypred4val = gbr.predict(xval)

ypred5val = xgb.predict(xval)

ypred6val = mlp.predict(xval)

ypred7val = dtr.predict(xval)

ypred0test = regr.predict(Xtest)

ypred1test = rr.predict(Xtest)

ypred2test = rfr.predict(Xtest)

#ypred3test = svr.predict(Xtest)

ypred4test = gbr.predict(Xtest)

ypred5test = xgb.predict(Xtest)

ypred6test = mlp.predict(Xtest)

ypred7test = dtr.predict(Xtest)
dfresult_val = pd.DataFrame(data=ypred0val,columns = ['linear'])

dfresult_val['ridge'] = ypred1val

dfresult_val['ftr']=ypred2val

#dfresult_val['svr'] = ypred3val

dfresult_val['gb']  = ypred4val

dfresult_val['xgb'] = ypred5val

dfresult_val['mlp'] = ypred6val

dfresult_val['dtr'] = ypred7val

dfresult_val['target'] = yval.values

dfresult_val.head()
dfresult_test = pd.DataFrame(data=ypred0test,columns = ['linear'])

dfresult_test['ridge'] = ypred1test

dfresult_test['ftr'] =ypred2test

#dfresult_test['svr'] = ypred3test

dfresult_test['gb']  = ypred4test

dfresult_test['xgb'] = ypred5test

dfresult_test['mlp'] = ypred6test

dfresult_test['dtr'] = ypred7test

dfresult_test['target'] = ytest.values

dfresult_test.head()
ytrain_st =dfresult_val['target']

Xtrain_st =dfresult_val.drop(columns=['target'])



ytest_st = dfresult_test['target']

Xtest_st = dfresult_test.drop(columns=['target'])

regr.fit(Xtrain_st, ytrain_st)

ypred_st0 = regr.predict(Xtest_st)

print(r2_score(ytest, ypred_st0))

print(mean_squared_error(ytest, ypred_st0))
rr.fit(Xtrain_st, ytrain_st)

ypred_st1 = rr.predict(Xtest_st)

print(r2_score(ytest, ypred_st1))

print(mean_squared_error(ytest, ypred_st1))
rfr.fit(Xtrain_st, ytrain_st)

ypred_st2 = rfr.predict(Xtest_st)

print(r2_score(ytest, ypred_st2))

print(mean_squared_error(ytest, ypred_st2))
#svr.fit(Xtrain_st, ytrain_st)

#ypred_st3 = svr.predict(Xtest_st)

#print(r2_score(ytest, ypred_st3))

#print(mean_squared_error(ytest, ypred_st3))
gbr.fit(Xtrain_st, ytrain_st)

ypred_st4 = gbr.predict(Xtest_st)

print(r2_score(ytest, ypred_st4))

print(mean_squared_error(ytest, ypred_st4))
xgb.fit(Xtrain_st, ytrain_st)

ypred_st5 = xgb.predict(Xtest_st)

print(r2_score(ytest, ypred_st5))

print(mean_squared_error(ytest, ypred_st5))
mlp.fit(Xtrain_st, ytrain_st)

ypred_st6 = mlp.predict(Xtest_st)

print(r2_score(ytest, ypred_st6))

print(mean_squared_error(ytest, ypred_st6))
dtr.fit(Xtrain_st, ytrain_st)

ypred_st7 = dtr.predict(Xtest_st)

print(r2_score(ytest, ypred_st7))

print(mean_squared_error(ytest, ypred_st7))