import numpy as np 

import pandas as pd 

from warnings import filterwarnings

filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train_data = pd.read_csv("/kaggle/input/home-data-for-ml-course/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/home-data-for-ml-course/test.csv")

test_data.head()
train_data.info()
for i in train_data.columns:

    if train_data[i].dtype.type != np.int64 and train_data[i].dtype.type != np.float64:

        print(train_data[i].nunique())
from sklearn.preprocessing import OneHotEncoder



train_data2 = train_data.copy()

test_data2 = test_data.copy()



droplist = []

for i in train_data2.columns:

    if train_data2[i].dtype.type != np.int64 and train_data2[i].dtype.type != np.float64 and train_data2[i].nunique() <= 10:

        train_data2 = pd.concat([train_data2,pd.get_dummies(train_data2[i])],axis=1,sort=False)

        test_data2 = pd.concat([test_data2,pd.get_dummies(test_data2[i])],axis=1,sort=False)

    if train_data2[i].dtype.type != np.int64 and train_data2[i].dtype.type != np.float64:

        droplist.append(i)
droplist
train_data2.drop(droplist,axis=1,inplace=True)

test_data2.drop(droplist,axis=1,inplace=True)
print(train_data2.info())

print(test_data2.info())
train_data2.columns[:-196]
train_data2 = train_data2.apply(lambda x: x.fillna(x.mean()),axis=0)

test_data2  = test_data2.apply( lambda x: x.fillna(x.mean()),axis=0)
y_train_data = train_data2["SalePrice"]

train_data2.drop(["SalePrice"],axis=1,inplace=True)
for d in train_data2.columns:

    if d not in test_data2.columns:

        print(d)

        train_data2.drop([d],axis=1,inplace=True)
for d,g in enumerate(test_data2.columns):

    print(d,train_data2.columns[d],test_data2.columns[d])
train_data2.iloc[:,[d for d in range(len(train_data2.columns)) if d != 0]]
for d,g in enumerate(test_data2.columns):

    while train_data2.columns[d] != test_data2.columns[d]:

        print(d,train_data2.columns[d],test_data2.columns[d])

        train_data2 = train_data2.iloc[:,[x for x in range(len(train_data2.columns)) if x != d]]
print(train_data2.info())

print(test_data2.info())
x_train_data = train_data2.iloc[:,:]

x_test_data = test_data2.iloc[:,:]
from statsmodels.regression.linear_model import OLS



index_list = [x for x in range(0,len(x_train_data.columns))]



popped = []

for i in popped:

    index_list.pop(i)

done = False

while done == False:

    ols = OLS(endog=y_train_data,exog=x_train_data.iloc[:,index_list])

    results = ols.fit()

    #results.summary()

    maxpvalue = results.pvalues.values.argmax()

    if results.pvalues.values[maxpvalue] > 0.05:

        popped.append(maxpvalue)

        index_list.pop(maxpvalue)

    else:

        done = True

print(popped)
ols = OLS(endog=y_train_data,exog=x_train_data.iloc[:,index_list])

results = ols.fit()

print(results.summary())
x_train_data = x_train_data.iloc[:,index_list]

x_test_data = x_test_data.iloc[:,index_list]
from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test = train_test_split(x_train_data,y_train_data,test_size=0.2,random_state=42)
from sklearn.preprocessing import MinMaxScaler



mms1 = MinMaxScaler()

X_train_data = mms1.fit_transform(x_train_data)

X_train = mms1.transform(x_train)

X_test2 = mms1.transform(x_test)

X_test_data = mms1.transform(x_test_data)

mms7 = MinMaxScaler()

Y_train_data = mms7.fit_transform(y_train_data.values.reshape(-1,1))

Y_train = mms7.transform(y_train.values.reshape(-1,1))

Y_test2 = mms7.transform(y_test.values.reshape(-1,1))
from sklearn.metrics import r2_score,mean_absolute_error

from xgboost import XGBRegressor



xgb = XGBRegressor()

xgb.fit(X_train,Y_train)

ypred = xgb.predict(X_test2)

print("R2 Score: ",r2_score(Y_test2,ypred))

print("MAE: ",mean_absolute_error(Y_test2,ypred))

print("TRUE R2 Score: ",r2_score(y_test.values.reshape(-1,1),mms7.inverse_transform(ypred.reshape(-1,1))))

print("TRUE MAE: ",mean_absolute_error(y_test.values.reshape(-1,1),mms7.inverse_transform(ypred.reshape(-1,1))))
from xgboost import XGBRegressor



xgb2  = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,

       colsample_bynode=1, colsample_bytree=0.4, gamma=0.0002,

       importance_type='gain', learning_rate=0.009, max_delta_step=0,

       max_depth=8, min_child_weight=0, missing=None, n_estimators=3232,

       n_jobs=1, nthread=None, objective='reg:squarederror', random_state=0,

       reg_alpha=0.000035, reg_lambda=1, scale_pos_weight=1, seed=42,

       silent=None, subsample=0.45, verbosity=1)

xgb2.fit(X_train,Y_train)

ypred = xgb2.predict(X_test2)

print("R2 Score: ",r2_score(Y_test2,ypred))

print("MAE: ",mean_absolute_error(Y_test2,ypred))

print("TRUE R2 Score: ",r2_score(y_test.values.reshape(-1,1),mms7.inverse_transform(ypred.reshape(-1,1))))

print("TRUE MAE: ",mean_absolute_error(y_test.values.reshape(-1,1),mms7.inverse_transform(ypred.reshape(-1,1))))
submission = xgb2.predict(X_test_data)

submission_real = mms7.inverse_transform(submission.reshape(-1,1))

submission_final = pd.DataFrame({"Id":test_data.Id.values.reshape(-1),"SalePrice":submission_real.reshape(-1)})

submission_final.head()
submission_final.to_csv("submission.csv",index=False)