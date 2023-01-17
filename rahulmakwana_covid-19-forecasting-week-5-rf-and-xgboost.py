import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/train.csv')
df_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/test.csv')
df_sample = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/submission.csv')
df_train.columns

df_test.columns
df_train.shape
df_test.shape
df_train.isnull().sum()
df_test.isnull().sum()
# Replacing all the Province_State that are null by the Country_Region values
df_train.Province_State.fillna(df_train.Country_Region, inplace=True)
df_test.Province_State.fillna(df_test.Country_Region, inplace=True)

df_train.County.fillna(df_train.Province_State, inplace=True)
df_test.County.fillna(df_test.Province_State, inplace=True)

df_train.isnull().sum()
df_train.columns
# taking care of categorical values from train set
# we can also use labelencoder for date column
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
df_train['Country_Region'] = labelencoder.fit_transform(df_train['Country_Region'])
df_train['Target'] = labelencoder.fit_transform(df_train['Target'])
# df_train['Date'] = labelencoder.fit_transform(df_train['Date'])

# taking care of categorical values from test set

df_test['Country_Region'] = labelencoder.fit_transform(df_test['Country_Region'])
df_test['Target'] = labelencoder.fit_transform(df_test['Target'])

# taking care of the date column
df_train['Date'] = pd.to_datetime(df_train['Date'], infer_datetime_format=True)
df_test['Date'] = pd.to_datetime(df_test['Date'], infer_datetime_format=True)

df_train.loc[:, 'Date'] = df_train.Date.dt.strftime("%Y%m%d")
df_train.loc[:, 'Date'] = df_train['Date'].astype(int)

df_test.loc[:, 'Date'] = df_test.Date.dt.strftime("%Y%m%d")
df_test.loc[:, 'Date'] = df_test['Date'].astype(int)
ID=df_train['Id']
FID=df_test['ForecastId']
# splitting the dataset for training and testing

y_train=df_train['TargetValue']
X_train=df_train.drop(['Id', 'County', 'Province_State','TargetValue'],axis=1)
df_test=df_test.drop(columns=['County','Province_State','ForecastId'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

X_train.columns
X_train
# Fitting Random Forest Regression to the dataset
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_jobs=-1)
estimators = 100
model.set_params(n_estimators=estimators)

scores = []

pipeline = Pipeline([('scaler2' , StandardScaler()),
                        ('RandomForestRegressor: ', model)])
pipeline.fit(X_train , y_train)
y_pred = pipeline.predict(X_test)

scores.append(pipeline.score(X_test, y_test))
print(scores)


y_pred_main = pipeline.predict(df_test)


main_submission = pd.DataFrame({'id':FID,'TargetValue':y_pred_main})
main_submission

# cross-validation method
from sklearn.model_selection import cross_val_score, KFold

cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
print("cross-validation score:",cv_scores.mean())

# Cross-validation with a k-fold method

kfold = KFold(n_splits=10, shuffle=True)
kf_cv_scores = cross_val_score(pipeline, X_train, y_train, cv=kfold)
print("K-fold CV average score:" ,kf_cv_scores.mean())
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error
from math import sqrt

print('explained variance_score:', explained_variance_score(y_test, y_pred))
print('max_error:', max_error(y_test, y_pred))
print('mean_absolute_error score:', mean_absolute_error(y_test, y_pred))
print('mean_squared_error score:', mean_squared_error(y_test, y_pred))
print('root mean_squared_error:', sqrt(mean_squared_error(y_test, y_pred)))
# Fitting xgboost Regressor to the dataset
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor 

model_2 = XGBRegressor(n_jobs=-1)
estimators = 1000
model_2.set_params(n_estimators=estimators)

scores = []

pipeline_2 = Pipeline([('scaler2' , StandardScaler()),
                        ('XGBRegressor: ', model)])
pipeline_2.fit(X_train , y_train)
y_pred_2 = pipeline_2.predict(X_test)

scores.append(pipeline_2.score(X_test, y_test))
print(scores)


y_pred_main=pipeline.predict(df_test)


main_submission_2 = pd.DataFrame({'id':FID,'TargetValue':y_pred_main})
main_submission_2
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error
from math import sqrt

print('explained variance_score:', explained_variance_score(y_test, y_pred_2))
print('max_error:', max_error(y_test, y_pred_2))
print('mean_absolute_error score:', mean_absolute_error(y_test, y_pred_2))
print('mean_squared_error score:', mean_squared_error(y_test, y_pred_2))
print('root mean_squared_error:', sqrt(mean_squared_error(y_test, y_pred_2)))
main_pred=pd.DataFrame({'id':FID,'TargetValue':y_pred_main})
print(main_pred)
a=main_pred.groupby(['id'])['TargetValue'].quantile(q=0.05).reset_index()
b=main_pred.groupby(['id'])['TargetValue'].quantile(q=0.5).reset_index()
c=main_pred.groupby(['id'])['TargetValue'].quantile(q=0.95).reset_index()
a.columns=['Id','q0.05']
b.columns=['Id','q0.5']
c.columns=['Id','q0.95']
a=pd.concat([a,b['q0.5'],c['q0.95']],1)
a['q0.05']=a['q0.05']
a['q0.5']=a['q0.5']
a['q0.95']=a['q0.95']
print(a)
sub=pd.melt(a, id_vars=['Id'], value_vars=['q0.05','q0.5','q0.95'])
sub['variable']=sub['variable'].str.replace("q","", regex=False)
sub['ForecastId_Quantile']=sub['Id'].astype(str)+'_'+sub['variable']
sub['TargetValue']=sub['value']
sub=sub[['ForecastId_Quantile','TargetValue']]
sub.reset_index(drop=True,inplace=True)
sub.to_csv("submission.csv",index=False)
sub.head()