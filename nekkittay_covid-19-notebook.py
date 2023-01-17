import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.preprocessing import LabelEncoder

from datetime import datetime

import os
train_data = pd.read_csv('../input/covid19-global-forecasting-week-5/train.csv', index_col='Id')

test_data = pd.read_csv('../input/covid19-global-forecasting-week-5/test.csv', index_col='ForecastId')

submission_data = pd.read_csv('../input/covid19-global-forecasting-week-5/submission.csv')
Factors = ["Weight", "Population", "Target", "Date"]

l = LabelEncoder()

re = pd.to_datetime(train_data["Date"], errors='coerce')

train_data["Date"] = re.dt.strftime("%Y%m%d").astype(int)

train_data["Target"]=l.fit_transform(train_data["Target"])

train_data["Country_Region"]=l.fit_transform(train_data["Country_Region"])

x = train_data[Factors]

y = train_data.TargetValue

x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=0.8,test_size=0.2,random_state=0)

model = RandomForestRegressor(n_jobs=-1 ,random_state=1)

model.fit(x_train, y_train)
train_data.head()
score = model.score(x_valid, y_valid)

print(score)
test_cols = ["Weight", "Population", "Target", "Date"]

re_test = pd.to_datetime(test_data["Date"], errors='coerce')

test_data["Date"] = re_test.dt.strftime("%Y%m%d").astype(int)

test_data["Target"]=l.fit_transform(test_data["Target"])

test_data_pred = test_data[test_cols]

pred = model.predict(test_data_pred)
pred_list = [x for x in pred]

output = pd.DataFrame({"Id": test_data_pred.index,"TargetValue": pred_list})

print(output)
Q1 = output.groupby(["Id"])['TargetValue'].quantile(q=0.05).reset_index()

Q2 = output.groupby(["Id"])['TargetValue'].quantile(q=0.5).reset_index()

Q3 = output.groupby(["Id"])['TargetValue'].quantile(q=0.95).reset_index()



Q1.columns = ["Id", "0.05"]

Q2.columns = ["Id", "0.5"]

Q3.columns = ["Id", "0.95"]
concatOut = pd.concat([Q1,Q2['0.5'],Q3['0.95']],1)

concatOut["Id"] = concatOut["Id"] 

concatOut.head()
sub = pd.melt(concatOut, id_vars=["Id"], value_vars=['0.05','0.5','0.95'])

sub['ForecastId_Quantile']=sub["Id"].astype(str)+'_'+sub['variable']

sub['TargetValue']=sub['value']

sub=sub[['ForecastId_Quantile','TargetValue']]

sub.reset_index(drop=True,inplace=True)

sub.to_csv("submission.csv",index=False)

sub.head()