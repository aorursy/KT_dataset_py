import numpy as np

import pandas as pd

from datetime import datetime

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score

import seaborn as sns

from sklearn.linear_model import LinearRegression
Plant1_data = pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')

Plant1_sensor = pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')
Plant1_data.isnull().sum()
Plant1_sensor.isnull().sum()
Plant1_data.info()
type_data = Plant1_data.groupby('DATE_TIME').count().reset_index().loc[:,['DATE_TIME','TOTAL_YIELD']]

type_data.head()
type_data = Plant1_data.groupby('PLANT_ID').count().reset_index().loc[:,['PLANT_ID','TOTAL_YIELD']]

type_data.head()
type_data = Plant1_data.groupby('SOURCE_KEY').count().reset_index().loc[:,['SOURCE_KEY','TOTAL_YIELD']]

type_data.head()
type_data = Plant1_data.groupby('DAILY_YIELD').count().reset_index().loc[:,['DAILY_YIELD','TOTAL_YIELD']]

type_data.head()
Plant1_data = Plant1_data.drop(["PLANT_ID","DC_POWER","AC_POWER"], axis = 1)

Plant1_data
Plant1_sensor.info()
type_data = Plant1_sensor.groupby('DATE_TIME').count().reset_index().loc[:,['DATE_TIME','MODULE_TEMPERATURE']]

type_data.head()
type_data = Plant1_sensor.groupby('PLANT_ID').count().reset_index().loc[:,['PLANT_ID','MODULE_TEMPERATURE']]

type_data.head()
type_data = Plant1_sensor.groupby('SOURCE_KEY').count().reset_index().loc[:,['SOURCE_KEY','MODULE_TEMPERATURE']]

type_data.head()
Plant1_sensor = Plant1_sensor.drop(["PLANT_ID","SOURCE_KEY"], axis = 1)

Plant1_sensor.head()
Plant1_data["DATE_TIME"] = pd.to_datetime(Plant1_data["DATE_TIME"])

Plant1_sensor["DATE_TIME"] = pd.to_datetime(Plant1_sensor["DATE_TIME"])

Plant1_table = pd.merge(Plant1_sensor,Plant1_data, on="DATE_TIME", how="inner")

Plant1_table
Plant1_table['Time'] = Plant1_table['DATE_TIME'][1].strftime('%H:%M')
Plant1_table
Plant1_table['Date'] = Plant1_table['DATE_TIME'][0].strftime('%m-%d-%Y')
Plant1_table
Plant1_table = Plant1_table.drop(["DATE_TIME"], axis = 1)

Plant1_table
lb_make = LabelEncoder()



Plant1_table["SOURCE_KEY_code"] = lb_make.fit_transform(Plant1_table["SOURCE_KEY"])

Plant1_table[["SOURCE_KEY", "SOURCE_KEY_code"]].head(11)
Plant1_table = Plant1_table.drop(["SOURCE_KEY","Time"], axis = 1)

Plant1_table
Plant1_table1 = Plant1_table.drop(["Date"], axis = 1)

Plant1_table1
predictors = Plant1_table1.iloc[0:200,1:6]

predictors.head(5)
targets = Plant1_table1.iloc[0:200,4:5]

targets.head(5)
X_train, X_test, y_train, y_test = train_test_split(predictors,targets, test_size=0.1, random_state=0)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
model = LinearRegression()
model.fit(X_train, y_train)
print(accuracy_score(y_train, model.predict(X_train), normalize=True))
kf = KFold(n_splits=10,shuffle=False)

kf.split(predictors)
X = predictors

y = targets
accuracy_model = []

 

for train_index, test_index in kf.split(predictors):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = y[train_index], y[test_index]



    model = clf.fit(X_train, y_train)

    accuracy_model.append(accuracy_score(y_test, model.predict(X_test), normalize=True))



print(accuracy_model)