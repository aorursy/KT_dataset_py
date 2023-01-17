import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import os
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

WORK_DIR = '/kaggle/input/used-cars-price-prediction'
df_train = pd.read_csv(WORK_DIR + '/train-data.csv')

df_test = pd.read_csv(WORK_DIR + '/test-data.csv')
df_train.head(5)
miss_percent = (df_train.isnull().sum() / len(df_train)) * 100

missing = pd.DataFrame({"percent":miss_percent, 'count':df_train.isnull().sum()}).sort_values(by="percent", ascending=False)

missing.loc[missing['percent'] > 0]
df_train = df_train.drop(columns=['New_Price', 'Unnamed: 0'], axis=1).dropna(axis=0, how='any')

df_test = df_test.drop(columns=['New_Price', 'Unnamed: 0'], axis=1).dropna(axis=0, how='any')
df_train['Mileage'].apply(lambda x: str(x).split(" ")[1]).unique()
df_train['Mileage'] = df_train['Mileage'].apply(lambda x: float(str(x).split(" ")[0]))

df_test['Mileage'] = df_test['Mileage'].apply(lambda x: float(str(x).split(" ")[0]))
df_train['Fuel_Type'].unique()
fuel_type_train = pd.get_dummies(df_train['Fuel_Type'])
df_train = df_train.drop('Fuel_Type', axis=1).join(fuel_type_train)
fuel_type_test = pd.get_dummies(df_test['Fuel_Type'])
df_test = df_test.drop('Fuel_Type', axis=1).join(fuel_type_test)
transmission_type_train = pd.get_dummies(df_train['Transmission'])
df_train = df_train.drop('Transmission', axis=1).join(transmission_type_train)
transmission_type_test = pd.get_dummies(df_test['Transmission'])
df_test = df_test.drop('Transmission', axis=1).join(transmission_type_test)
df_train['Power'].apply(lambda x: str(x).split(" ")[1]).unique()
df_train = df_train[~df_train.Power.str.contains("null")]
df_train['Power'] = df_train['Power'].apply(lambda x: float(str(x).split(" ")[0]))
df_test = df_test[~df_test.Power.str.contains("null")]
df_test['Power'] = df_test['Power'].apply(lambda x: float(str(x).split(" ")[0]))
df_train['Engine'].apply(lambda x: str(x).split(" ")[1]).unique()
df_train['Engine'] = df_train['Engine'].apply(lambda x: float(str(x).split(" ")[0]))
df_test['Engine'] = df_test['Engine'].apply(lambda x: float(str(x).split(" ")[0]))
df_train['Owner_Type'].unique()
dict_encode_owner = {owner: i for i, owner in enumerate(df_train['Owner_Type'].unique())}
dict_encode_owner
df_train['Owner_Type'] = df_train['Owner_Type'].replace(dict_encode_owner)
df_test['Owner_Type'] = df_test['Owner_Type'].replace(dict_encode_owner)
df_train['Location'].unique()
dict_encode_city = {city: i for i, city in enumerate(df_train['Location'].unique())}
df_train['Location'] = df_train['Location'].replace(dict_encode_city)
df_test['Location'] = df_test['Location'].replace(dict_encode_city)
df_train['Name'].unique()
df_train['Name'] = df_train['Name'].apply(lambda x: str(x).split(" ")[0].lower())
dict_encode_car_name = {car: i for i, car in enumerate(df_train['Name'].unique())}
dict_encode_car_name['opelcorsa'] = 29
df_train['Name'] = df_train['Name'].replace(dict_encode_car_name)
df_test['Name'] = df_test['Name'].apply(lambda x: str(x).split(" ")[0].lower())
df_test['Name'] = df_test['Name'].replace(dict_encode_car_name)
df_train['Name'].unique()
df_test['Name'].unique()
import xgboost as xgb

from sklearn.model_selection import train_test_split

from sklearn.metrics import explained_variance_score
net = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,

                       colsample_bynode=1, colsample_bytree=1, gamma=0,

                       importance_type='gain', learning_rate=0.08, max_delta_step=0,

                       max_depth=7, min_child_weight=1, missing=None, n_estimators=100,

                       n_jobs=1, nthread=None, random_state=0,

                       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

                       silent=None, subsample=0.75, verbosity=1, objective='reg:squarederror')
seed = np.random.randint(0,1000)
seed
#Random state 878 whas the most accuracy == 0.9520



traindf, testdf = train_test_split(df_train, test_size = 0.2, random_state=878)



traindf_x = traindf.drop(['Price'], axis = 1).values

traindf_y = traindf['Price'].values



testdf_x = testdf.drop(['Price'], axis = 1).values

testdf_y = testdf['Price'].values
net.fit(traindf_x, traindf_y)
predictions = net.predict(testdf_x)

print(explained_variance_score(predictions,testdf_y))
predictions = net.predict(df_test.values)
my_submission = pd.DataFrame(predictions, columns = ['Price'])

my_submission.to_csv('submission.csv', index=False)