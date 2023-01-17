# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

from math import sqrt

from sklearn.svm import SVR

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, ARDRegression

from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, RandomForestRegressor

df = pd.read_csv("../input/solar-power-generation-data/Plant_1_Generation_Data.csv")

df
# Sensor data from the plant. It's showing only from 1 source per plant

df2 = pd.read_csv("../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv")

df2

#pd.unique(df2.SOURCE_KEY)
df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])  

df2['DATE_TIME'] = pd.to_datetime(df2['DATE_TIME'])  
# Extract datetime to time of the day features (0.0 - 24.0 hour float format)

df['TIME'] = df['DATE_TIME'].dt.hour + df['DATE_TIME'].dt.minute / 60

df
# Every plant has many sources. Every sources has their own characteristic.

# For more percise, we separated dataset for each source.



source_list = pd.unique(df['SOURCE_KEY'])

print(source_list)



target_source = '1BY6WEcLGh8j5v7'



# we selected dataset from 1 source

dataset = df[df.SOURCE_KEY==target_source]

dataset.reset_index(drop=True, inplace=True)

print(dataset.shape)

dataset.head()
# Join yield table witn sensor table via datetime-index 

dataset=dataset.merge(df2,on="DATE_TIME", how='inner')

print(f"Null checking:\n{dataset.isnull().sum()}")

dataset
# Drop unformated colume

dataset = dataset.drop(['DATE_TIME', 'PLANT_ID_x', 'SOURCE_KEY_x', 'PLANT_ID_y', 'SOURCE_KEY_y'], axis=1)

dataset
# Feature engineering: Calculate delta-yield per 15 min.

dataset['DELTA_YIELD'] = dataset['TOTAL_YIELD'].diff().fillna(0)



# Feature engineering: Calculate moving average delta-yield.

dataset['MA_4'] = dataset['DELTA_YIELD'].rolling(window=4).mean().fillna(0) # 1 Hour MA

dataset['MA_96'] = dataset['DELTA_YIELD'].rolling(window=69).mean().fillna(0) # 1 Day MA



#dataset = dataset.round(4)



dataset
# shuffle dataset in-place and reset the index (increase distribution)

#dataset = dataset.sample(frac=1).reset_index(drop=True)

#dataset
# split data to 10 fold

data_fold = []

for i in range(0,10):

    f = dataset.iloc[i::10, :]

    data_fold.append(f)

data_fold[0]
# select features for training

#col_X = ['DC_POWER','AC_POWER','TIME','AMBIENT_TEMPERATURE','MODULE_TEMPERATURE','IRRADIATION','MA_4','MA_96']

col_X = ['TIME','DC_POWER','MODULE_TEMPERATURE']

col_y = ['DELTA_YIELD']
models_name = ['LinearRegression','Ridge', 'Lasso', 'ElasticNet', 'SGDRegressor','AdaBoostRegressor', 'BaggingRegressor', 'RandomForestRegressor']



models = []

models.append(LinearRegression())

models.append(Ridge(alpha=1.0))

models.append(Lasso(alpha=0.1,max_iter=2000))

models.append(ElasticNet(random_state=0,max_iter=2000000))

models.append(SGDRegressor(max_iter=1000, tol=1e-3))

models.append(AdaBoostRegressor(random_state=0, n_estimators=100))

models.append(BaggingRegressor(base_estimator=SVR(),n_estimators=10, random_state=0))

models.append(RandomForestRegressor(max_depth=5, random_state=0))

for i, m in enumerate(models):

    print(f"[{models_name[i]}] Predict 15 min instance yield from current data.")

    for f in range(0,10):

        #X = data_fold[f].loc[:, col_X].values

        #y = data_fold[f].loc[:, col_y].values.ravel()

        

        X = data_fold[f][col_X]

        y = data_fold[f][col_y]

        

        # Normalize

        X =(X-X.min())/(X.max()-X.min())

        y =(y-y.min())/(y.max()-y.min())

        

        y = y.values.ravel()



        # split into train test sets

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)



        m.fit(X_train, y_train)

        y_pred = m.predict(X_test)

        print(f"Fold({f}) RMSE : {sqrt(mean_squared_error(y_test, y_pred))}")

    print()