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
g1_df = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_1_Generation_Data.csv', parse_dates=['DATE_TIME'], infer_datetime_format=True)

g1_df = g1_df.groupby('DATE_TIME').agg({

    'DC_POWER': 'mean',

    'AC_POWER': 'mean',

    'DAILY_YIELD': 'mean',

    'TOTAL_YIELD': 'mean',

}).reset_index()

g1_df
w1_df = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv', parse_dates=['DATE_TIME'], infer_datetime_format=True)

w1_df = w1_df[['DATE_TIME', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']]

w1_df
import datetime

def extractFeatures(df, window_day = 3):

    m_df = df.copy()

    for i in range(1, window_day+1):

        m_df[f'DATE_TIME_P{i}D'] = m_df['DATE_TIME'] + datetime.timedelta(days=-i)

        

    m_df['DATE_TIME_N3D'] = m_df['DATE_TIME'] + datetime.timedelta(days=3)

    m_df['DATE_TIME_N7D'] = m_df['DATE_TIME'] + datetime.timedelta(days=7)



    

    for i in range(1, window_day+1):

        m_df = m_df.join(df.set_index('DATE_TIME'), how='inner', on=f'DATE_TIME_P{i}D', rsuffix=f'_P{i}D')

        

    m_df = m_df.join(df.set_index('DATE_TIME')[['TOTAL_YIELD']], how='inner', on='DATE_TIME_N3D', rsuffix='_N3D')

    m_df = m_df.join(df.set_index('DATE_TIME')[['TOTAL_YIELD']], how='inner', on='DATE_TIME_N7D', rsuffix='_N7D')

    

    feature_columns = []

    label_columns = ['TOTAL_YIELD_N3D', 'DATE_TIME_N7D']

    for c in m_df.columns:

        if c.startswith('DATE_TIME'):

            continue

        if c in label_columns:

            continue

        feature_columns.append(c)

        

    X = m_df[feature_columns].values

    y3 = m_df['TOTAL_YIELD_N3D'].values

    y7 = m_df['TOTAL_YIELD_N7D'].values

    return X, y3, y7
m_df = pd.merge(g1_df, w1_df, how='inner', left_on='DATE_TIME', right_on='DATE_TIME')

X, y3, y7 = extractFeatures(m_df)
from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression

from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error
def KFoldScore(reg, X, y, cv=5):

    kf = KFold(n_splits=cv)

    kf.get_n_splits(X)

    

    accuracies = []

    

    for train_idx, test_idx in kf.split(X):

        X_train = X[train_idx]

        X_test = X[test_idx]

        y_train = y[train_idx]

        y_test = y[test_idx]

        

        reg.fit(X_train, y_train)

        y_pred = np.round(reg.predict(X_test))

        

        acc = np.sqrt(mean_squared_error(y_test, y_pred))

        accuracies.append(acc)

        

    return np.mean(accuracies)
dt = DecisionTreeRegressor(random_state=1)

lr = LinearRegression()

nn = MLPRegressor(max_iter=500)
dt_cv_score = KFoldScore(dt, X, y3, cv=5)

lr_cv_score = KFoldScore(lr, X, y3, cv=5)

nn_cv_score = KFoldScore(nn, X, y3, cv=5)



print(f'decision tree score: {dt_cv_score}\nlinear regression score: {lr_cv_score}\nNN score: {nn_cv_score}')
dt = DecisionTreeRegressor(random_state=1)

lr = LinearRegression()

nn = MLPRegressor(max_iter=500)
dt_cv_score = KFoldScore(dt, X, y7, cv=5)

lr_cv_score = KFoldScore(lr, X, y7, cv=5)

nn_cv_score = KFoldScore(nn, X, y7, cv=5)



print(f'decision tree score: {dt_cv_score}\nlinear regression score: {lr_cv_score}\nNN score: {nn_cv_score}')