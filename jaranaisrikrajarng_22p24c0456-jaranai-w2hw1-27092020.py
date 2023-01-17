# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Read CSV File
Gen01_file = '/kaggle/input/solar-power-generation-data/Plant_1_Generation_Data.csv'
Wea01_file = '/kaggle/input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv' 

Gen_df01 = pd.read_csv(Gen01_file)
Weather_df01 = pd.read_csv(Wea01_file)


print('1 Power Generation Data have {} records'.format(len(Gen_df01)))
Gen_df01.info()
Gen_df01.head()
print('1 Weather Sensor Data have {} records'.format(len(Weather_df01)))
Weather_df01.info()
Weather_df01.head()
# Let's visualize it anyway.
fig, ax = plt.subplots(2,1, figsize=(20,8))
ax[0].set_title("POWER PLANT 1 DAILY CAPACITY")  
sns.lineplot(data=[Gen_df01["DC_POWER"],Gen_df01["AC_POWER"]], ax=ax[0], palette="tab20", linewidth=1)
ax[1].set_title("WEATHER ON POWER PLANT 1")  
sns.lineplot(data=[Weather_df01["AMBIENT_TEMPERATURE"],Weather_df01["MODULE_TEMPERATURE"],Weather_df01["IRRADIATION"]], ax=ax[1], palette="tab20", linewidth=2.5)
Gen_df01 = pd.read_csv(Gen01_file, parse_dates=['DATE_TIME'], infer_datetime_format=True)
Plan01_P = Gen_df01.groupby('DATE_TIME').agg({
    'DC_POWER': 'mean',
    'AC_POWER': 'mean',
    'DAILY_YIELD': 'mean',
    'TOTAL_YIELD': 'mean',
}).reset_index()
Plan01_P.head()
#ตรวจสอบค่า Null
Plan01_P.isnull()
Weather_df01 = pd.read_csv(Wea01_file, parse_dates=['DATE_TIME'], infer_datetime_format=True)
Plan01_W = Weather_df01[['DATE_TIME', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']]
Plan01_W.head()
Plan01_W.isnull()
#Merge Data Power and Weather 01
#Extract Feature
import datetime
def ExtractFeatures(df, window_day = 3):
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

Merge_Plan01    = pd.merge(Plan01_P, Plan01_W, how='inner', left_on='DATE_TIME', right_on='DATE_TIME')
Fea, PW_DAY3, PW_DAY7 = ExtractFeatures(Merge_Plan01)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
def K_Fold(model, X, Y, cv=10):
    kf = KFold(n_splits=cv,random_state=None,shuffle=True)
    kf.get_n_splits(X,Y)
    
    rmse = []
    score=[]
    mae=[]
    
    for train_data, test_data in kf.split(X):
        X_train = X[train_data]
        X_test = X[test_data]

        Y_train = Y[train_data]
        Y_test = Y[test_data]
        
        m=model.fit(X_train, Y_train)
        Y_pred = np.round(model.predict(X_test))
        
        accur = np.sqrt(mean_squared_error(Y_test, Y_pred))
        rmse.append(accur)

        mae.append(mean_absolute_error(Y_test, Y_pred))
        score.append(m.score(X,Y))
        
    return np.mean(rmse),np.mean(score),np.mean(mae)
Random_Forest = RandomForestRegressor(random_state=10)
Decision_Tree = DecisionTreeRegressor(random_state=10)
Linear = LinearRegression()
#3 Day
Random_Forest_Score_3DAY = K_Fold(Random_Forest, Fea, PW_DAY3, cv=10)
Decision_Tree_Score_3DAY = K_Fold(Decision_Tree, Fea, PW_DAY3, cv=10)
Linear_Score_3DAY = K_Fold(Linear, Fea, PW_DAY3, cv=10)
print('Regression Result in 3 Day')
print('Random Forest is (RMSE,Score,MAE) :', Random_Forest_Score_3DAY)
print('Decision Tree is (RMSE,Score,MAE) :', Decision_Tree_Score_3DAY)
print('Linear Regression is (RMSE,Score,MAE) :', Linear_Score_3DAY)
#7 Day
Random_Forest_Score_7DAY = K_Fold(Random_Forest, Fea, PW_DAY7, cv=10)
Decision_Tree_Score_7DAY = K_Fold(Decision_Tree, Fea, PW_DAY7, cv=10)
Linear_Score_7DAY = K_Fold(Linear, Fea, PW_DAY7, cv=10)
print('Regression Result in 7 Day')
print('Random Forest is (RMSE,Score,MAE) :', Random_Forest_Score_7DAY)
print('Decision Tree is (RMSE,Score,MAE) :', Decision_Tree_Score_7DAY)
print('Linear Regression is (RMSE,Score,MAE) :', Linear_Score_7DAY)