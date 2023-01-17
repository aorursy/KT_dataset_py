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
import datetime
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression

from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm

from scipy.stats import normaltest

import holoviews as hv

from holoviews import opts

import cufflinks as cf

hv.extension('bokeh')
cf.set_config_file(offline = True)

sns.set(style="whitegrid")
PATH_P1 = '/kaggle/input/solar-power-generation-data/Plant_1_Generation_Data.csv'
data_P1  = pd.read_csv(PATH_P1) # load data
data_P1.head()
Plant_S = data_P1[['DATE_TIME', 'DC_POWER', 'AC_POWER', 'DAILY_YIELD', 'TOTAL_YIELD']]
Plant_S.head()
Plant_S.info() 
Plant_S = Plant_S.groupby('DATE_TIME')[['DC_POWER','AC_POWER', 'DAILY_YIELD','TOTAL_YIELD']].agg('sum')
Plant_S = Plant_S.reset_index()
Plant_S.head()
Plant_S['DATE_TIME'] = pd.to_datetime(Plant_S['DATE_TIME'], errors='coerce')
Plant_S['time'] = Plant_S['DATE_TIME'].dt.time

Plant_S['date'] = pd.to_datetime(Plant_S['DATE_TIME'].dt.date)
Plant_S.head()
Plant_S.info()
Plant_S.plot(x= 'time', y='DC_POWER', style='.', figsize = (15, 8))

Plant_S.groupby('time')['DC_POWER'].agg('mean').plot(legend=True, colormap='Reds_r')

plt.ylabel('DC Power')

plt.title('DC POWER PLOT')

plt.show()
Plant_S.plot(x='time', y='DAILY_YIELD', style='b.', figsize=(15,5))

Plant_S.groupby('time')['DAILY_YIELD'].agg('mean').plot(legend=True, colormap='Reds_r')

plt.title('DAILY YIELD')

plt.ylabel('Yield')

plt.show()
PATH_P2 = '/kaggle/input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv'
data_P2  = pd.read_csv(PATH_P2) 
data_P2.head()
Plant_W = data_P2[['DATE_TIME', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']]
Plant_W.head()
Plant_W.info()
Plant_W['DATE_TIME'] = pd.to_datetime(Plant_W['DATE_TIME'], errors='coerce')
Plant_W['date'] = pd.to_datetime(pd.to_datetime(Plant_W['DATE_TIME']).dt.date)

Plant_W['time'] = pd.to_datetime(Plant_W['DATE_TIME']).dt.time
Plant_W.head()
Plant_W.plot(x='time', y = 'AMBIENT_TEMPERATURE' , style='b.', figsize=(15,5))

Plant_W.groupby('time')['AMBIENT_TEMPERATURE'].agg('mean').plot(legend=True, colormap='Reds_r')

plt.title('Daily AMBIENT TEMPERATURE MEAN (RED)')

plt.ylabel('Temperature (°C)')

plt.show()
Plant_W.plot(x='time', y='MODULE_TEMPERATURE', figsize=(15,8), style='b.')

Plant_W.groupby('time')['MODULE_TEMPERATURE'].agg('mean').plot(colormap='Reds_r', legend=True)

plt.title('DAILY MODULE TEMPERATURE & MEAN(red)')

plt.ylabel('Temperature(°C)')
Plant_W.plot(x='time', y = 'IRRADIATION', style='.', legend=True, figsize=(15,5))

Plant_W.groupby('time')['IRRADIATION'].agg('mean').plot(legend=True, colormap='Reds_r')

plt.title('IRRADIATION')
Power_plan = Plant_S.merge(Plant_W, left_on='DATE_TIME', right_on='DATE_TIME')
Power_plan.head()
del Power_plan['date_x']

del Power_plan['date_y']

del Power_plan['time_x']

del Power_plan['time_y']
Power_plan.head()
Power_plan.info()
correlation_plan = Power_plan.drop(columns=['DAILY_YIELD', 'TOTAL_YIELD']).corr(method = 'spearman')
plt.figure(dpi=100)

sns.heatmap(correlation_plan, robust=True, annot=True, fmt='0.3f', linewidths=.5, square=True)

plt.show()
plan_Ge = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_1_Generation_Data.csv', parse_dates=['DATE_TIME'], infer_datetime_format=True)

plan_Ge = plan_Ge.groupby('DATE_TIME').agg({

    'DC_POWER': 'mean',

    'AC_POWER': 'mean',

    'DAILY_YIELD': 'mean',

    'TOTAL_YIELD': 'mean',

}).reset_index()



plan_Ge.head()
Plan_Wea = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv', parse_dates=['DATE_TIME'], infer_datetime_format=True)

Plan_Wea = Plan_Wea[['DATE_TIME', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']]



Plan_Wea.head()
def ExtractFeatures(Plan_df, window_day = 3):

    merge_Plan = Plan_df.copy()

    for i in range(1, window_day+1):

        merge_Plan[f'DATE_TIME_P{i}D'] = merge_Plan['DATE_TIME'] + datetime.timedelta(days=-i)

        

    merge_Plan['DATE_TIME_N3D'] = merge_Plan['DATE_TIME'] + datetime.timedelta(days=3)

    merge_Plan['DATE_TIME_N7D'] = merge_Plan['DATE_TIME'] + datetime.timedelta(days=7)



    

    for i in range(1, window_day+1):

        merge_Plan = merge_Plan.join(Plan_df.set_index('DATE_TIME'), how='inner', on=f'DATE_TIME_P{i}D', rsuffix=f'_P{i}D')

        

    merge_Plan = merge_Plan.join(Plan_df.set_index('DATE_TIME')[['TOTAL_YIELD']], how='inner', on='DATE_TIME_N3D', rsuffix='_N3D')

    merge_Plan = merge_Plan.join(Plan_df.set_index('DATE_TIME')[['TOTAL_YIELD']], how='inner', on='DATE_TIME_N7D', rsuffix='_N7D')

    

    Col_feature = []

    Col_Label = ['TOTAL_YIELD_N3D', 'DATE_TIME_N7D']

    for c in merge_Plan.columns:

        if c.startswith('DATE_TIME'):

            continue

        if c in Col_Label:

            continue

        Col_feature.append(c)

        

    F    = merge_Plan[Col_feature].values

    DAY3 = merge_Plan['TOTAL_YIELD_N3D'].values

    DAY7 = merge_Plan['TOTAL_YIELD_N7D'].values

    return F, DAY3, DAY7
merge_Plan    = pd.merge(plan_Ge, Plan_Wea, how='inner', left_on='DATE_TIME', right_on='DATE_TIME')

F, DAY3, DAY7 = ExtractFeatures(merge_Plan)
F
DAY3
DAY7
def K_fold_score(fore, F, DAY, cv=10):

    kf = KFold(n_splits=cv)

    kf.get_n_splits(F)

    

    accuracy = []

    

    for train_data, test_data in kf.split(F):

        F_train = F[train_data]

        F_test = F[test_data]

        F_train = F[train_data]

        F_test = F[test_data]

        

        fore.fit(F_train, F_train)

        F_pred = np.round(fore.predict(F_test))

        

        accur = np.sqrt(mean_squared_error(F_test, F_pred))

        accuracy.append(accur)

        

    return np.mean(accuracy)
from sklearn.ensemble import RandomForestRegressor
Ran_For = RandomForestRegressor(random_state=10)

Dec_Tree = DecisionTreeRegressor(random_state=10)
Random_Forest_Score3DAY = K_fold_score(Ran_For, F, DAY3, cv=10)

Decision_Tree_Score3DAY = K_fold_score(Dec_Tree, F, DAY3, cv=10)



print(f'Random Forest Score 3 DAY is: {Random_Forest_Score3DAY}\nDecision Tree Score 3 DAY is: {Decision_Tree_Score3DAY}')
Ran_For = RandomForestRegressor(random_state=10)

Dec_Tree = DecisionTreeRegressor(random_state=10)
Random_Forest_Score7DAY = K_fold_score(Ran_For, F, DAY7, cv=10)

Decision_Tree_Score7DAY = K_fold_score(Dec_Tree, F, DAY7, cv=10)



print(f'Random Forest Score 7 DAY is: {Random_Forest_Score7DAY}\nDecision Tree Score 7 DAY is: {Decision_Tree_Score7DAY}')