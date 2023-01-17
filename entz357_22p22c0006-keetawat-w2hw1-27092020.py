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
# Import package
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
%matplotlib inline
plant1_gen = pd.read_csv("../input/solar-power-generation-data/Plant_1_Generation_Data.csv")
plant1_gen.tail()
plant1_gen.info()
print(f"No. of source_key: {len(plant1_gen['SOURCE_KEY'].unique())}")
print(f"No. of plant: {len(plant1_gen['PLANT_ID'].unique())}")
plant1_gen.drop("PLANT_ID", axis=1, inplace=True)
plant1_data = plant1_gen.copy()
plant1_data = plant1_data.groupby('DATE_TIME')[['DC_POWER', 'AC_POWER', 'DAILY_YIELD', 'TOTAL_YIELD']].agg('sum')
plant1_data = plant1_data.reset_index()
plant1_data.head()
plant1_data['DATE_TIME'] = pd.to_datetime(plant1_data['DATE_TIME'], errors='coerce')
plant1_data['time'] = plant1_data['DATE_TIME'].dt.time
plant1_data['date'] = pd.to_datetime(plant1_data['DATE_TIME'].dt.date)
plant1_data.shape
plant1_data.head()
plant1_data.plot(x='time', y='DC_POWER', style='.', figsize=(15,8))
plant1_data.groupby('time')['DC_POWER'].agg('mean').plot(legend=True, colormap='Reds_r')
plt.ylabel('DC Power')
plt.title('DC Power plot')
plant1_data.plot(x='time', y='DAILY_YIELD', style='.', colormap='Dark2',figsize=(15,5))
plant1_data.groupby('time')['DAILY_YIELD'].agg('mean').plot(legend=True, colormap='Reds_r')
plt.title('DAILY YIELD')
plt.ylabel('Yield')
plt.show()
# function to multi plot

def multi_plot(data= None, row = None, col = None, title='None'):
    cols = data.columns # take all column
    gp = plt.figure(figsize=(20,20)) 
    
    gp.subplots_adjust(wspace=0.2, hspace=0.8)
    for i in range(1, len(cols)+1):
        ax = gp.add_subplot(row,col, i)
        data[cols[i-1]].plot(ax=ax, style = 'k.')
        ax.set_title('{} {}'.format(title, cols[i-1]))
# pivot table data
daily_yield = plant1_data.pivot_table(values='DAILY_YIELD', index='time', columns='date')

# daiy yield each day
daily_yield.boxplot(figsize=(18,5), rot=90, grid=False)
plt.title('DAILY YIELD IN EACH DAY')
plt.show()
plant1_gen.head(100)
plant1_gen.isnull().sum()
plant1_sensor = pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')
plant1_sensor.tail()
print(f"No. of source_key: {len(plant1_sensor['SOURCE_KEY'].unique())}")
print(f"No. of plant: {len(plant1_sensor['PLANT_ID'].unique())}")
plant1_sensor.drop(['SOURCE_KEY', 'PLANT_ID'], axis=1, inplace=True)
plant1_sensor.head()
plant1_sensor.isnull().sum()
plant1_sensor.info()
plant1_sensor['DATE_TIME'] = pd.to_datetime(plant1_sensor['DATE_TIME'], errors='coerce')
plant1_sensor['date'] = pd.to_datetime(pd.to_datetime(plant1_sensor['DATE_TIME']).dt.date)
plant1_sensor['time'] = pd.to_datetime(plant1_sensor['DATE_TIME']).dt.time
plant1_sensor.tail()
plant1_sensor.plot(x='time', y = 'AMBIENT_TEMPERATURE' , style='b.', figsize=(15,5))
plant1_sensor.groupby('time')['AMBIENT_TEMPERATURE'].agg('mean').plot(legend=True, colormap='Reds_r')
plt.title('Daily AMBIENT TEMPERATURE MEAN (RED)')
plt.ylabel('Temperature (°C)')
plt.show()
ambient = plant1_sensor.pivot_table(values='AMBIENT_TEMPERATURE', index='time', columns='date')
ambient.boxplot(figsize=(15,5), grid=False, rot=90)
plt.title('AMBIENT TEMPERATURE BOXES')
plt.ylabel('Temperature (°C)')
plant1_sensor.plot(x='time', y='MODULE_TEMPERATURE', figsize=(15,8), style='b.')
plant1_sensor.groupby('time')['MODULE_TEMPERATURE'].agg('mean').plot(colormap='Reds_r', legend=True)
plt.title('DAILY MODULE TEMPERATURE & MEAN(red)')
plt.ylabel('Temperature(°C)')
module_temp = plant1_sensor.pivot_table(values='MODULE_TEMPERATURE', index='time', columns='date')
module_temp.boxplot(figsize=(15,5), grid=False, rot=90)
plt.title('MODULE TEMPERATURE BOXES')
plt.ylabel('Temperature (°C)')
plant1_sensor.plot(x='time', y = 'IRRADIATION', style='.', legend=True, figsize=(15,5))
plant1_sensor.groupby('time')['IRRADIATION'].agg('mean').plot(legend=True, colormap='Reds_r')
plt.title('IRRADIATION')
irra = plant1_sensor.pivot_table(values='IRRADIATION', index='time', columns='date')
irra.boxplot(figsize=(15,5), rot = 90, grid=False)
plt.title('IRRADIATION BOXES')
plant1_gen['DATE_TIME'] = pd.to_datetime(plant1_gen['DATE_TIME'], errors='coerce')
plant1_gen['time'] = plant1_gen['DATE_TIME'].dt.time
plant1_gen['date'] = pd.to_datetime(plant1_gen['DATE_TIME'].dt.date)
df = plant1_gen.merge(plant1_sensor.drop(['date', 'time'], axis=1), left_on="DATE_TIME", right_on="DATE_TIME")
df.tail()
cols = df.columns.drop(['DATE_TIME', 'SOURCE_KEY'])
fig = plt.figure(figsize=(12,12))
sns.heatmap(df[cols].corr(), cmap='RdYlGn', annot=True, linewidths=1)
cols = df.columns.drop(['DATE_TIME', 'SOURCE_KEY', 'DAILY_YIELD','TOTAL_YIELD'])
sns.pairplot(df[cols])
pd.to_datetime(pd.to_datetime(plant1_sensor['DATE_TIME']).dt.date)
def add_feature(df):
    df['year'] = pd.to_datetime(df['DATE_TIME']).dt.year
    df['month']= pd.to_datetime(df['DATE_TIME']).dt.month
    df['day'] = pd.to_datetime(df['DATE_TIME']).dt.day
    df['dayofweek'] = pd.to_datetime(df['DATE_TIME']).dt.dayofweek
    df['hour'] = pd.to_datetime(df['DATE_TIME']).dt.hour

add_feature(df)
df
df.plot(x='DATE_TIME', y = 'TOTAL_YIELD', figsize=(15,5))
plt.title('TOTAL_YIELD')
df.plot(x='DATE_TIME', y = 'IRRADIATION', figsize=(15,5))
plt.title('TOTAL_YIELD')
df2 = df.copy()
# add Target TOTAL_YIELD

df2['TOTAL_YIELD_t-1'] = df2.groupby(['SOURCE_KEY', 'time']).shift(4)['TOTAL_YIELD']
df2['TOTAL_YIELD_t-2'] = df2.groupby(['SOURCE_KEY', 'time']).shift(5)['TOTAL_YIELD']
df2['TOTAL_YIELD_t-3'] = df2.groupby(['SOURCE_KEY', 'time']).shift(6)['TOTAL_YIELD']

df2['TOTAL_YIELD_t-4'] = plant1_gen.groupby(['SOURCE_KEY','time']).shift(7)['TOTAL_YIELD']
df2['TOTAL_YIELD_t-5'] = plant1_gen.groupby(['SOURCE_KEY','time']).shift(8)['TOTAL_YIELD']
df2['TOTAL_YIELD_t-6'] = plant1_gen.groupby(['SOURCE_KEY','time']).shift(9)['TOTAL_YIELD']
df2['TOTAL_YIELD_t-7'] = plant1_gen.groupby(['SOURCE_KEY','time']).shift(10)['TOTAL_YIELD']

df2
df2 = df2.drop(['DATE_TIME', 'SOURCE_KEY', 'date', 'time'], axis=1)
df2
tenf = [df[df.index % 10 == i] for i in range(10)] 
tenf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
def KFoldScore(reg, X, y, cv=10):
    kf = KFold(n_splits = cv)
    kf.get_n_splits(X)
    accuracy = []
    for train_idx, test_idx in kf.split(X):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        reg.fit(X_train, y_train)
        y_pred = np.round(reg.predict(X_test))
        
        acc = np.sqrt(mean_squared_error(y_test, y_pred))
        accuracy.append(acc)
        
    return accuracy
LR = LinearRegression()
RF = RandomForestRegressor(max_depth=2, random_state=0)
CB = CatBoostRegressor(iterations=50, learning_rate=0.16, depth=4, verbose=0)
XGB = xgb.XGBRegressor(n_estimators=50, objective='reg:squarederror', learning_rate=0.16,
                      colsample_bytree=0.6, max_depth=4, min_child_weight=6)
X = df2.dropna().drop('TOTAL_YIELD_t-3',axis=1).values
y3 = df2.dropna()['TOTAL_YIELD_t-3'].values
X, y3
def print_Fold(score):
    for i in range(len(score)):
        print(f'Score for Fold {i+1}: {score[i]}')
    print(f'\n')
    print(f'Average Score: {np.mean(score)}')
LR_score = KFoldScore(LR, X, y3)
print_Fold(LR_score)
RF_score = KFoldScore(RF, X, y3)
print_Fold(RF_score)
CB_score = KFoldScore(CB, X, y3)
print_Fold(CB_score)
XGB_score = KFoldScore(XGB, X, y3)
print_Fold(XGB_score)
X = df2.dropna().drop('TOTAL_YIELD_t-7',axis=1).values
y7 = df2.dropna()['TOTAL_YIELD_t-7'].values
X, y7
LR_score_7 = KFoldScore(LR, X, y7)
print_Fold(LR_score_7)
RF_score_7 = KFoldScore(RF, X, y7)
print_Fold(RF_score_7)
CB_score_7 = KFoldScore(CB, X, y7)
print_Fold(CB_score_7)
XGB_score_7 = KFoldScore(XGB, X, y7)
print_Fold(XGB_score_7)
