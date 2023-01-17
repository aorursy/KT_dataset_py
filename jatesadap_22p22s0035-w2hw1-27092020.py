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
g_data1 = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_1_Generation_Data.csv')
w_data1 = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')
print(g_data1.info(),g_data1.head(),w_data1.info(),w_data1.head())
# Convert the DATE_TIME column to a datetime data type.
g_data1['DATE_TIME'] = pd.to_datetime(g_data1['DATE_TIME'], dayfirst = True)
w_data1['DATE_TIME'] = pd.to_datetime(w_data1['DATE_TIME'], yearfirst = True)

print(g_data1.info(),g_data1.head(),w_data1.info(),w_data1.head())
g_data = g_data1.groupby('DATE_TIME')[['DC_POWER','AC_POWER', 'DAILY_YIELD','TOTAL_YIELD']].agg('sum').reset_index()
g_data2 = g_data1.groupby('DATE_TIME')[['DC_POWER','AC_POWER', 'DAILY_YIELD','TOTAL_YIELD']].agg('sum').reset_index()
g_data,g_data.info()
g_data['time'] = g_data['DATE_TIME'].dt.time
g_data['date'] = pd.to_datetime(g_data['DATE_TIME'].dt.date)
g_data
import matplotlib.pyplot as plt

g_data.plot(x= 'date', y='DC_POWER', style='.', figsize = (15, 8))
g_data.groupby('date')['DC_POWER'].agg('mean').plot(legend=True, colormap='Reds_r')
plt.ylabel('Power')
plt.title('DC POWER PLOT')
plt.show()
g_data.plot(x= 'time', y=['DC_POWER','AC_POWER'], style='.', figsize = (15, 8))
g_data.groupby('time')['AC_POWER'].agg('mean').plot(legend=True, colormap='Reds_r')
plt.ylabel('Power')
plt.title('DC - AC POWER PLOT')
plt.show()
g_data.plot(x='time', y='DAILY_YIELD', style='b.', figsize=(15,5))
g_data.groupby('time')['DAILY_YIELD'].agg('mean').plot(legend=True, colormap='Reds_r')
plt.title('DAILY YIELD')
plt.ylabel('Yield')
plt.show()
g_data.plot(x='date', y='DAILY_YIELD', style='b.', figsize=(15,5))
g_data.groupby('date')['DAILY_YIELD'].agg('mean').plot(legend=True, colormap='Reds_r')
plt.title('DAILY YIELD')
plt.ylabel('Yield')
plt.show()
w_data = w_data1[['DATE_TIME', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']]
w_data
g_data1.info(),g_data.info()
df = pd.merge(g_data,w_data,how='inner',on='DATE_TIME')
df
df.plot("DATE_TIME", "AC_POWER", style=".")
df.plot("DATE_TIME", "DC_POWER", style=".")
df.plot("DATE_TIME", "TOTAL_YIELD", style=".")
import datetime
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
df = pd.merge(g_data2,w_data,how='inner',left_on='DATE_TIME',right_on='DATE_TIME')
#F, DAY3, DAY7 = ExtractFeatures(merge_Plan)
F, DAY3, DAY7 = ExtractFeatures(df)
g_data.info()
df
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
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

Ran_For = RandomForestRegressor(random_state=1)
Dec_Tree = DecisionTreeRegressor(random_state=1)
Ran_For7 = RandomForestRegressor(random_state=1)
Dec_Tree7 = DecisionTreeRegressor(random_state=1)
Random_Forest_Score3DAY = K_fold_score(Ran_For, F, DAY3, cv=10)
Decision_Tree_Score3DAY = K_fold_score(Dec_Tree, F, DAY3, cv=10)

print(f'Random Forest Score 3 DAY is: {Random_Forest_Score3DAY}\nDecision Tree Score 3 DAY is: {Decision_Tree_Score3DAY}')
Random_Forest_Score7DAY = K_fold_score(Ran_For7, F, DAY7, cv=10)
Decision_Tree_Score7DAY = K_fold_score(Dec_Tree7, F, DAY7, cv=10)

print(f'Random Forest Score 7 DAY is: {Random_Forest_Score7DAY}\nDecision Tree Score 7 DAY is: {Decision_Tree_Score7DAY}')
df = pd.merge(g_data1, w_data1, on=["DATE_TIME"], how="inner")
df = df.drop(columns=["PLANT_ID_x", "PLANT_ID_y", "SOURCE_KEY_y"])

df,df.info()
from sklearn.linear_model import LinearRegression, Ridge, LassoLars

from sklearn.metrics import mean_squared_error as MSE
# group range of day that we interested in (day = 3, 7)
def group_date(day):
    cdf = df.copy()
    date = df["DATE_TIME"]

    for i in range(day):
        col = list(df.columns)
        date = date + np.timedelta64(1, "D")

        new_col = dict()
        for j in col[2:]:
            new_col[j] = j + f"_{i}"

        next_day = df.copy()
        next_day["DATE_TIME"] = date
        next_day = next_day.rename(columns=new_col)

        cdf = pd.merge(cdf, next_day, on=["DATE_TIME", "SOURCE_KEY_x"], how="inner")

    # get rid all nth feature except total_yield
    col = list(cdf.columns)
    col = [i for i in col if (i[-1] != str(day-1)) and (i not in ["DATE_TIME", "SOURCE_KEY_x"])] 
    col.sort()

    col.append(f"TOTAL_YIELD_{day-1}")

    ll = ["DATE_TIME", "SOURCE_KEY_x"]
    for i in col:
        ll.append(i)

    cdf = cdf[ll]

    return cdf

# train model and return record of rmse for all model
def fit_and_evaluate(cdf, day):
    np.random.seed(281)
    cdf = cdf.to_numpy()
    
    np.random.shuffle(cdf)
    
    LM = LinearRegression()
    R = Ridge(alpha=0.5)
    LL = LassoLars(alpha=0.5)

    num = len(cdf)//10

    x = []
    y = []

    for i in range(len(cdf)):
        x.append(cdf[i][2:2+day*6])
        y.append(cdf[i][-1])

    x = np.array(x)
    y = np.array(y)

    record = pd.DataFrame(dtype=np.float64, columns=["LinearRegression", "Ridge", "LARS Lasso"])

    for i in range(10):
        x_test = x[num*i:num*(i+1)]
        y_test = y[num*i:num*(i+1)]

        x_train = np.concatenate((x[:num*(i-1)], x[num*(i+1):]), axis=0)
        y_train = np.concatenate((y[:num*(i-1)], y[num*(i+1):]), axis=0)

        record.loc[i] = rmse_model(LM, x_train, y_train, x_test, y_test), \
                        rmse_model(R, x_train, y_train, x_test, y_test), \
                        rmse_model(LL, x_train, y_train, x_test, y_test)
    return record

# rmse 
def rmse_model(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    y_hat = model.predict(x_test)
    return MSE(y_test, y_hat) ** 0.5
cdf3 = group_date(3)
rec3 = fit_and_evaluate(cdf3, 3)

print("RMSE FROM PAST 3 DAY")
rec3
cdf7 = group_date(7)
rec7 = fit_and_evaluate(cdf7, 7)

print("RMSE FROM PAST 7 DAY")
rec7