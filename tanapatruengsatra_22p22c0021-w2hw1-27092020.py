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
df_p1g = pd.read_csv("/kaggle/input/solar-power-generation-data/Plant_1_Generation_Data.csv")
df_p1g.head(10)
df_p1w = pd.read_csv("/kaggle/input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv")
df_p1w.head(10)
import numpy as np
import math
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
from datetime import datetime
profile = ProfileReport(df_p1g, title="Profiling Report")
profile.to_widgets()
profile2 = ProfileReport(df_p1w, title="Weather Sensor Report")
profile2.to_widgets()
def create_fold(data):
    l = len(data)
    fold = []
    
    for i in range(l):
        m = i%10
        if i < 10:
            fold.append([])
        fold[m].append(i)
       
    return fold
fold = create_fold(df_p1g)
fold_data = []
for f in range(len(fold)):
    fold_data.append(df_p1g.iloc[fold[f]])
def clean_format_date_weather(d):
    date_str = d
    date_time = date_str.split()
    date = date_time[0]
    time_str = ":".join(date_time[1].split(":")[0:2])
    dt_string = date+ " " + time_str
    return dt_string

def clean_format_date_weather(d):
    date_str = d
    date_time = date_str.split()
    date = date_time[0]
    time_str = ":".join(date_time[1].split(":")[0:2])
    dt_string = date+ " " + time_str
    return dt_string

def clean_format_date_generator(d):
    date_str = d
    date_time = date_str.split()
    date = date_time[0].split("-")
    day = date[0]
    month = date[1]
    year = date[2]
    date_str = year+"-"+month+"-"+day
    dt_string = date_str+ " " + date_time[1]
    return dt_string

def extract_date(row):
    date_str = row[0]
    date_time = date_str.split()
    date = date_time[0].replace("-", "/")
    dt_string = date+ " " + date_time[1]
    date_object = datetime.strptime(dt_string, "%Y/%m/%d %H:%M")
    return date_object

def generate_date(row):
    return row.split()[0]

df_p1w['DATE_TIME_CLEAN'] = df_p1w['DATE_TIME'].apply(lambda x: clean_format_date_weather(x))

df_p1g['DATE_TIME_CLEAN'] = df_p1g['DATE_TIME'].apply(lambda x: clean_format_date_generator(x))

merged_data = pd.merge(df_p1g,df_p1w,left_on='DATE_TIME_CLEAN', right_on='DATE_TIME_CLEAN')

selected_full_data = merged_data[['DATE_TIME_CLEAN', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'SOURCE_KEY_x', 'TOTAL_YIELD']]
selected_full_data['DATE'] = selected_full_data['DATE_TIME_CLEAN'].apply(lambda x: generate_date(x))
#selected_full_data = selected_full_data.groupby(['SOURCE_KEY_x', 'DATE']).agg({'AMBIENT_TEMPERATURE':['mean'],'MODULE_TEMPERATURE':['mean'], 'TOTAL_YIELD':['mean']})
am_temp = selected_full_data.groupby(['SOURCE_KEY_x', 'DATE']).agg({'AMBIENT_TEMPERATURE':['mean']}).unstack()
mod_temp = selected_full_data.groupby(['SOURCE_KEY_x', 'DATE']).agg({'MODULE_TEMPERATURE':['mean']}).unstack()
total_yield = selected_full_data.groupby(['SOURCE_KEY_x', 'DATE']).agg({'TOTAL_YIELD':['mean']}).unstack()


am_temp.head()
mod_temp.head()
total_yield.head()
l = len(total_yield)

am_temp = np.array(am_temp)
mod_temp = np.array(mod_temp)
total_yield = np.array(total_yield)

def onehot_encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    return(res)
def extract_XY_with_date_range(num_days):
    X = []
    Y = []
    for s in range(l):
        date_list = am_temp[s]
        d_len = len(date_list)
        sel_am_temp = am_temp[s]
        sel_mod_temp = mod_temp[s]
        sel_total_yield = total_yield[s]
        for i in range(d_len):
            row = []
            row.append(source_key_list[s][0])
            for d in range (num_days):
                row.append(sel_am_temp[i-d])
                row.append(sel_mod_temp[i-d])
            X.append(row)
            Y.append(sel_total_yield[i])
    #print(X)
    X = np.array(X)
    Y = np.array(Y)
    return {"X":X, "Y":Y}
    
data_train = extract_XY_with_date_range(3)
# Encode one-hot
X0 = pd.DataFrame(data_train["X"])
X = np.array(onehot_encode_and_bind(X0, 0).drop(0,axis=1))

Y = data_train["Y"]
print(X.shape, Y.shape)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
def evaluate_model(model, X, y_true):
    y_pred = model.predict(X)
    plt.plot(y_true)
    plt.plot(y_pred)
    plt.show()
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_true, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_true, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_true, y_pred)))
    

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1) 

l_model = LinearRegression()
l_model.fit(X_train, y_train)
print("LinearRegression")
evaluate_model(l_model, X_train, y_train)
r_model = RandomForestRegressor(n_estimators=20, random_state=0, criterion="mse")
r_model.fit(X_train, y_train)
print("RandomForestRegressor")
evaluate_model(r_model, X_train, y_train)
xgbr = xgb.XGBRegressor(verbosity=0)
xgbr.fit(X_train, y_train)
print("XGBRegressor")
evaluate_model(xgbr, X_train, y_train)

data_train = extract_XY_with_date_range(7)

X0 = pd.DataFrame(data_train["X"])
X = np.array(onehot_encode_and_bind(X0, 0).drop(0,axis=1))

Y = data_train["Y"]
print(X.shape, Y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1) 
l_model = LinearRegression()
l_model.fit(X_train, y_train)
print("LinearRegression")
evaluate_model(l_model, X_train, y_train)
r_model = RandomForestRegressor(n_estimators=20, random_state=0, criterion="mse")
r_model.fit(X_train, y_train)
print("RandomForestRegressor")
evaluate_model(r_model, X_train, y_train)
xgbr = xgb.XGBRegressor(verbosity=0)
xgbr.fit(X_train, y_train)
xgbr.score(X_train, y_train)
print("XGBRegressor")
evaluate_model(xgbr, X_train, y_train)


