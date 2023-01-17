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
import seaborn as sns
import missingno as msno
import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv("/kaggle/input/Train.csv")
# Check if there are no missing values
msno.matrix(data)
data.shape
data.dtypes
data.info()
data.describe()
data.describe(include='object')
print("max date :" +data.date_time.max())
print("min date :" +data.date_time.min())
plt.figure(figsize = (8,6))
sns.countplot(y='is_holiday', data = data)
plt.show()
holidays = data.loc[data.is_holiday != 'None']
plt.figure(figsize=(4,4))
sns.countplot(y='is_holiday', data=holidays)
plt.show()
data["year"] = data["date_time"].apply(lambda x : x.split(" ")[0].split("-")[0])
data["month"] = data["date_time"].apply(lambda x : x.split(" ")[0].split("-")[1])
data["day"] = data["date_time"].apply(lambda x : x.split(" ")[0].split("-")[2])
data["time"] = data["date_time"].apply(lambda x : x.split(" ")[1].split(":")[0])
data["day_of_week"] = pd.DatetimeIndex(data["date_time"].apply(lambda x : x.split(" ")[0])).dayofweek
data.head()
data["snow_p_h"] = data["snow_p_h"].apply(lambda x : 1 if x!=0 else 0)          

data["wind_direction"] = data["wind_direction"].apply(lambda x : x//90)
data["wind_direction"] = data["wind_direction"].apply(lambda x : 0 if x == 4 else x)
data["speed_temp"] = np.sqrt(np.multiply(data["wind_speed"],data["temperature"]))
data.plot(x='air_pollution_index', y='traffic_volume', style='.', alpha=.3)
data.plot(x='humidity', y='traffic_volume', style='.', alpha=.3)
data.plot(x='wind_speed', y='traffic_volume', style='.', alpha=.3)
data.plot(x='wind_direction', y='traffic_volume', style='.', alpha=.3)
data.plot(x='visibility_in_miles', y='traffic_volume', style='.', alpha=.3)
data.plot(x='dew_point', y='traffic_volume', style='.', alpha=.3)
data.plot(x='temperature', y='traffic_volume', style='.', alpha=.3)
data.plot(x='clouds_all', y='traffic_volume', style='.', alpha=.3)
data.plot(x='snow_p_h', y='traffic_volume', style='.', alpha=.3)
for i in range(7):
    data[168*i:168*i+24].plot(x='time', y='traffic_volume', style='.', alpha=.5)
time_arr = data["time"].values
c=0
for t in range(len(time_arr)-1):
    if int(time_arr[t+1])==int(time_arr[t]):
#         print(str(t)+" "+str(time_arr[t])+" "+str(time_arr[t+1]))
        c+=1
print(c)
# taking holiday as a feature

for i in range(len(data)):
    if data.at[i,"is_holiday"] != "None":
        d = str(data.at[i,"date_time"].split(" ")[0])
        j=i
        while str(data.at[j,"date_time"].split(" ")[0])==d:
            data.at[j,"is_holiday"] = 1
            j+=1
    else: data.at[i,"is_holiday"] = 0
#adding dummy values
data = pd.get_dummies(data, columns = ["day_of_week", "month", "wind_direction"], prefix_sep='_', drop_first=True)
data.head()
data = data.drop_duplicates(subset=['date_time', 'traffic_volume'], keep="last")
# data.to_csv("mod_data.csv", index=False)
mod_data = data.drop(columns=["date_time","traffic_volume","weather_description","weather_type","dew_point","visibility_in_miles"])
mod_data.head()
X = mod_data.values
Y = data["traffic_volume"].values
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=.8, random_state=0)

from xgboost import XGBRegressor
model_xgb = XGBRegressor(
                        gamma=5, 
                        learning_rate=.3,
                        max_depth=15,
                        reg_lambda=100,
                        n_estimators = 500
                        )
                         
model_xgb.fit(X_train, Y_train,eval_metric='rmse', verbose = True, eval_set = [(X_test, Y_test)])

y_pred = model_xgb.predict(X_test)

from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
print(explained_variance_score(Y_test, y_pred))
print(mean_squared_error(Y_test, y_pred))
print(np.sqrt(mean_squared_error(Y_test, y_pred)))


test_data = pd.read_csv("/kaggle/input/Test.csv")
test_data["year"] = test_data["date_time"].apply(lambda x : x.split(" ")[0].split("-")[0])
test_data["month"] = test_data["date_time"].apply(lambda x : x.split(" ")[0].split("-")[1])
test_data["day"] = test_data["date_time"].apply(lambda x : x.split(" ")[0].split("-")[2])
test_data["time"] = test_data["date_time"].apply(lambda x : x.split(" ")[1].split(":")[0])
test_data["day_of_week"] = pd.DatetimeIndex(test_data["date_time"].apply(lambda x : x.split(" ")[0])).dayofweek


test_data["snow_p_h"] = test_data["snow_p_h"].apply(lambda x : 1 if x!=0 else 0)

test_data["wind_direction"] = test_data["wind_direction"].apply(lambda x : x//90)
test_data["wind_direction"] = test_data["wind_direction"].apply(lambda x : 0 if x == 4 else x)
test_data["speed_temp"] = np.sqrt(np.multiply(test_data["wind_speed"],test_data["temperature"]))
for i in range(len(test_data)):
    if test_data.at[i,"is_holiday"] != "None":
        d = str(test_data.at[i,"date_time"].split(" ")[0])
        j=i
        while str(test_data.at[j,"date_time"].split(" ")[0])==d:
            test_data.at[j,"is_holiday"] = 1
            j+=1
    else: test_data.at[i,"is_holiday"] = 0
test_data = pd.get_dummies(test_data, columns = ["day_of_week","month","wind_direction"], prefix_sep='_', drop_first=True)


# test_data["wind_speed_sq"] = test_data["wind_speed"].apply(lambda x:x**0.5)

test_mod_data = test_data.drop(columns=["date_time","weather_description","weather_type","dew_point","visibility_in_miles"])

test_X = test_mod_data.values
#copying data to new data frame
df_traffic_features = data.copy()
#clouds, rain and snow distribution over different weather conditions
df_traffic_features.groupby('weather_description').aggregate({'traffic_volume':[np.mean,np.size],
                                                              'clouds_all':'count','rain_p_h':'mean','snow_p_h':'mean'})
df_traffic_features['weather_description'] = df_traffic_features['weather_description'].map(lambda x:x.lower())
#Any row containing "thunderstorm" is replaced by "thunderstorm"
df_traffic_features.loc[df_traffic_features['weather_description'].str.contains('thunderstorm'),'weather_description'] = 'thunderstorm'   
weather = ['thunderstorm','mist','fog','haze']
df_traffic_features.loc[np.logical_not(df_traffic_features['weather_description'].isin(weather)),'weather_description'] = 'other'
#aggreagating traffic volume over year and plotting 

df_date_traffic = df_traffic_features.groupby('year').aggregate({'traffic_volume':'mean'})
plt.figure(figsize=(8,6))
sns.lineplot(x = df_date_traffic.index, y = df_date_traffic.traffic_volume, data = df_date_traffic)
plt.show()
df_traffic_features.weather_description.value_counts()
#creating dummy variables for these newly created categories in weather description
df_traffic_features = pd.get_dummies(columns=['weather_description'],data=df_traffic_features)
df_traffic_features.columns
pred_y = model_xgb.predict(test_X)
for i in range(len(pred_y)):
    if pred_y[i]<=0:
        sum = 0
        for j in range(10):
            sum = sum + pred_y[i-24*j]
        pred_y[i] = sum/10
submission = pd.DataFrame(columns = ["date_time","traffic_volume"])
submission["date_time"] = test_data["date_time"]
submission["traffic_volume"] = pred_y
submission.to_csv("submission.csv",index = False)