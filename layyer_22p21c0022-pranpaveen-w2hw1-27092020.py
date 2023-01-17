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
import sklearn
from matplotlib import pyplot as plt
import seaborn as sns
gen_1 = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_1_Generation_Data.csv')
gen_1.head()
gen_1.info()
gen_1.isna().sum()
gen_1["PLANT_ID"].unique()
gen_1 = gen_1.drop(columns='PLANT_ID')
gen_1["DATE_TIME"] = pd.to_datetime(gen_1["DATE_TIME"],format='%d-%m-%Y %H:%M')
gen_1 = gen_1.groupby("DATE_TIME").sum()
gen_1.head()
gen_1 = gen_1.between_time('6:00','18:30')
gen_1.head()
sns.heatmap(gen_1.corr() ,cmap='coolwarm')
gen_1.plot(y="TOTAL_YIELD")
gen_1.plot(y="DAILY_YIELD")
gen_1.loc[gen_1.index.date == pd.to_datetime('2020-06-03')].plot(y = "DAILY_YIELD")
gen_1.loc[gen_1.index.date == pd.to_datetime('2020-06-03')].plot(y = "TOTAL_YIELD")
gen_1.loc[gen_1.index.date == pd.to_datetime('2020-06-03')].plot(y = "AC_POWER")
plt.figure(figsize=(7,7))
cumsum_val = np.cumsum(gen_1.loc[gen_1.index.date == pd.to_datetime('2020-06-03')]["DC_POWER"])
plt.plot(cumsum_val/cumsum_val.max())
plt.plot(gen_1.loc[gen_1.index.date == pd.to_datetime('2020-06-03')]["DAILY_YIELD"]/gen_1.loc[gen_1.index.date == pd.to_datetime('2020-06-03')]["DAILY_YIELD"].max())
plt.legend(['DC_POWER','DAILY_YEILD'])
def find_outline(data_array):
    out_index = []
    n_array = len(data_array)
    for i in range(n_array):
        if i==0 or i==n_array-1:
            continue
        if data_array[i] < data_array[i-1]:
            out_index.append(i)
    return out_index
out_index = find_outline(gen_1["TOTAL_YIELD"].values)
out_idx = gen_1.iloc[out_index].index
for i in out_idx:
    gen_1.at[i,'TOTAL_YIELD'] = np.nan
    gen_1.at[i,'DAILY_YIELD'] = np.nan
gen_1 = gen_1.interpolate()
plt.figure(figsize=(7,7))
cumsum_val = np.cumsum(gen_1.loc[gen_1.index.date == pd.to_datetime('2020-06-03')]["DC_POWER"])
plt.plot(cumsum_val/cumsum_val.max())
plt.plot(gen_1.loc[gen_1.index.date == pd.to_datetime('2020-06-03')]["DAILY_YIELD"]/gen_1.loc[gen_1.index.date == pd.to_datetime('2020-06-03')]["DAILY_YIELD"].max())
plt.legend(['DC_POWER','DAILY_YEILD'])
weather_1 = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')
weather_1.head()
weather_1['SOURCE_KEY'].unique(),weather_1['PLANT_ID'].unique()
weather_1 = weather_1.drop(columns=['SOURCE_KEY','PLANT_ID'])
weather_1.isna().sum()
weather_1["DATE_TIME"] = pd.to_datetime(weather_1["DATE_TIME"],format='%Y-%m-%d %H:%M')
weather_1 = weather_1.set_index("DATE_TIME")
weather_1.head()
weather_1.plot(y='AMBIENT_TEMPERATURE')
weather_1.plot(y='MODULE_TEMPERATURE')
weather_1.plot(y='IRRADIATION')
weather_1 = weather_1.between_time('6:00','18:30')
weather_1.head()
data = pd.concat([gen_1,weather_1],axis=1)
data.head()
data.isna().sum()
data[data.isna().any(axis=1)]
gen_1.loc[gen_1.index.date == pd.to_datetime('2020-06-17')].head()
data = data.interpolate()
sns.heatmap(data.corr() ,cmap='coolwarm')
data.iloc[300:400].plot(y="AC_POWER")
data.iloc[300:400].plot(y="IRRADIATION")
data.iloc[300:400].plot(y="MODULE_TEMPERATURE")
data.tail()
data.sort_index()
range_between_date = pd.date_range("2020-05-15","2020-06-18",freq='15min')
range_between_date = range_between_date[range_between_date.indexer_between_time('6:00','18:30')]
time_missing = list(set(range_between_date).difference(set(data.index)))
time_missing
timedf = pd.DataFrame()
timedf['dt'] = range_between_date
timedf = timedf.set_index('dt')
timedf.head()
data = pd.concat([data,timedf],axis=1)
data = data.interpolate()
data.head()
X_raw = data.drop(columns=["AC_POWER","MODULE_TEMPERATURE","IRRADIATION","TOTAL_YIELD"]).values #ตัด feature ที่ correlate กันมากออกไป
y_raw = data["DAILY_YIELD"].values # เนื่องจาก TOTAL_YIELD เป็น cumulative sum ของ DAILY_YIELD จึงเลือกที่จะทำนาย DAILY_YIELD แทน
def create_dataset(X_raw,y_raw,feature_day,predict_day):
    n_windows = len(X_raw) - feature_day*51 +1 - predict_day*51
    x,y= [],[]
    for i in range(n_windows):
        x.append(X_raw[i : i+51*feature_day])
        y.append(y_raw[i+(51*(predict_day+feature_day))-1])
    return np.array(x),np.array(y)
X,y = create_dataset(X_raw,y_raw,3,3)
X.shape,y.shape
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
X = X.reshape(len(X),-1)
y= y.reshape(-1,1)
X.shape,y.shape
tranform = MinMaxScaler().fit(X)
tranform_y = MinMaxScaler().fit(y)

X = tranform.transform(X)
y = tranform_y.transform(y)
pca = PCA(n_components=32)
pca.fit(X,y)
X = pca.transform(X)
kf = KFold(n_splits=10)
kf.get_n_splits(X)

lr_score,xgb_score,svm_score = [],[],[]
for i,(train_index, test_index) in enumerate(kf.split(X)):
    print("Fold number ",i)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    xgb = XGBRegressor().fit(X_train,y_train)
    lr = LinearRegression().fit(X_train,y_train)
    svm = SVR().fit(X_train,y_train)
    lr_score.append(np.sqrt(mean_squared_error(tranform_y.inverse_transform(y_test),tranform_y.inverse_transform(lr.predict(X_test).reshape(-1,1)))))
    svm_score.append(np.sqrt(mean_squared_error(tranform_y.inverse_transform(y_test),tranform_y.inverse_transform(svm.predict(X_test).reshape(-1,1)))))
    xgb_score.append(np.sqrt(mean_squared_error(tranform_y.inverse_transform(y_test),tranform_y.inverse_transform(xgb.predict(X_test).reshape(-1,1)))))
    
    print("Linear Regression : rmse = ",lr_score[i])
    print("SVM regresstor : rmse = ",svm_score[i])
    print("Xgboost regresstor: rmse = ",xgb_score[i])
    print("")
    
print("Average score")
print("Linear Regression : rmse = ",np.average(lr_score))
print("SVM regresstor : rmse = ",np.average(svm_score))
print("Xgboost regresstor: rmse = ",np.average(xgb_score))
plt.figure(figsize=(15,7))
plt.plot(tranform_y.inverse_transform(y))
plt.plot(tranform_y.inverse_transform(lr.predict(X)).reshape(-1,1))
X,y = create_dataset(X_raw,y_raw,3,7)
X = X.reshape(len(X),-1)
y= y.reshape(-1,1)
X.shape,y.shape
tranform = MinMaxScaler().fit(X)
tranform_y = MinMaxScaler().fit(y)

X = tranform.transform(X)
y = tranform_y.transform(y)
pca = PCA(n_components=32)
pca.fit(X,y)
X = pca.transform(X)
kf = KFold(n_splits=10)
kf.get_n_splits(X)

lr_score7,xgb_score7,svm_score7 = [],[],[]
for i,(train_index, test_index) in enumerate(kf.split(X)):
    print("Fold number ",i)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    xgb7 = XGBRegressor().fit(X_train,y_train)
    lr7 = LinearRegression().fit(X_train,y_train)
    svm7 = SVR().fit(X_train,y_train)
    lr_score7.append(np.sqrt(mean_squared_error(tranform_y.inverse_transform(y_test),tranform_y.inverse_transform(lr7.predict(X_test).reshape(-1,1)))))
    svm_score7.append(np.sqrt(mean_squared_error(tranform_y.inverse_transform(y_test),tranform_y.inverse_transform(svm7.predict(X_test).reshape(-1,1)))))
    xgb_score7.append(np.sqrt(mean_squared_error(tranform_y.inverse_transform(y_test),tranform_y.inverse_transform(xgb7.predict(X_test).reshape(-1,1)))))
    
    print("Linear Regression : rmse = ",lr_score7[i])
    print("SVM regresstor : rmse = ",svm_score7[i])
    print("Xgboost regresstor: rmse = ",xgb_score7[i])
    print("")
    
print("Average score")
print("Linear Regression : rmse = ",np.average(lr_score7))
print("SVM regresstor : rmse = ",np.average(svm_score7))
print("Xgboost regresstor: rmse = ",np.average(xgb_score7))
plt.figure(figsize=(15,7))
plt.plot(tranform_y.inverse_transform(y))
plt.plot(tranform_y.inverse_transform(lr7.predict(X)).reshape(-1,1))
gen_2 = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_2_Generation_Data.csv')
weather_2 = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv')
gen_2.head()
gen_2 = gen_2.drop(columns='PLANT_ID')
gen_2["DATE_TIME"] = pd.to_datetime(gen_2["DATE_TIME"],format='%Y-%m-%d %H:%M')
gen_2 = gen_2.groupby("DATE_TIME").sum()
gen_2 = gen_2.between_time('6:00','18:30')
out_idx = gen_2.iloc[out_index].index
for i in out_idx:
    gen_2.at[i,'TOTAL_YIELD'] = np.nan
    gen_2.at[i,'DAILY_YIELD'] = np.nan
gen_2 = gen_2.interpolate()
weather_2['SOURCE_KEY'].unique(),weather_2['PLANT_ID'].unique()
weather_2 = weather_2.drop(columns=['SOURCE_KEY','PLANT_ID'])
weather_2["DATE_TIME"] = pd.to_datetime(weather_2["DATE_TIME"],format='%Y-%m-%d %H:%M')
weather_2 = weather_2.set_index("DATE_TIME")
weather_2 = weather_2.between_time('6:00','18:30')
weather_2.head()
data_2 = pd.concat([gen_2,weather_2],axis=1)
data_2.head()
range_between_date = pd.date_range("2020-05-15","2020-06-17",freq='15min')
range_between_date = range_between_date[range_between_date.indexer_between_time('6:00','18:30')]
time_missing = list(set(range_between_date).difference(set(data_2.index)))
timedf = pd.DataFrame()
timedf['dt'] = range_between_date
timedf = timedf.set_index('dt')
data_2 = pd.concat([data_2,timedf],axis=1)
data_2 = data_2.interpolate()
data_2.head()
X_raw2 = data_2.drop(columns=["AC_POWER","MODULE_TEMPERATURE","IRRADIATION","TOTAL_YIELD"]).values
y_raw2 = data_2["DAILY_YIELD"].values
X2,y2 = create_dataset(X_raw2,y_raw2,3,3)
X2 = X2.reshape(len(X2),-1)
y2 = y2.reshape(-1,1)
X2 = tranform.transform(X2)
y2 = tranform_y.transform(y2)
X2 = pca.transform(X2)
plt.figure(figsize=(7,7))
plt.plot(y2)
plt.plot(lr.predict(X2))
