import pandas as pd
import numpy as np
import seaborn as sns
import xgboost as xgb
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

solar1 = pd.read_csv("/kaggle/input/solar-power-generation-data/Plant_1_Generation_Data.csv")
sensor1 = pd.read_csv("/kaggle/input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv")

solar2 = pd.read_csv("/kaggle/input/solar-power-generation-data/Plant_2_Generation_Data.csv")
sensor2 = pd.read_csv("/kaggle/input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv")
solar1.head()
sensor1.head()
solar2.head()
sensor2.head()
solar1.info()
solar1.describe()
solar1.isna().sum()
solar2.isna().sum()
print ('Plant 1 has '+ str(solar1['SOURCE_KEY'].nunique()) + ' inverters')
print ('Sensor 1 has '+ str(sensor1['SOURCE_KEY'].nunique()) + ' inverters')
print ('Plant 2 has '+ str(solar2['SOURCE_KEY'].nunique()) + ' inverters')
print ('Sensor 2 has '+ str(sensor2['SOURCE_KEY'].nunique()) + ' inverters')
solar1["DATE_TIME"] = pd.to_datetime(solar1["DATE_TIME"], format="%d-%m-%Y %H:%M")
sensor1["DATE_TIME"] = pd.to_datetime(sensor1["DATE_TIME"], format="%Y-%m-%d %H:%M:%S")
solar2["DATE_TIME"] = pd.to_datetime(solar2["DATE_TIME"], format="%Y-%m-%d %H:%M:%S")
sensor2["DATE_TIME"] = pd.to_datetime(sensor2["DATE_TIME"], format="%Y-%m-%d %H:%M:%S")

solar1["DATE"] = solar1["DATE_TIME"].dt.date
solar1["TIME"] = solar1["DATE_TIME"].dt.time
solar2["DATE"] = solar2["DATE_TIME"].dt.date
solar2["TIME"] = solar2["DATE_TIME"].dt.time
solar1
solar1.drop("PLANT_ID", axis=1, inplace=True)
sensor1.drop(["PLANT_ID","SOURCE_KEY"], axis=1, inplace=True)
solar2.drop("PLANT_ID", axis=1, inplace=True)
sensor2.drop(["PLANT_ID","SOURCE_KEY"], axis=1, inplace=True)

solar1_id = solar1['SOURCE_KEY'].unique()
solar1["SOURCE_KEY"] = solar1["SOURCE_KEY"].apply(lambda x: int(np.where(solar1_id == x)[0]))
solar2_id = solar2['SOURCE_KEY'].unique()
solar2["SOURCE_KEY"] = solar2["SOURCE_KEY"].apply(lambda x: int(np.where(solar2_id == x)[0]))
solar1
solar2
sensor2
plant1 = pd.merge(solar1,sensor1,on='DATE_TIME')
plant2 = pd.merge(solar2,sensor2,on='DATE_TIME')
plant1
plant2
plant1[:10]
data = plant1[(plant1['SOURCE_KEY']==0) & (plant1['DATE_TIME'].between('2020-05-15','2020-05-21'))]
data['TIME'] = data['TIME'].astype(str)
g = sns.relplot(
        data=data,
        x='TIME',
        y='DC_POWER',
        row='DATE',
        kind='line',
        height=2,
        aspect=6)

g.set(xlim=('00:00:00', '23:45:00'), xticks=['00:00:00','06:00:00','12:00:00','18:00:00','23:45:00'])
fulltime = pd.date_range(start='2020-05-15 00:00',end='2020-06-17 23:45' , freq='15T')
fulltime = pd.DataFrame({'DATE_TIME':fulltime})
solar01_inv_0 = plant1[plant1['SOURCE_KEY']==0].reset_index(drop=True)
solar01_inv_0 = pd.merge(fulltime, solar01_inv_0, how="outer")
solar01_inv_0
solar01_inv_0.index = solar01_inv_0['DATE_TIME']
solar01_inv_0.drop('DATE_TIME', axis=1, inplace=True)
morning       = solar01_inv_0.between_time('00:00:00','05:45:00')
afternoon     = solar01_inv_0.between_time('06:00:00','18:30:00')
night         = solar01_inv_0.between_time('18:45:00','23:45:00')
morning['DC_POWER'].fillna(value=0, inplace=True)
morning['AC_POWER'].fillna(value=0, inplace=True)
morning['DAILY_YIELD'].fillna(value =0, inplace=True)
night['DC_POWER'].fillna(value=0, inplace=True)
night['AC_POWER'].fillna(value=0, inplace=True)
night['DAILY_YIELD'].fillna(value =0, inplace=True)
solar01_inv_0 = pd.concat([morning,afternoon, night])
solar01_inv_0 = solar01_inv_0.sort_index()
data = solar01_inv_0
data['TIME'] = data['TIME'].astype(str)
sns.set(font_scale =1.5)

g = sns.relplot(
        data=data,
        x='TIME',
        y='DAILY_YIELD',
        col='DATE',
        kind='scatter',
        height=2,
        aspect=3,
        col_wrap=3
        )

g.set(xlim=('00:00:00', '23:45:00'), xticks=['00:00:00','06:00:00','12:00:00','18:00:00','23:45:00'])
'''
kfold = 10
day3_tenfold = []
day3_labels = []

for k in range(kfold):
    x = []
    y = []
    for value in data.loc[range(k,len(data),kfold)].to_numpy():
        x.append(value[0:6],value[7:])
        y.append(value[6])
    day3_tenfold.append(x)
    day3_labels.append(y)
    
day3_tenfold = np.array(day3_tenfold)
day3_labels = np.array(day3_labels)
print(day3_tenfold.shape)
print(day3_labels.shape)
'''
avg_plant_1 = []

for i in range(0,len(plant1),plant1['SOURCE_KEY'].nunique()):
    avg_plant_1.append([plant1["DC_POWER"][i:i+22].sum()/22, plant1["AC_POWER"][i:i+22].sum()/22, plant1["DAILY_YIELD"][i:i+22].sum()/22,plant1["TOTAL_YIELD"][i:i+22].sum()/22])
    
avg_plant_1 = np.asarray(avg_plant_1)
print(avg_plant_1.shape)
avg = {'DC_POWER' : [],'AC_POWER' : [],'DAILY_YIELD' : [],'TOTAL_YIELD' : []}
for i in range(0,len(plant1),plant1['SOURCE_KEY'].nunique()):
    avg["DC_POWER"].append(plant1["DC_POWER"][i:i+22].sum()/22)
    avg["AC_POWER"].append(plant1["AC_POWER"][i:i+22].sum()/22)
    avg["DAILY_YIELD"].append(plant1["DAILY_YIELD"][i:i+22].sum()/22)
    avg["TOTAL_YIELD"].append(plant1["TOTAL_YIELD"][i:i+22].sum()/22)
    
print(avg)
solar01_inv_0
plantx = pd.merge(plant1, fulltime, how="right")
print(plantx[1958:2000])
plantx = plantx.fillna(value=0)
plantx[1958:]
label_encoder = preprocessing.LabelEncoder() 
# plantx['DATE']= label_encoder.fit_transform(plantx['DATE'])
X = plantx[["DC_POWER","AC_POWER","DAILY_YIELD","TOTAL_YIELD","AMBIENT_TEMPERATURE","MODULE_TEMPERATURE"]][:-2112]
y = plantx["TOTAL_YIELD"][2112:]
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.1)
XGB = xgb.XGBRegressor(n_estimators=1000,
                       objective ='reg:squarederror',
                       learning_rate = 0.3,
                       colsample_bytree=0.9,
                       max_depth = 10,
                       min_child_weight = 6)
XGB.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds=50)
label_encoder = preprocessing.LabelEncoder() 
# plantx['DATE']= label_encoder.fit_transform(plantx['DATE'])
X = plantx[["DC_POWER","AC_POWER","DAILY_YIELD","TOTAL_YIELD","AMBIENT_TEMPERATURE","MODULE_TEMPERATURE"]][:-4928]
y = plantx["TOTAL_YIELD"][4928:]
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.1)
XGB = xgb.XGBRegressor(n_estimators=1000,
                       objective ='reg:squarederror',
                       learning_rate = 0.4,
                       colsample_bytree=0.9,
                       max_depth = 10,
                       min_child_weight = 6)
XGB.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds=50)
avg = {'DC_POWER' : [],'AC_POWER' : [],'DAILY_YIELD' : [],'TOTAL_YIELD' : []}
for i in range(0,len(plant1),plant1['SOURCE_KEY'].nunique()):
    avg["DC_POWER"].append(plant1["DC_POWER"][i:i+22].sum()/22)
    avg["AC_POWER"].append(plant1["AC_POWER"][i:i+22].sum()/22)
    avg["DAILY_YIELD"].append(plant1["DAILY_YIELD"][i:i+22].sum()/22)
    avg["TOTAL_YIELD"].append(plant1["TOTAL_YIELD"][i:i+22].sum()/22)
    
print(avg)
tmp_data = []
tmp_label = []

for i in range(len(avg["DC_POWER"])-(96*3)):
    tmp_data.append(avg["DAILY_YIELD"][i])
    tmp_label.append(avg["TOTAL_YIELD"][i+96*3])

tmp_data = np.asarray(tmp_data)
tmp_label = np.asarray(tmp_label)
    
label_encoder = preprocessing.LabelEncoder() 
X = tmp_data
y = tmp_label
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.1)
print(X_train.shape, X_test.shape)
XGB = xgb.XGBRegressor(n_estimators=1000,
                       objective ='reg:squarederror',
                       learning_rate = 0.4,
                       colsample_bytree=0.9,
                       max_depth = 10,
                       min_child_weight = 6)
XGB.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds=50)