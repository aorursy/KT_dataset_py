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
import numpy as np
import pandas as pd
plant1G= pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')
plant1G.head()
plant1G.info()
dateTime = plant1G[["DATE_TIME", "TOTAL_YIELD"]].groupby("DATE_TIME").count()
print("A number of dateTime : ",len(dateTime))
dateTime.head()
plantID = plant1G[["PLANT_ID", "TOTAL_YIELD"]].groupby("PLANT_ID").count()
print("A number of plantID : ", len(plantID))
plantID
sourceKey = plant1G[["SOURCE_KEY", "TOTAL_YIELD"]].groupby("SOURCE_KEY").count()
print("A number of sourceKey : ",len(sourceKey))
sourceKey
df1 = plant1G.drop('PLANT_ID', axis='columns')
plant1W = pd.read_csv("../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv")
plant1W.head()
plant1W.info()
dateTime = plant1W[["DATE_TIME","AMBIENT_TEMPERATURE"]].groupby("DATE_TIME").count()
print("A number of dateTime : ",len(dateTime))
dateTime.head()
sourceKey = plant1W[["SOURCE_KEY", "AMBIENT_TEMPERATURE"]].groupby("SOURCE_KEY").count()
print("A number of sourceKey : ",len(sourceKey))
sourceKey
df2 = plant1W.drop(['PLANT_ID', 'SOURCE_KEY'], axis="columns")
df1["DATE_TIME"] = pd.to_datetime(df1["DATE_TIME"])
df2["DATE_TIME"] = pd.to_datetime(df2["DATE_TIME"])
df = pd.merge(df2,df1, on="DATE_TIME", how="inner")
df
import matplotlib.pyplot as plt
SK = df[["SOURCE_KEY"]].groupby("SOURCE_KEY").count().index
plt.ylim(150000,250000)
plt.xlim(1400,2100)
plt.title("SOURCE_KEY yield total")
plt.xlabel("time units")
plt.ylabel("total yield")
initial = {}
for i in SK:
    a = df[df["SOURCE_KEY"]==i].reset_index()
    plt.plot(a["TOTAL_YIELD"] - a["TOTAL_YIELD"][0])
    initial[i] = a["TOTAL_YIELD"][0]
plt.plot([1400,2100],[188000,235000],c="black",linestyle="dashed")
print(df[["SOURCE_KEY"]].groupby("SOURCE_KEY").count().index[0])
print(df[["SOURCE_KEY"]].groupby("SOURCE_KEY").count().index[11])
df['growth_yield'] = df['TOTAL_YIELD'] - df['SOURCE_KEY'].apply(lambda v: initial[v])
df.drop("TOTAL_YIELD", axis=1,inplace = True)
df.head()
past = 3
data = df.copy()
date = df['DATE_TIME']
for t in range(past):
    date = date + np.timedelta64(1,'D')
    p = data.copy()
    p['DATE_TIME'] = date
    data = pd.merge(data,p, on=["DATE_TIME", "SOURCE_KEY"], how="left",suffixes=("_present","_past"))
data.head()
data.isna().sum()
data = data.fillna(0)
data.isna().sum()
data['SOURCE_KEY'] = ['SK_low' if ((s =="1BY6WEcLGh8j5v7") or (s == "bvBOhCH3iADSZry")) else 'SK_high' for s in data['SOURCE_KEY'] ]
data = pd.concat([data, pd.get_dummies(data['SOURCE_KEY'])], axis = "columns")
data.drop("SOURCE_KEY",axis=1, inplace=True)
data.head()
tenFold = []
labels = []
for k in range(10):
    m = []  
    y = []
    for v in data.loc[range(k,len(df),10)].to_numpy():
        m.append(np.append(v[1:7],v[8:]))
        y.append(v[7])
    tenFold.append(m)
    labels.append(y)
tenFold = np.array(tenFold)
labels = np.array(labels)
print("           (fold, data, features)\n tenFold: ",tenFold.shape)
print(" labels:  ", labels.shape)
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.linear_model import LinearRegression
import time
from sklearn import metrics
RFR = RandomForestRegressor(max_depth=2, random_state=0)
XGBR = xgb.XGBRegressor(n_estimators=50,
                       objective ='reg:squarederror',
                       learning_rate = 0.16,
                       colsample_bytree=0.6,
                       max_depth = 4,
                       min_child_weight = 6)
LR = LinearRegression()
def preprocessing(WS,GD, past):
    pWS = WS.copy()
    pGD = GD.copy()
    pGD = pGD.drop('PLANT_ID', axis='columns')
    pWS.drop(['PLANT_ID', 'SOURCE_KEY'], axis="columns", inplace=True)
    pGD["DATE_TIME"] = pd.to_datetime(pGD["DATE_TIME"])
    pWS["DATE_TIME"] = pd.to_datetime(pWS["DATE_TIME"])
    df = pd.merge(pWS,pGD, on="DATE_TIME", how="inner")
    SK = df[["SOURCE_KEY"]].groupby("SOURCE_KEY").count().index
    initial = {}
    for i in SK:
        a = df[df["SOURCE_KEY"]==i].reset_index()
        initial[i] = a["TOTAL_YIELD"][0]
    df['growth_yield'] = df['TOTAL_YIELD'] - df['SOURCE_KEY'].apply(lambda v: initial[v])
    df.drop("TOTAL_YIELD", axis=1,inplace = True)
    data = df.copy()
    date = df['DATE_TIME']
    p = data.copy()
    for t in range(past):
        date = date + np.timedelta64(1,'D')
        p['DATE_TIME'] = date
        data = pd.merge(data,p, on=["DATE_TIME", "SOURCE_KEY"], how="left",suffixes=("_present","_past"))
    data = data.fillna(0)
    data['SOURCE_KEY'] = ['SK_low' if ((s =="1BY6WEcLGh8j5v7") or (s == "bvBOhCH3iADSZry")) else 'SK_high' for s in data['SOURCE_KEY'] ]
    data = pd.concat([data, pd.get_dummies(data['SOURCE_KEY'])], axis = "columns")
    data.drop("SOURCE_KEY",axis=1, inplace=True)
    return data

def KFold(data, K=10):
    tenFold = []
    labels = []
    for k in range(K):
        m = []  
        y = []
        for v in data.loc[range(k,len(data),K)].to_numpy():
            m.append(np.append(v[1:7],v[8:]))
            y.append(v[7])
        tenFold.append(m)
        labels.append(y)
    tenFold = np.array(tenFold)
    labels = np.array(labels)
    return tenFold, labels

def evaluate(tenFold, labels, model):
    clf = model
    MAE = []
    MSE = []
    RMSE = []
    R2 = []
    times = []
    for i in range(len(tenFold)):
        start = time.time()
        for id, f in enumerate(tenFold):
          if id == i:
            continue
          clf.fit(np.array(f),np.array(labels[id]))
        end = time.time()
        times.append(end - start)
        pred = clf.predict(np.array(tenFold[i]))
        gt = labels[i]
        print(f"Fold #{i+1}---------------------------------------")
        mae = metrics.mean_absolute_error(gt, pred)
        MAE.append(mae)
        print('Mean Absolute Error (MAE):', mae)
        mse = metrics.mean_squared_error(gt, pred)
        MSE.append(mse)
        print('Mean Squared Error (MSE):', mse)
        rmse = np.sqrt(metrics.mean_squared_error(gt, pred))
        RMSE.append(rmse)
        print('Root Mean Squared Error (RMSE):', rmse)
        r2 = clf.score(np.array(tenFold[i]), np.array(labels[i]))
        R2.append(r2)
        print('R-Squared (R2):', r2)
        print('Time: ',end - start)
        print('')
        
    print('\n\n')
    print('AVG*********************************************')
    print('Mean Absolute Error (MAE):', np.mean(MAE))
    print('Mean Squared Error (MSE):', np.mean(MSE))
    print('Root Mean Squared Error (RMSE):', np.mean(RMSE))
    print('R-Squared (R2):', np.mean(R2))
    print('Time :', np.mean(times))
    print('************************************************')
    return np.mean(MAE),np.mean(MSE),np.mean(RMSE),np.mean(R2),np.mean(times)
def pipeline(WS,GD,model,past=3):
    
    data = preprocessing(WS,GD,past)
    
    tenFold, labels = KFold(data, 10)

    mae, mse, rmse, r2, times = evaluate(tenFold, labels, model)
    mae = np.round(mae,2) 
    mse = np.round(mse,2) 
    rmse = np.round(rmse,2) 
    r2 = np.round(r2,4) 
    times = np.round(times,4)
    return (mae, mse, rmse, r2, times)
data = preprocessing(plant1W,plant1G,past=3) #ใช้ข้อมูล 3 วันก่อนหน้า
tenFold, labels = KFold(data, 10)
def evaluateViz(tenFold, labels, model):
    MAE = []
    RMSE = []
    R2 = []
    fold=[]
    times = []
    for i in range(len(tenFold)):
        start = time.time()
        for id, f in enumerate(tenFold):
            if id == i:
                continue
            model.fit(np.array(f),np.array(labels[id]))
        end = time.time()
        times.append(end - start)
        pred = model.predict(np.array(tenFold[i]))
        gt = labels[i]
        fold.append(i+1)
    
        mae = metrics.mean_absolute_error(gt, pred)
        MAE.append(mae)
    
        rmse = np.sqrt(metrics.mean_squared_error(gt, pred))
        RMSE.append(rmse)
    
        r2 = LR.score(np.array(tenFold[i]), np.array(labels[i]))
        R2.append(r2)
        print('Time: ',end - start)
        
    plt.figure(figsize=(20, 3))

    plt.subplot(131)
    plt.plot(fold, RMSE)
    plt.title('Root Mean Squared Error (RMSE)')

    plt.subplot(132)
    plt.plot(fold, MAE)
    plt.title('Mean Absolute Error (MAE)')

    plt.subplot(133)
    plt.plot(fold, R2)
    plt.title('R-Squared (R2)')
    
    plt.show()

evaluateViz(tenFold, labels, LR) #แสดงกราฟ Linear Regression
evaluateViz(tenFold, labels, RFR) #แสดงกราฟ Random Forest Regression
evaluateViz(tenFold, labels, XGBR) #แสดงกราฟ XGB Regression
data = preprocessing(plant1W,plant1G,past=7)#ใช้ข้อมูล 7 วันก่อนหน้า
tenFold, labels = KFold(data, 10)
evaluateViz(tenFold, labels, LR) #แสดงกราฟ Linear Regression
evaluateViz(tenFold, labels, RFR) #แสดงกราฟ Random Forest Regression
evaluateViz(tenFold, labels, XGBR) #แสดงกราฟ XGB Regression
table1 = {}
table1["LR"] = list(pipeline(plant1W,plant1G,LR))
table1["RFR"] = list(pipeline(plant1W,plant1G,RFR))
table1["XGBR"] = list(pipeline(plant1W,plant1G,XGBR)) #แสดงผลลัพธ์โดยเฉลี่ย 3 วันก่อนหน้า
pd.DataFrame(table1).rename(index={0:"MAE",1:"MSE",2:"RMSE",3:"R2",4:"Time"}).T #แสดงตารางผลลัพธ์โดยเฉลี่ย 3 วันก่อนหน้า
table2 = {}
table2["LR"] = list(pipeline(plant1W,plant1G,LR,7))
table2["RFR"] = list(pipeline(plant1W,plant1G,RFR,7))
table2["XGBR"] = list(pipeline(plant1W,plant1G,XGBR,7)) #แสดงผลลัพธ์โดยเฉลี่ย 7 วันก่อนหน้า
pd.DataFrame(table2).rename(index={0:"MAE",1:"MSE",2:"RMSE",3:"R2",4:"Time"}).T #แสดงตารางผลลัพธ์โดยเฉลี่ย 3 วันก่อนหน้า
