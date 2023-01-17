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
import matplotlib.pyplot as plt
Plant1_gen = pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')
Plant1_wea = pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')
Plant2_gen = pd.read_csv('../input/solar-power-generation-data/Plant_2_Generation_Data.csv')
Plant2_wea = pd.read_csv('../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv')

Plant_id = Plant1_gen[["PLANT_ID","TOTAL_YIELD"]].groupby("PLANT_ID").count()
print(Plant_id)
Plant_id = Plant1_wea[["PLANT_ID","DATE_TIME"]].groupby("PLANT_ID").count()
print(Plant_id)
sou=Plant1_wea[["SOURCE_KEY","DATE_TIME"]].groupby("SOURCE_KEY").count()
print(sou)
Plant1_gen=Plant1_gen.drop(["PLANT_ID"],axis=1)
Plant1_wea=Plant1_wea.drop(["PLANT_ID"],axis=1)
Plant1_wea=Plant1_wea.drop(["SOURCE_KEY"],axis=1)
Plant1_gen["DATE_TIME"] = pd.to_datetime(Plant1_gen["DATE_TIME"])
Plant1_wea["DATE_TIME"] = pd.to_datetime(Plant1_wea["DATE_TIME"])
Plant = pd.merge(Plant1_wea,Plant1_gen, on="DATE_TIME", how="inner")
Plant.head()

sourceKey = Plant[["TOTAL_YIELD","SOURCE_KEY"]].groupby("SOURCE_KEY").count().index
#print(type(sourceKey))
print(sourceKey)
sourceKey.to_numpy()
n=1
for i in sourceKey:
    ind = Plant[Plant['SOURCE_KEY']==i]
    globals()['inv_%s' % n] =  ind
    #print(i)
    n=n+1

inv_1
for i in range(1,22):   
    plt.plot( globals()['inv_%s' % i]["DAILY_YIELD"])
plt.show()
for i in range(1,22):   
    plt.plot( globals()['inv_%s' % i]["TOTAL_YIELD"])
plt.show()
plt.plot(Plant.TOTAL_YIELD)
plt.show()
plt.plot(Plant.AC_POWER)
plt.show()

plt.plot(Plant.DC_POWER)
plt.plot(Plant.DAILY_YIELD)
from datetime import datetime 

dayList =[]
YearList = []
dates = Plant['DATE_TIME']
for i in dates:
    a,b = i.strftime('%Y-%m-%d %H:%M:%S').split(' ')
    _,M,D = a.split('-')
    H,m,_ = b.split(':')
    Day_IN_Y = (M+':'+D)
    Time_at_D = (int(H)*60)+int(m)
    dayList.append(Time_at_D)
    YearList.append(Day_IN_Y)
Plant['Time_at_D']=dayList
Plant['Day_IN_Y']=YearList
Plant
#10 fold
n=10
#print(Plant.shape)
index,_ = Plant.shape
#e =index%10 if have to balanc fold
#index=index -e
#print(index-e)
listn = {}
lists = []
for i in range(n):
    lists=[]
    for j in range(i,index,n):
        lists.append(j)

    globals()['Fold_%s' % i] =  Plant.iloc[lists]
    #name='Fold'+str(i+1)
    #listn[name]=lists

Fold_1
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
import time
from sklearn import metrics
feature_cols =['AC_POWER','DC_POWER','Time_at_D','AMBIENT_TEMPERATURE','MODULE_TEMPERATURE']

Fold_Container = ['Fold_0','Fold_1','Fold_2','Fold_3','Fold_4','Fold_5','Fold_6','Fold_7','Fold_8','Fold_9']
def testModel(typeM,ModelName):
    MAE = []
    MSE = []
    RMSE = []
    R2 = []
    times = []
    for i in Fold_Container:
        start = time.time()#####################################
        X=globals()['%s' % i][feature_cols]
        Y=globals()['%s' % i]["DAILY_YIELD"]
        model = typeM.fit(X, Y)
        end = time.time()######################################
        for j in Fold_Container:
            if i==j:
                continue
            XT=globals()['%s' % j][feature_cols]
            YT=globals()['%s' % j]["DAILY_YIELD"]
            ans =model.predict(XT)
            mae = metrics.mean_absolute_error(YT, ans)
            MAE.append(mae)
            mse = metrics.mean_squared_error(YT, ans)
            MSE.append(mse)
            rmse = np.sqrt(metrics.mean_squared_error(YT, ans))
            RMSE.append(rmse)
            r2 = LR.score(X, YT)
            R2.append(r2)
            times.append(end - start)
            #print(str(i)+' test with '+str(j)+'...............................................>>>>>>>>>>>>>>>>>>>')
            #print('R-Squared (R2):', r2)
            #print('Root Mean Squared Error (RMSE):', rmse)
            #print('Mean Squared Error (MSE):', mse)
            #print('Mean Absolute Error (MAE):', mae)
            #print('Time: ',end - start)
            #print('')    
    print(ModelName)
    print('<<<<_______________________________________________________>>>>')
    print('Mean Absolute Error (MAE):', np.mean(MAE))
    print('Mean Squared Error (MSE):', np.mean(MSE))
    print('Root Mean Squared Error (RMSE):', np.mean(RMSE))
    print('R-Squared (R2):', np.mean(R2))
    print('Time :', np.mean(times))
    print('<<<<_______________________________________________________>>>>')
    print('\n\n')
LR = LinearRegression()
testModel(LR,'LinearRegression')
RF = RandomForestRegressor(max_depth=2, random_state=0)
testModel(RF,'RandomForestRegressor')
CB = CatBoostRegressor(iterations=50,learning_rate=0.16,depth=4,verbose=0)
testModel(CB,'CatBoostRegressor')