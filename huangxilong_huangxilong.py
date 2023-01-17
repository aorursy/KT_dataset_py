#导入常见的库
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
#导入数据
data=pd.read_csv('/kaggle/input/bnu-esl-2020/train.csv')
test=pd.read_csv('/kaggle/input/bnu-esl-2020/test.csv')
data.shape
test.shape
data.head()
test.head()
data.describe()
plt.figure(figsize=(16,6))
sns.heatmap(data.isnull(),cmap='viridis')
plt.figure(figsize=(16,6))
sns.heatmap(test.isnull(),cmap='viridis')
_des=data.describe()
data.total_bedrooms.fillna(_des['total_bedrooms']['50%'],inplace=True)
plt.figure(figsize=(16,6))
sns.heatmap(data.isnull(),cmap='viridis')
data.info()
#先copy一份，并去掉ID字段
df=data.copy()

df.drop('Unnamed: 0',axis=1,inplace=True)
df.head()
sns.set()
plt.figure(figsize=(10,8))
plt.scatter('longitude','latitude',data=df,alpha=0.1)
plt.ylabel('Latitude')
plt.xlabel('Longitude')
plt.show()
plt.figure(figsize=(10,7))
df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
        s=df["population"]/100, label="population", figsize=(15,8),
        c="median_house_value", cmap=plt.get_cmap("jet"),colorbar=True,
    )
plt.legend()
#查看数据之间的散点图
from pandas.plotting import scatter_matrix

sns.set()
feat = ['median_house_value','median_income','total_rooms','housing_median_age','population','total_bedrooms','households']
scatter_matrix(df[feat],figsize=(15,8))
df.hist(bins=70, figsize=(20,20))
plt.show()
corr = df.corr()
plt.figure(figsize=(8, 8))
sns.heatmap(corr, cmap='viridis')

corr.median_house_value.sort_values(ascending=False)
print(corr.households.total_rooms)
print(corr.households.total_bedrooms)
print(corr.total_bedrooms.total_rooms)
print(corr.households.population)
df['average_rooms']=df.total_rooms/df.population
df['average_bedrooms']=df.total_bedrooms/df.population
_features=['households','total_bedrooms','total_rooms']
df.drop(_features,axis=1,inplace=True)
df.head()
corr = df.corr()
plt.figure(figsize=(8, 8))
sns.heatmap(corr, cmap='viridis')

corr.median_house_value.sort_values(ascending=False)
corr.average_bedrooms.average_rooms
df.ocean_proximity.replace({'<1H OCEAN':1.0,'INLAND':2.0,'ISLAND':3.0,'NEAR BAY':4.0,'NEAR OCEAN':5.0},inplace = True)
df.ocean_proximity.head()
df=df.loc[df['median_house_value']<500001,:]
# 分离自变量和因变量

tmpX=df.drop('median_house_value',axis=1)
tmpY=df.median_house_value

# 分离训练集和验证集

from sklearn.model_selection import train_test_split

trainX,testX,trainY,testY=train_test_split(tmpX,tmpY,test_size=0.1,random_state=1)
trainX.shape
trainY.shape
testX.shape
import xgboost as xgb
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 1,eta=0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 2000)
xg_reg.fit(trainX,trainY)

predY = xg_reg.predict(testX)
#计算在验证集上的得分
from sklearn.metrics import r2_score
r2xgb=r2_score(testY,predY)
print('the R squared of the xgboost method is:', r2xgb)
# 对测试集作一定的处理

#去掉ID 
test.drop('id',axis=1,inplace=True)

#填充缺失数据

_des=test.describe()
test.total_bedrooms.fillna(_des['total_bedrooms']['50%'],inplace=True)


#生成新的特征

test['average_rooms']=test.total_rooms/test.population
test['average_bedrooms']=test.total_bedrooms/test.population

#去掉旧的特征
test.drop(['total_bedrooms','total_rooms','households'],axis=1,inplace=True)

#生成虚拟变量
test.ocean_proximity.replace({'<1H OCEAN':1.0,'INLAND':2.0,'ISLAND':3.0,'NEAR BAY':4.0,'NEAR OCEAN':5.0},inplace = True)

test.head()
resY = xg_reg.predict(test)
output=pd.read_csv('/kaggle/input/bnu-esl-2020/test.csv')
_features=['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income','ocean_proximity']
output.insert(1,'predicted',resY)
output.drop(_features, axis = 1,inplace = True)
output.to_csv("predict.csv",index=False)




