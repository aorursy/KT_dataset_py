import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
import random
data= pd.read_csv("../input/train_V2.csv")
#EXPLORATORY DATA ANALYSIS

data.head()
#Details of the data type
data.info()
data["groupId"].value_counts()
data["matchId"].value_counts()
#Dimension of the data
data.shape
df=pd.DataFrame()
df[["groupId","matchId", "winPlacePerc"]]= pd.DataFrame(data[["groupId","matchId", "winPlacePerc"]].copy())

#Converting strings to categorical codes
df["groupId"]= df["groupId"].astype('category').cat.codes
df["matchId"]= df["matchId"].astype('category').cat.codes
df["winPlacePerc"]= df["winPlacePerc"].astype('category').cat.codes

#Check Correlation
df.corr()

#The correlation matrix does not show any relation
list(data.columns.values)
data=data.drop(data[["Id", "groupId", "matchId"]], axis=1)
data.columns.values
data.head()
cor= data.corr()
cor.style.background_gradient().set_precision(2)
data['WinPlaceBucket'] = np.where(data.winPlacePerc >0.75, 'Quartile1', 
                          np.where(data.winPlacePerc>.5, 'Quartile2', 
                                   np.where(data.winPlacePerc>.25, 'Quartile3', 'Quartile4')))
data[["winPlacePerc","WinPlaceBucket"]]
sns.pairplot(data.sample(100), size = 2.5) #hue="WinPlaceBucket")
data.columns.values
col=["boosts", "damageDealt", "kills", "killStreaks", "walkDistance", "weaponsAcquired","winPlacePerc"]
sns.pairplot(data[col].sample(100000), size = 2.5, kind="reg", diag_kind="kde")
sns.catplot(x="WinPlaceBucket", kind="count", palette="ch:.25", data=data.sort_values("WinPlaceBucket"))
sns.relplot(x="walkDistance", y="kills", hue="WinPlaceBucket", data=data)
sns.relplot(x="swimDistance", y="kills", hue="WinPlaceBucket", data=data)
sns.relplot(x="rideDistance", y="kills", hue="WinPlaceBucket", data=data)
sns.relplot(x="walkDistance", y="weaponsAcquired", hue="WinPlaceBucket", data=data)
sns.relplot(x="rideDistance", y="weaponsAcquired", hue="WinPlaceBucket", data=data)
sns.relplot(x="DBNOs", y="weaponsAcquired", hue="WinPlaceBucket", data=data)
sns.catplot(x="WinPlaceBucket", y="kills", kind="boxen",
            data=data.sort_values("WinPlaceBucket"))
#data.columns.values
data["matchType"].value_counts()
data['KillsBucket'] = np.where(data.kills >10, 'Kills:10+', 
                          np.where(data.kills>5, 'Kills:6 to 10', 
                                   np.where(data.kills>=3, 'Kills:3 to 5', 
                                            np.where(data.kills==2, 'Kills:2', 
                                                     np.where(data.kills==1, 'Kill:1', 'Kill:0')))))
data["KillsBucket"].value_counts()
sns.catplot(x="winPlacePerc", y="KillsBucket", kind="boxen",
            data=data.sort_values("KillsBucket"))
sns.relplot(x="walkDistance", y="KillsBucket", hue="WinPlaceBucket", data=data)
sns.relplot(x="swimDistance", y="KillsBucket", hue="WinPlaceBucket", data=data)
sns.relplot(x="rideDistance", y="KillsBucket", hue="WinPlaceBucket", data=data)
random.seed(120)
sns.pointplot(x="boosts", y="winPlacePerc", data=data.sample(1000), color="maroon")
sns.pointplot(x="heals", y="winPlacePerc", data=data.sample(1000), color="purple")
plt.text(14,0.5,"heals", color="purple")
plt.text(14,0.4,"boosts", color="maroon")
plt.xlabel("Heals/Boosts")
plt.ylabel("Win Place %")
data['GroupBucket'] = np.where(data.numGroups >50, 'Solo', 
                          np.where(data.numGroups>25 , 'Duo', 'Squad'))
data["GroupBucket"].value_counts()
sns.relplot(x="kills", y="winPlacePerc", hue="GroupBucket", kind="line", data=data)
sns.relplot(x="vehicleDestroys", y="winPlacePerc", kind="line",ci="sd", data=data)
sns.relplot(x="vehicleDestroys", y="kills", kind="line",ci="sd", data=data)
sns.relplot(x="vehicleDestroys", y="kills", kind="line",ci=None, hue="WinPlaceBucket", data=data)
data.columns.values
data_v2= data.drop(data[["matchDuration", "matchType", "maxPlace", "rankPoints", "WinPlaceBucket",
                        "KillsBucket", "GroupBucket"]], axis=1)
data_v2.columns.values
data_v2.info()
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split as split
train, test= split(data_v2, test_size=0.2, random_state=123)
train_y= train["winPlacePerc"].copy()
test_y= test["winPlacePerc"].copy()
train_x=train.drop(train[["winPlacePerc"]],axis=1)
test_x=test.drop(test[["winPlacePerc"]],axis=1)
print("train_x: ", np.isnan(train_x).any())
print("train_y: ",np.isnan(train_y).any())
print("test_x: ",np.isnan(test_x).any())
print("test_y: ",np.isnan(test_y).any())
test_y=test_y.fillna(test_y.mean())
#Check again
print("test_y: ",np.isnan(test_y).any())
reg= linear_model.LinearRegression()
reg.fit(train_x, train_y)
#Predicting the test set
test_y_pred= reg.predict(test_x)
list(zip(train.columns.values,reg.coef_))
mse= mean_squared_error(test_y_pred,test_y)
print("Mean Squared Error: ",mse)
r2_score(test_y_pred,test_y)
test_actual= pd.read_csv("../input/test_V2.csv")
test_actual.head()
test_model= test_actual.drop(test_actual[["Id", "groupId", "matchId", "matchDuration", "matchType", "maxPlace", "rankPoints"]], axis=1)
test_model_predict= reg.predict(test_model)
test_model_predict
op= pd.DataFrame(list(zip(test_actual["Id"], test_model_predict)))
op= op.rename(columns={0: 'Id', 1: 'winPlacePerc'})
op.head()
op.to_csv("sample_submission.csv", encoding='utf-8', index=False)
