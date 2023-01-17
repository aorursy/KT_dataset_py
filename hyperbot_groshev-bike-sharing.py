import pylab
import calendar
import numpy as np
import pandas as pd
import seaborn as sn
from scipy import stats
import missingno as msno
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=DeprecationWarning)
%matplotlib inline
dataTrain = pd.read_csv("../input/bike-sharing-demand/train.csv")
dataTest = pd.read_csv("../input/bike-sharing-demand/test.csv")
data = dataTrain.append(dataTest)
data.reset_index(inplace=True)
data.drop('index',inplace=True,axis=1)
data["date"] = data.datetime.apply(lambda x : x.split()[0])
data["hour"] = data.datetime.apply(lambda x : x.split()[1].split(":")[0]).astype("int")
data["year"] = data.datetime.apply(lambda x : x.split()[0].split("-")[0])
data["weekday"] = data.date.apply(lambda dateString : datetime.strptime(dateString,"%Y-%m-%d").weekday())
data["month"] = data.date.apply(lambda dateString : datetime.strptime(dateString,"%Y-%m-%d").month)
data.head()
categoryVariableList = ["hour","weekday","month","season","weather","holiday","workingday", "year"]
for var in categoryVariableList:
    data[var] = data[var].astype("category")
from sklearn.ensemble import RandomForestRegressor

dataWind0 = data[data["windspeed"]==0]
dataWindNot0 = data[data["windspeed"]!=0]
rfModel_wind = RandomForestRegressor()
windColumns = ["season","weather","humidity","temp"]
rfModel_wind.fit(dataWindNot0[windColumns], dataWindNot0["windspeed"])

wind0Values = rfModel_wind.predict(X= dataWind0[windColumns])
dataWind0["windspeed"] = wind0Values
data = dataWindNot0.append(dataWind0)
data.reset_index(inplace=True)
data.drop('index',inplace=True,axis=1)
dataTrain = data[pd.notnull(data['count'])].sort_values(by=["datetime"])
dataTest = data[~pd.notnull(data['count'])].sort_values(by=["datetime"])
fig, axes = plt.subplots(nrows=2,ncols=2)
fig.set_size_inches(12, 10)
sn.boxplot(data=dataTrain,y="count",orient="v",ax=axes[0][0])
sn.boxplot(data=dataTrain,y="count",x="season",orient="v",ax=axes[0][1])
sn.boxplot(data=dataTrain,y="count",x="hour",orient="v",ax=axes[1][0])
sn.boxplot(data=dataTrain,y="count",x="workingday",orient="v",ax=axes[1][1])

axes[0][0].set(ylabel='Count',title="Box Plot On Count")
axes[0][1].set(xlabel='Season', ylabel='Count',title="Box Plot On Count Across Season")
axes[1][0].set(xlabel='Hour Of The Day', ylabel='Count',title="Box Plot On Count Across Hour Of The Day")
axes[1][1].set(xlabel='Working Day', ylabel='Count',title="Box Plot On Count Across Working Day")
dataTrainWithoutOutliers = dataTrain[np.abs(dataTrain["count"]-dataTrain["count"].mean())<=(3*dataTrain["count"].std())] 
print ("Shape Of The Before Ouliers: ", dataTrain.shape)
print ("Shape Of The After Ouliers: ", dataTrainWithoutOutliers.shape)
corrmat = dataTrainWithoutOutliers
toDrop = categoryVariableList
toDrop.append('date')
toDrop.append('datetime')
corrmat.drop(toDrop, axis=1)
corrmat.head()
f, ax = plt.subplots(figsize=(12, 9))
sn.heatmap(corrmat.corr(), vmax=1, square=True);

dropFeatures = ['casual',"count","datetime","date","registered"]
datetimecol = dataTest["datetime"]
yLabels = dataTrainWithoutOutliers["count"]
dataTrainWithoutOutliers  = dataTrainWithoutOutliers.drop(dropFeatures,axis=1)
dataTest  = dataTest.drop(dropFeatures,axis=1)
def rmsle(y, y_,convertExp=True):
    if convertExp:
        y = np.exp(y),
        y_ = np.exp(y_)
    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))
    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))
from sklearn.ensemble import GradientBoostingRegressor
gbm = GradientBoostingRegressor(n_estimators=4000,alpha=0.01)
yLabelsLog = np.log1p(yLabels)
gbm.fit(dataTrainWithoutOutliers,yLabelsLog)
preds = gbm.predict(X= dataTrainWithoutOutliers)
print ("RMSLE Value For Gradient Boost: ",rmsle(np.exp(yLabelsLog),np.exp(preds),False))
submission = pd.DataFrame({
        "datetime": datetimecol,
        "count": [max(0, x) for x in np.exp(preds)]
    })
submission.to_csv('bike_predictions1.csv', index=False)