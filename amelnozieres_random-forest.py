# data visualisation and manipulation

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns

% matplotlib inline  



#import the necessary modelling algos.



#regression

from sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV

from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor



#model selection

from sklearn.model_selection import train_test_split,cross_validate

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV



#evaluation metrics

from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error # for regression





data = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

print(data.head(5))



print(data["count"].describe())
data.shape
test.shape
# Extract hours from datetime

data['datetime'] = pd.to_datetime(data['datetime'])

data['hour'] = data['datetime'].dt.hour

data['month'] = data['datetime'].dt.month



test['datetime'] = pd.to_datetime(test['datetime'])

test['hour'] = data['datetime'].dt.hour

test['month'] = data['datetime'].dt.month
data['season'] = data.season.astype('category')

data['month'] = data.month.astype('category')

data['hour'] = data.hour.astype('category')

data['holiday'] = data.holiday.astype('category')

data['workingday'] = data.workingday.astype('category')

data['weather'] = data.weather.astype('category')





test['season'] = test.season.astype('category')

test['month'] = test.month.astype('category')

test['hour'] = test.hour.astype('category')

test['holiday'] = test.holiday.astype('category')

test['workingday'] = test.workingday.astype('category')

test['weather'] = test.weather.astype('category')

data.drop('datetime',axis=1,inplace=True)

test.drop('datetime',axis=1,inplace=True)
data.columns
test.columns
# Removing outliers



dataWithoutOutliers = data[np.abs(data["count"]-data["count"].mean())<=(3*data["count"].std())] 

print ("Shape Of The Before Ouliers: ",data.shape)

print ("Shape Of The After Ouliers: ",dataWithoutOutliers.shape)



dataTrain = pd.read_csv("../input/train.csv")

dataTest = pd.read_csv("../input/test.csv")



MergeData = dataTrain.append(dataTest)

MergeData.reset_index(inplace=True)

MergeData.drop('index',inplace=True,axis=1)
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



MergeData["date"] = MergeData.datetime.apply(lambda x : x.split()[0])

MergeData["hour"] = MergeData.datetime.apply(lambda x : x.split()[1].split(":")[0]).astype("int")

MergeData["year"] = MergeData.datetime.apply(lambda x : x.split()[0].split("-")[0])

MergeData["weekday"] = MergeData.date.apply(lambda dateString : datetime.strptime(dateString,"%Y-%m-%d").weekday())

MergeData["month"] = MergeData.date.apply(lambda dateString : datetime.strptime(dateString,"%Y-%m-%d").month)



from sklearn.ensemble import RandomForestRegressor



dataWind0 = MergeData[MergeData["windspeed"]==0]

dataWindNot0 = MergeData[MergeData["windspeed"]!=0]

rfg_wind = RandomForestRegressor()

windColumns = ["season","weather","humidity","month","temp","year","atemp"]

rfg_wind.fit(dataWindNot0[windColumns], dataWindNot0["windspeed"])



wind0Values = rfg_wind.predict(X= dataWind0[windColumns])

dataWind0["windspeed"] = wind0Values

MergeData = dataWindNot0.append(dataWind0)

MergeData.reset_index(inplace=True)

MergeData.drop('index',inplace=True,axis=1)
categoricalFeatureNames = ["season","holiday","workingday","weather","weekday","month","year","hour"]

numericalFeatureNames = ["temp","humidity","windspeed","atemp"]

dropFeatures = ['casual',"count","datetime","date","registered"]
for var in categoricalFeatureNames:

    MergeData[var] = MergeData[var].astype("category")




train = MergeData[pd.notnull(MergeData['count'])].sort_values(by=["datetime"])

test = MergeData[~pd.notnull(MergeData['count'])].sort_values(by=["datetime"])

datetimecol = test["datetime"]

yLabels = train["count"]

yLablesRegistered = train["registered"]

yLablesCasual = train["casual"]



dataTrain  = train.drop(dropFeatures,axis=1)

dataTest  = test.drop(dropFeatures,axis=1)
def rmsle(y, y_,convertExp=True):

    if convertExp:

        y = np.exp(y),

        y_ = np.exp(y_)

    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))

    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))

    calc = (log1 - log2) ** 2

    return np.sqrt(np.mean(calc))



from sklearn.ensemble import RandomForestRegressor

rfModel = RandomForestRegressor(n_estimators=100)

yLabelsLog = np.log1p(yLabels)

rfModel.fit(dataTrain,yLabelsLog)

preds = rfModel.predict(X= dataTrain)

print ("RMSLE Value For Random Forest: ",rmsle(np.exp(yLabelsLog),np.exp(preds),False))



predsTest = rfModel.predict(X= dataTest)

fig,(ax1,ax2)= plt.subplots(ncols=2)

fig.set_size_inches(12,5)

sn.distplot(yLabels,ax=ax1,bins=50)

sn.distplot(np.exp(predsTest),ax=ax2,bins=50)



[max(0, x) for x in np.exp(predsTest)]
np.exp(predsTest)
from sklearn.ensemble import GradientBoostingRegressor

gbm = GradientBoostingRegressor(n_estimators=4000,alpha=0.01); ### Test 0.41

yLabelsLog = np.log1p(yLabels)

gbm.fit(dataTrain,yLabelsLog)

preds = gbm.predict(X= dataTrain)

print ("RMSLE Value For Gradient Boost: ",rmsle(np.exp(yLabelsLog),np.exp(preds),False))
gbmpredsTest = gbm.predict(X= dataTest)

fig,(ax1,ax2)= plt.subplots(ncols=2)

fig.set_size_inches(12,5)

sn.distplot(yLabels,ax=ax1,bins=50)

sn.distplot(np.exp(gbmpredsTest),ax=ax2,bins=50)



submission = pd.DataFrame({

        "datetime": datetimecol,

        "count": [max(0, x) for x in np.exp(gbmpredsTest)]

    })

submission.to_csv('my_submission_19.csv', index=False)
