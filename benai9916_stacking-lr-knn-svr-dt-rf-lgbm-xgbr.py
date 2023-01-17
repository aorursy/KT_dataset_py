import pylab
import calendar

import numpy as np
import pandas as pd

import seaborn as sn
import matplotlib.pyplot as plt

from scipy import stats
import missingno as msno
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import scorer, mean_absolute_error, mean_squared_error, r2_score

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

pd.options.mode.chained_assignment = None

%matplotlib inline
bike = pd.read_csv("../input/bike-sharing-demand/train.csv")
# Show first 5 rows

bike.head(5)
# size of the data
bike.shape
# some basic information about data
bike.info()
bike["date"] = bike.datetime.apply(lambda x : x.split()[0])

bike["hour"] = bike.datetime.apply(lambda x : x.split()[1].split(":")[0])

bike["weekday"] = bike.date.apply(lambda dateString : calendar.day_name[datetime.strptime(dateString,"%Y-%m-%d").weekday()])

bike["month"] = bike.date.apply(lambda dateString : calendar.month_name[datetime.strptime(dateString,"%Y-%m-%d").month])

bike["season"] = bike.season.map({1: "Spring", 2 : "Summer", 3 : "Fall", 4 :"Winter" })

bike["weather"] = bike.weather.map({1: " Clear + Few clouds + Partly cloudy + Partly cloudy",\
                                        2 : " Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist ", \
                                        3 : " Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds", \
                                        4 :" Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog " })
categoryVariableList = ["hour","weekday","month","season","weather","holiday","workingday"]

for var in categoryVariableList:
    bike[var] = bike[var].astype("category")
dailyData  = bike.drop(["datetime"],axis=1)
# count the data type in each column
bike_df = pd.DataFrame(bike.dtypes.value_counts()).reset_index().rename(columns={"index":"variableType",0:"count"})

bike_df
msno.matrix(bike,figsize=(12,5))
fig, axes = plt.subplots(nrows=2,ncols=2)
fig.set_size_inches(18, 10)
fig.tight_layout(pad=3.0)
sn.boxplot(data=bike,y="count",orient="v",ax=axes[0][0])
sn.boxplot(data=bike,y="count",x="season",orient="v",ax=axes[0][1])
sn.boxplot(data=bike,y="count",x="hour",orient="v",ax=axes[1][0])
sn.boxplot(data=bike,y="count",x="workingday",orient="v",ax=axes[1][1])

axes[0][0].set(ylabel='Count',title="Box Plot On Count")
axes[0][1].set(xlabel='Season', ylabel='Count',title="Box Plot On Count Across Season")
axes[1][0].set(xlabel='Hour Of The Day', ylabel='Count',title="Box Plot On Count Across Hour Of The Day")
axes[1][1].set(xlabel='Working Day', ylabel='Count',title="Box Plot On Count Across Working Day")
bikeDataWithoutOutliers =  bike[np.abs(bike["count"]-bike["count"].mean())<=(3*bike["count"].std())] 
print ("Shape Of The Before Ouliers: ",bike.shape)
print ("Shape Of The After Ouliers: ",bikeDataWithoutOutliers.shape)
corrMatt = bike[["temp","atemp","casual","registered","humidity","windspeed","count"]].corr()
mask = np.array(corrMatt)
mask[np.tril_indices_from(mask)] = False
fig,ax= plt.subplots()
fig.set_size_inches(20,10)
sn.heatmap(corrMatt, mask=mask,vmax=.8, square=True,annot=True)
fig,(ax1,ax2,ax3) = plt.subplots(ncols=3)
fig.set_size_inches(20, 5)
sn.regplot(x="temp", y="count", data=bike,ax=ax1)
sn.regplot(x="windspeed", y="count", data=bike,ax=ax2)
sn.regplot(x="humidity", y="count", data=bike,ax=ax3)
fig,axes = plt.subplots(ncols=2,nrows=2)
fig.set_size_inches(12, 10)
sn.distplot(bike["count"],ax=axes[0][0])
stats.probplot(bike["count"], dist='norm', fit=True, plot=axes[0][1])

# without outlier data frame
sn.distplot(np.log(bikeDataWithoutOutliers["count"]),ax=axes[1][0])
stats.probplot(np.log1p(bikeDataWithoutOutliers["count"]), dist='norm', fit=True, plot=axes[1][1])
fig,(ax1,ax2,ax3,ax4)= plt.subplots(nrows=4)
fig.set_size_inches(12,20)
bike
sortOrder = ["January","February","March","April","May","June","July","August","September","October","November","December"]
hueOrder = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]

monthAggregated = pd.DataFrame(bike.groupby("month")["count"].mean()).reset_index()
monthSorted = monthAggregated.sort_values(by="count",ascending=False)
sn.barplot(data=monthSorted,x="month",y="count",ax=ax1,order=sortOrder)
ax1.set(xlabel='Month', ylabel='Avearage Count',title="Average Count By Month")

hourAggregated = pd.DataFrame(bike.groupby(["hour","season"],sort=True)["count"].mean()).reset_index()
sn.pointplot(x="hour", y="count",hue="season", data=hourAggregated, join=True,ax=ax2)
ax2.set(xlabel='Hour Of The Day', ylabel='Users Count',title="Average Users Count By Hour Of The Day Across Season",label='big')

hourAggregated = pd.DataFrame(bike.groupby(["hour","weekday"],sort=True)["count"].mean()).reset_index()
sn.pointplot(x="hour", y="count",hue="weekday",hue_order=hueOrder, data=hourAggregated, join=True,ax=ax3)
ax3.set(xlabel='Hour Of The Day', ylabel='Users Count',title="Average Users Count By Hour Of The Day Across Weekdays",label='big')

hourTransformed = pd.melt(bike[["hour","casual","registered"]], id_vars=['hour'], value_vars=['casual', 'registered'])
hourAggregated = pd.DataFrame(hourTransformed.groupby(["hour","variable"],sort=True)["value"].mean()).reset_index()
sn.pointplot(x="hour", y="value",hue="variable",hue_order=["casual","registered"], data=hourAggregated, join=True,ax=ax4)
ax4.set(xlabel='Hour Of The Day', ylabel='Users Count',title="Average Users Count By Hour Of The Day Across User Type",label='big')
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
from sklearn.ensemble import RandomForestRegressor

dataWind0 = data[data["windspeed"]==0]
dataWindNot0 = data[data["windspeed"]!=0]

# initalize object
rfModel_wind = RandomForestRegressor()

# columns
windColumns = ["season","weather","humidity","month","temp","year","atemp"]

# fit the data
rfModel_wind.fit(dataWindNot0[windColumns], dataWindNot0["windspeed"])

# predict
wind0Values = rfModel_wind.predict(X= dataWind0[windColumns])

dataWind0["windspeed"] = wind0Values
data = dataWindNot0.append(dataWind0)
data.reset_index(inplace=True)
data.drop('index',inplace=True,axis=1)
categoricalFeatureNames = ["season","holiday","workingday","weather","weekday","month","year","hour"]

numericalFeatureNames = ["temp","humidity","windspeed","atemp"]

dropFeatures = ['casual',"datetime","date","registered"]
for var in categoricalFeatureNames:
    data[var] = data[var].astype("category")
data_df = data.copy(deep=True)

# drop unwanted columns
data_df  = data_df.drop(dropFeatures,axis=1)
# drop missing values
data_df  = data_df.dropna()

data_df.shape
y = data_df['count']
x = data_df.drop('count', axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size = .3, random_state=0)
# define the base models
level0 = list()
level0.append(('rig',Ridge()))
level0.append(('laso', Lasso()))
level0.append(('knn', KNeighborsRegressor()))
level0.append(('cart', DecisionTreeRegressor()))
level0.append(('svm', SVR(kernel='linear')))

# define meta learner model
level1 = LinearRegression()

# define the stacking ensemble
model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)

# fit the model on all available data
model.fit(x_train, y_train)
# make prediction
y_predict = model.predict(x_test)

print('MAE : ', mean_absolute_error(y_test, y_predict))
print('MSE : ', mean_squared_error(y_test, y_predict))
print('R Squared : ', r2_score(y_test, y_predict))
from mlxtend.regressor import StackingCVRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor

xgb = XGBRegressor()
lgbm = LGBMRegressor()
rf = RandomForestRegressor()
ridge = Ridge()
lasso = Lasso()
svr = SVR(kernel='linear')

stack = StackingCVRegressor(regressors=(ridge, lasso, svr, rf, lgbm, xgb),
                            meta_regressor=xgb, cv=12,
                            use_features_in_secondary=True,
                            store_train_meta_features=True,
                            shuffle=False,
                            random_state=42)

stack.fit(x_train, y_train)
# change the data type to int
x_test_2 = x_test.astype(int)

# feature column name (to avoid error)
x_test_2.columns = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11']
# make prediction

y_predict_x = stack.predict(x_test_2)
print('MAE : ', mean_absolute_error(y_test, y_predict_x))
print('MSE : ', mean_squared_error(y_test, y_predict_x))
print('R Squared : ', r2_score(y_test, y_predict_x))
