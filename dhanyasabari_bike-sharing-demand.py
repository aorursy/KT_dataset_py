# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pylab

import calendar

import seaborn as sn

from scipy import stats

import missingno as msno

from datetime import datetime

import matplotlib.pyplot as plt

import warnings

pd.options.mode.chained_assignment = None

warnings.filterwarnings("ignore", category = DeprecationWarning)

%matplotlib inline

dailyData = pd.read_csv("../input/bike-sharing-demand/train.csv")
dailyData.shape
dailyData.head(10)
dailyData.dtypes
dailyData.datetime.apply(lambda x:x.split()[0])
#creating new columns from datetime column

# apply() : apply is a function in pandas library. It helps to apply a function(lambda/userdefined/Numpy) to the rows/columns in a dataFrame.

# The default value for axis in apply function is axis = 0 (column).

# lambda function: it takes input as a dataframe(all/specified number rows of a df or all/specified number columns)

dailyData["date"] = dailyData.datetime.apply(lambda x:x.split()[0])

dailyData["hour"] = dailyData.datetime.apply(lambda x: x.split()[1].split(":")[0])

# strptime: create a datetime object from a string 

# datetime.strptime(date_string, format) where datetime is an object that supplies different classes like strptime

# for manipulating and formatting date ot time 

dailyData["weekday"] = dailyData.date.apply(lambda dateString: calendar.day_name[datetime.strptime(dateString, "%Y-%m-%d").weekday()])

dailyData["month"] = dailyData.date.apply(lambda dateString: calendar.month_name[datetime.strptime(dateString, "%Y-%m-%d").month])

dailyData["season"] = dailyData.season.map({1:"Spring", 2:"Summer", 3:"Fall", 4:"winter"})

dailyData["weather"] = dailyData.weather.map({1: " Clear + Few clouds + Partly cloudy + Partly cloudy",\

                                        2 : " Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist ", \

                                        3 : " Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds", \

                                        4 :" Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog " })
#creating category variables

#A categorical variable is one that usually takes a fixed, number of possible values

categoryvariables = ["hour", "weekday", "month", "season", "weather", "holiday", "workingday"]

for var in categoryvariables:

    dailyData[var] = dailyData[var].astype("category")
#Dropping Datetime column

dailyData = dailyData.drop(["datetime"], axis =1)
#creating the data for the plot

typesCountSerie = dailyData.dtypes.value_counts()



# format columns as arrays of either strings or integers

# typeNames are easier to sort as array of `string` rather than an array of `dtype`

typeNamesColumn = list(map(lambda t: t.name , typesCountSerie.index.values));

typeCountColumn = typesCountSerie.values

# create an initial dataframe, with multiple occurences of the same "variableType"

intialDataTypeDf = pd.DataFrame({

    "variableType": typeNamesColumn, 

    "count": typeCountColumn

})



# Group initial data frame by "variableType", 

# then reset_index to have a proper dataframe

groupedDataTypeDf = intialDataTypeDf.groupby(['variableType']).sum()[['count']].reset_index()

#dataTypeDf = pd.DataFrame(dailyData.dtypes.value_counts()).reset_index().rename(columns={"index":"variableType",0:"count"})

fig,ax = plt.subplots()

fig.set_size_inches(12, 5)

#plotting the barchart

sn.barplot(data=groupedDataTypeDf, x="variableType", y="count", ax=ax)

ax.set(xlabel = 'variableType', ylabel = 'Count', title = "Count of the different Datatypes")
msno.matrix(dailyData, figsize=(12,5))
msno.bar(dailyData, figsize=(12,5))
fig, axes = plt.subplots(nrows=2, ncols=2)

fig.set_size_inches(12,10)

sn.boxplot(data = dailyData, y = "count", orient = "v", ax=axes[0][0])

sn.boxplot(data = dailyData, y = "count", x = "season", orient = "v", ax=axes[0][1])

sn.boxplot(data = dailyData, y = "count", x= "hour", orient = "v", ax=axes[1][0])

sn.boxplot(data = dailyData, y = "count", x = "workingday", orient = "v", ax=axes[1][1])



axes[0][0].set(ylabel='Count',title="Box Plot On Count")

axes[0][1].set(xlabel = 'Season', ylabel='Count',title="Box Plot On Count across season")

axes[1][0].set(xlabel = 'Hour of the day', ylabel='Count',title="Box Plot On Count across Hour of the day")

axes[1][1].set(xlabel = 'Working Day', ylabel='Count',title="Box Plot On Count  across Working day")
#checking how many count values are with in 3*standard deviation

np.sum(np.abs(dailyData["count"]-dailyData["count"].mean())<=(3*dailyData["count"].std()))



dailyDataWithoutOutliers = dailyData[np.abs(dailyData["count"]-dailyData["count"].mean())<=(3*dailyData["count"].std())]
print("shape of the data with outliers", dailyData.shape)

print("shape of the data without outliers", dailyDataWithoutOutliers.shape)
dailyData[["temp","atemp","casual","registered","humidity","windspeed","count"]].dtypes



dailyDataCorr = dailyData[["temp","atemp","casual","registered","humidity","windspeed", "count"]].corr()

mask = np.array(dailyDataCorr)

mask[np.tril_indices_from(mask)] = False

fig, ax = plt.subplots()

fig.set_size_inches(20,10)

sn.heatmap(dailyDataCorr, mask = mask, vmax = .8, square = True, annot = True)
#casual(non registered)+registered = count

#https://www.kaggle.com/jjuanramos/bike-sharing-demand

plt.scatter(x = dailyData['casual'] + dailyData['registered'], y = dailyData['count'])

plt.show()
fig,(ax1,ax2,ax3) = plt.subplots(ncols=3)

fig.set_size_inches(12, 5)

sn.regplot(x="temp", y="count", data=dailyData,ax=ax1)

sn.regplot(x="windspeed", y="count", data=dailyData,ax=ax2)

sn.regplot(x="humidity", y="count", data=dailyData,ax=ax3)
fig,axes = plt.subplots(ncols=2,nrows=2)

fig.set_size_inches(12, 10)

sn.distplot(dailyData["count"],ax=axes[0][0])

stats.probplot(dailyData["count"], dist='norm', fit=True, plot=axes[0][1])

sn.distplot(np.log(dailyDataWithoutOutliers["count"]),ax=axes[1][0])

stats.probplot(np.log1p(dailyDataWithoutOutliers["count"]), dist='norm', fit=True, plot=axes[1][1])
fig,(ax1)= plt.subplots(nrows=1)

fig.set_size_inches(10,5)

sortOrder = ["January","February","March","April","May","June","July","August","September","October","November","December"]

hueOrder = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]



monthAggregated = pd.DataFrame(dailyData.groupby("month")["count"].mean()).reset_index()

monthSorted = monthAggregated.sort_values(by="count",ascending=False)

sn.barplot(data=monthSorted,x="month",y="count",ax=ax1,order=sortOrder)

ax1.set(xlabel='Month', ylabel='Avearage Count',title="Average Count By Month")
fig,ax2= plt.subplots(nrows=1)

fig.set_size_inches(10,5)

hourAggregated = pd.DataFrame(dailyData.groupby(["hour","season"],sort=True)["count"].mean()).reset_index()

sn.pointplot(x=hourAggregated["hour"], y=hourAggregated["count"],hue=hourAggregated["season"], data=hourAggregated, join=True,ax=ax2)

ax2.set(xlabel='Hour Of The Day', ylabel='Users Count',title="Average Users Count By Hour Of The Day Across Season",label='big')
fig,ax3 = plt.subplots(nrows = 1)

fig.set_size_inches(10,5)

hourAggregated = pd.DataFrame(dailyData.groupby(["hour","weekday"],sort=True)["count"].mean()).reset_index()

sn.pointplot(x=hourAggregated["hour"], y=hourAggregated["count"],hue=hourAggregated["weekday"],hue_order=hueOrder, data=hourAggregated, join=True,ax=ax3)

ax3.set(xlabel='Hour Of The Day', ylabel='Users Count',title="Average Users Count By Hour Of The Day Across Weekdays",label='big')
fig, ax4 = plt.subplots(nrows=1)

fig.set_size_inches(10, 5)

hourTransformed = pd.melt(dailyData[["hour","casual","registered"]], id_vars=['hour'], value_vars=['casual', 'registered'])

hourAggregated = pd.DataFrame(hourTransformed.groupby(["hour","variable"],sort=True)["value"].mean()).reset_index()

sn.pointplot(x=hourAggregated["hour"], y=hourAggregated["value"],hue=hourAggregated["variable"],hue_order=["casual","registered"], data=hourAggregated, join=True,ax=ax4)

ax4.set(xlabel='Hour Of The Day', ylabel='Users Count',title="Average Users Count By Hour Of The Day Across User Type",label='big')
dataTrain = pd.read_csv("../input/bike-sharing-demand/train.csv")

dataTest = pd.read_csv("../input/bike-sharing-demand/test.csv")
#combine test and train data

data = dataTrain.append(dataTest)

data.reset_index(inplace = True)

data.drop('index', inplace = True, axis = 1)
data["date"] = data.datetime.apply(lambda x : x.split()[0])

data["hour"] = data.datetime.apply(lambda x : x.split()[1].split(":")[0]).astype("int")

data["year"] = data.datetime.apply(lambda x : x.split()[0].split("-")[0])

data["weekday"] = data.date.apply(lambda dateString : datetime.strptime(dateString,"%Y-%m-%d").weekday())

data["month"] = data.date.apply(lambda dateString : datetime.strptime(dateString,"%Y-%m-%d").month)

dataWindspeedOriginal = data["windspeed"]
fig,ax= plt.subplots(nrows=1)

fig.set_size_inches(20,5)

#sortOrder = ["January","February","March","April","May","June","July","August","September","October","November","December"]

#hueOrder = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]



windspeed = pd.DataFrame(data.windspeed.value_counts()).reset_index()

plt.xticks(rotation=45) 

sn.barplot(data=windspeed,x="index",y="windspeed",ax=ax)

ax1.set(xlabel='windspeed Values', ylabel='Count',title="Count of windspeed values before imputing")
from sklearn.ensemble import RandomForestClassifier

wCol= ["season","weather","humidity","month","temp","year","atemp"]

#dataWind0 is the entire dataset(contains cols season, weather, humidity, month, temp, year, atemp)

#with windspeed value = 0

dataWind0 = data[data["windspeed"] == 0]

#dataNotWind0 is the entire dataset(contains cols season, weather, humidity, month, temp, year, atemp)

#without windspeed value = 0

dataWindNot0 = data[data["windspeed"] != 0]

dataWindNot0["windspeed"] = dataWindNot0["windspeed"].astype("str")



#predicting value for windspeed = 0

rfModel_wind = RandomForestClassifier()

rfModel_wind.fit(dataWindNot0[wCol], dataWindNot0["windspeed"])

Wind0Values = rfModel_wind.predict(X = dataWind0[wCol])



dataWind0["windspeed"] = Wind0Values

data = dataWindNot0.append(dataWind0)

data["windspeed"] = data["windspeed"].astype("float")

data.reset_index(inplace=True)

data.drop('index',inplace=True,axis=1)



fig, ax =plt.subplots(nrows=1)

fig.set_size_inches(20,5)

windspeed = pd.DataFrame(data.windspeed.value_counts()).reset_index()

plt.xticks(rotation=45) 

sn.barplot(data=windspeed,x="index",y="windspeed",ax=ax)

ax.set(xlabel='Windspeed Values', ylabel='Count',title="Count Of Windspeed Values After Imputing",label='big')
categoricalFeatureNames = ["season","holiday","workingday","weather","weekday","month","year","hour"]

numericalFeatureNames = ["temp","humidity","windspeed","atemp"]

dropFeatures = ['casual',"count","datetime","date","registered"]
for var in categoricalFeatureNames:

    data[var] = data[var].astype("category")
data.head()
dataTrain = data[pd.notnull(data['count'])].sort_values(by=["datetime"])#datatime is not droppe, month, week etc are created from it.

dataTest = data[~pd.notnull(data['count'])].sort_values(by = ["datetime"])

datetimecol = dataTest["datetime"]

yLabels = dataTrain["count"]

yLabelsRegistered = dataTrain["registered"]

yLabelsCasual = dataTrain["casual"]
from sklearn.model_selection import train_test_split

X_train, X_validate, y_train, y_validate = train_test_split( dataTrain, yLabels, test_size=0.3, random_state=42)

dateTimeColValidate = X_validate["datetime"]
dataTrain  = dataTrain.drop(dropFeatures,axis=1)

dataTest  = dataTest.drop(dropFeatures,axis=1)

X_train = X_train.drop(dropFeatures,axis=1)

X_validate = X_validate.drop(dropFeatures,axis=1)
def rmsle(y, y_,convertExp=True):

    if convertExp:

        y = np.exp(y),

        y_ = np.exp(y_)

    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))

    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))

    calc = (log1 - log2) ** 2

    return np.sqrt(np.mean(calc))
import warnings

pd.options.mode.chained_assignment = None

warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.model_selection import GridSearchCV



#Initialize Logistic Regression model

lModel = LinearRegression()

# Train the model

lModel.fit(X = X_train,y = np.log1p(y_train))

# Make predictions

preds = lModel.predict(X= X_validate)

print ("RMSLE Value For Linear Regression In Validation: ",rmsle(np.exp(np.log1p(y_validate)),np.exp(preds),False))
predsTest = lModel.predict(X=dataTest)

fig,(ax1,ax2)= plt.subplots(ncols=2)

fig.set_size_inches(20,5)

sn.distplot(yLabels,ax=ax1,bins=100)

sn.distplot(np.exp(predsTest),ax=ax2,bins=100)

ax1.set(title="Training Set Distribution")

ax2.set(title="Test Set Distribution")
#print ("RMSLE Value For Linear Regression In Validation: ",rmsle(np.exp(np.log1p(y_validate)),np.exp(predsTest),False))
from sklearn.linear_model import Ridge

from sklearn.model_selection import GridSearchCV

from sklearn import metrics



ridge_m_ = Ridge()

ridge_params_ = { 'max_iter':[3000],'alpha':[0.01,0.05,0.1, 1, 2, 3, 4, 10, 30,100,200,300,400,800,900,1000]}

rmsle_scorer = metrics.make_scorer(rmsle, greater_is_better = False)

grid_ridge_m = GridSearchCV( ridge_m_,

                          ridge_params_,

                          scoring = rmsle_scorer,

                          cv=5)

grid_ridge_m.fit(X = X_train, y = np.log1p(y_train))

preds = grid_ridge_m.predict(X = X_validate)

print(grid_ridge_m.best_params_)

print ("RMSLE Value For Ridge Regression: ",rmsle(np.exp(np.log1p(y_validate)),np.exp(preds),False))



fig,ax= plt.subplots()

fig.set_size_inches(20,5)

df = pd.DataFrame(grid_ridge_m.cv_results_)

df

df["alpha"] = df["params"].apply(lambda x:x["alpha"])

df["rmsle"] = df["mean_test_score"].apply(lambda x:-x)

sn.pointplot(data=df,x="alpha",y="rmsle",ax=ax)
from sklearn.linear_model import Lasso

lasso_m_ = Lasso()

alpha  = [0.001,0.005,0.01,0.3,0.1,0.3,0.5,0.7,1]

lasso_params_ = { 'max_iter':[3000],'alpha':alpha}

#rmsle_scorer = metrics.make_scorer(rmsle, greater_is_better=False)

grid_lasso_m = GridSearchCV(lasso_m_, lasso_params_, scoring = rmsle_scorer, cv = 5)

grid_lasso_m.fit(X = X_train,y = np.log1p(y_train))

preds = grid_lasso_m.predict(X= X_validate)

print (grid_lasso_m.best_params_)

print ("RMSLE Value: ",rmsle(np.exp(np.log1p(y_validate)),np.exp(preds), False))



fig,ax= plt.subplots()

fig.set_size_inches(20,5)

df = pd.DataFrame(grid_lasso_m.cv_results_)

df["alpha"] = df["params"].apply(lambda x:x["alpha"])

df["rmsle"] = df["mean_test_score"].apply(lambda x:-x)

sn.pointplot(data=df,x="alpha",y="rmsle",ax=ax)
from sklearn.ensemble import RandomForestRegressor

rfModel = RandomForestRegressor(n_estimators=100)

rfModel.fit(X = X_train,y = np.log1p(y_train))

preds = rfModel.predict(X= X_validate)

print ("RMSLE Value: ",rmsle(np.exp(np.log1p(y_validate)),np.exp(preds), False))
features = pd.DataFrame()

features['features'] = X_train.columns

features['coefficient'] = rfModel.feature_importances_

features.sort_values(by=['coefficient'],ascending=False,inplace=True)

fig,ax= plt.subplots()

fig.set_size_inches(20,5)

sn.barplot(data=features,x="features",y="coefficient",ax=ax)
predsTest = rfModel.predict(X=dataTest)

fig,(ax1,ax2)= plt.subplots(ncols=2)

fig.set_size_inches(20,5)

sn.distplot(yLabels,ax=ax1,bins=100)

sn.distplot(np.exp(predsTest),ax=ax2,bins=100)

ax1.set(title="Training Set Distbution")

ax2.set(title="Test Set Distribution")
submission = pd.DataFrame({

        "datetime": datetimecol,

        "count": [max(0, x) for x in np.exp(predsTest)]

    })

submission.to_csv('bike_predictions_gbm_separate_without_fe.csv', index=False)