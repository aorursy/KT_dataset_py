import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import warnings

warnings.filterwarnings('ignore')



import pylab

import calendar

import numpy as np

import pandas as pd

import seaborn as sns

from scipy import stats

import missingno as msno

from datetime import datetime

import matplotlib.pyplot as plt

import warnings
import pandas as pd

test = pd.read_csv("../input/bikesharingdemand-dataset/test.csv")

train = pd.read_csv("../input/bikesharingdemand-dataset/train.csv")

###################################################################

#1)Data Summary

###################################################################

print(test.head(12))

print(test.dtypes)

print(test.shape)

print(train.shape)

print(train.describe())
###################################################################

#2)Preprocessing

###################################################################

train['date'] = train.datetime.apply(lambda x:x.split(" ")[0])

print(train.date.head(2))

train['time'] = train.datetime.apply(lambda x:x.split()[1].split(":")[0])

print(train.time.head(2))

train['weekday'] = train.date.apply(lambda dateString : calendar.day_name[datetime.strptime(dateString,"%Y-%m-%d").weekday()])

train['month'] =train.date.apply(lambda dateString : calendar.month_name[datetime.strptime(dateString,"%Y-%m-%d").month])

train["season"] = train.season.map({1: "Spring", 2 : "Summer", 3 : "Fall", 4 :"Winter" })

train["weather"] = train.weather.map({1: " Clear + Few clouds + Partly cloudy + Partly cloudy",\

                                        2 : " Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist ", \

                                        3 : " Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds", \

                                        4 :" Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog " })

train['workingday'] = train.workingday.map({0:"weekday",\

                                         1:"weekend"})

train['holiday'] = train.holiday.map({0:"no_holiday",\

                                         1:"holiday"})

print(train.head(5))

print(train.dtypes)

#train['datetime'] = pd.to_datetime(train['datetime'])

print(train.dtypes)
#Detecting Missing value 

print(train.isnull().sum())



## Replace using median 

#median = train['weekday'].median()

#train['weekday'].fillna(median, inplace=True)



#Outlier

fig, axes = plt.subplots(nrows=2,ncols=2)

fig.set_size_inches(12, 10)

sns.boxplot(data=train, x='season',y='count',ax=axes[0][0])

g=sns.boxplot(data=train, x='weekday',y='count',ax=axes[0][1])

g.set_xticklabels(g.get_xticklabels(),rotation=30)

sns.boxplot(data=train, x='time',y='count',ax=axes[1][0])

sns.boxplot(data=train, x='holiday',y='count',ax=axes[1][1])



#frequency table

print(train.groupby(["season", "month"]).size()) 

#print(pd.crosstab(train.season,train.month))
###################################################################

#3)Correlation and distribution

###################################################################

#numerical variables

num = train.select_dtypes(include=['float64','int64'])

print(num.head())

sns.pairplot(num)

plt.show()

sns.pairplot(train, kind="scatter", hue="season", palette="Set2")

plt.show()
f, ax = plt.subplots(figsize=(11, 9)) #figure size

corr = num.corr()

mask = np.zeros_like(corr, dtype=np.bool) #upper triangle

mask[np.triu_indices_from(mask)] = True



ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(220, 10, as_cmap=True),

    square=True,annot=True,mask=mask

)



ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

);
#histogram

fig,axes = plt.subplots(ncols=2,nrows=2)

fig.set_size_inches(12, 10)

sns.distplot(train["count"],ax=axes[0][0])

stats.probplot(train["count"], dist='norm', fit=True, plot=axes[0][1])



sns.distplot(np.sqrt(train["count"]),ax=axes[1][0])

stats.probplot(np.sqrt(train["count"]), dist='norm', fit=True, plot=axes[1][1])
###################################################################

#3)Linear Regression

###################################################################

from sklearn.linear_model import LinearRegression



test = pd.read_csv("../input/bikesharingdemand-dataset/test.csv")

train = pd.read_csv("../input/bikesharingdemand-dataset/train.csv")



train["date"] = train.datetime.apply(lambda x : x.split()[0])

train["hour"] = train.datetime.apply(lambda x : x.split()[1].split(":")[0]).astype("int")

train["year"] = train.datetime.apply(lambda x : x.split()[0].split("-")[0])

train["weekday"] = train.date.apply(lambda dateString : datetime.strptime(dateString,"%Y-%m-%d").weekday())

train["month"] = train.date.apply(lambda dateString : datetime.strptime(dateString,"%Y-%m-%d").month)



dropFeatures = ['casual',"count","datetime","date","registered","humidity"]

dataTrain  = train.drop(dropFeatures,axis=1)



X_train = dataTrain

y_train = train["count"]

print(X_train.shape)

print(y_train.shape)

yLablesRegistered = train["registered"]

yLablesCasual = train["casual"]



print(X_train.dtypes)

print(y_train.dtypes)

lr = LinearRegression()

lr.fit(X_train,y_train)
train["date"] = train.datetime.apply(lambda x : x.split()[0])

train["hour"] = train.datetime.apply(lambda x : x.split()[1].split(":")[0])

train["weekday"] = train.date.apply(lambda dateString : calendar.day_name[datetime.strptime(dateString,"%Y-%m-%d").weekday()])

train["month"] = train.date.apply(lambda dateString : calendar.month_name[datetime.strptime(dateString,"%Y-%m-%d").month])

train["season"] = train.season.map({1: "Spring", 2 : "Summer", 3 : "Fall", 4 :"Winter" })

train["weather"] = train.weather.map({1: " Clear + Few clouds + Partly cloudy + Partly cloudy",\

                                        2 : " Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist ", \

                                        3 : " Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds", \

                                        4 :" Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog " })


dataTest = pd.read_csv("../input/bikesharingdemand-dataset/test.csv")

dataTrain = pd.read_csv("../input/bikesharingdemand-dataset/train.csv")



data = dataTrain.append(dataTest)

data.reset_index(inplace=True)

data.drop('index',inplace=True,axis=1)



data["date"] = data.datetime.apply(lambda x : x.split()[0])

data["hour"] = data.datetime.apply(lambda x : x.split()[1].split(":")[0]).astype("int")

data["year"] = data.datetime.apply(lambda x : x.split()[0].split("-")[0])

data["weekday"] = data.date.apply(lambda dateString : datetime.strptime(dateString,"%Y-%m-%d").weekday())

data["month"] = data.date.apply(lambda dateString : datetime.strptime(dateString,"%Y-%m-%d").month)



dataTrain = data[pd.notnull(data['count'])].sort_values(by=["datetime"])

dataTest = data[~pd.notnull(data['count'])].sort_values(by=["datetime"])

datetimecol = dataTest["datetime"]

yLabels = dataTrain["count"]

yLablesRegistered = dataTrain["registered"]

yLablesCasual = dataTrain["casual"]





categoricalFeatureNames = ["season","holiday","workingday","weather","weekday","month","year","hour"]

numericalFeatureNames = ["temp","humidity","windspeed","atemp"]

dropFeatures = ['casual',"count","datetime","date","registered"]





for var in categoricalFeatureNames:

    data[var] = data[var].astype("category")

    

dataTrain = data[pd.notnull(data['count'])].sort_values(by=["datetime"])

dataTest = data[~pd.notnull(data['count'])].sort_values(by=["datetime"])

datetimecol = dataTest["datetime"]

yLabels = dataTrain["count"]

yLablesRegistered = dataTrain["registered"]

yLablesCasual = dataTrain["casual"]







dropFeatures = ['casual',"count","datetime","date","registered"]

dataTrain  = dataTrain.drop(dropFeatures,axis=1)

dataTest  = dataTest.drop(dropFeatures,axis=1)
def rmsle(y, y_,convertExp=True):

    if convertExp:

        y = np.exp(y),

        y_ = np.exp(y_)

    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))

    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))

    calc = (log1 - log2) ** 2

    return np.sqrt(np.mean(calc))
from sklearn.linear_model import LinearRegression,Ridge,Lasso

from sklearn.model_selection import GridSearchCV

from sklearn import metrics

import warnings

pd.options.mode.chained_assignment = None

warnings.filterwarnings("ignore", category=DeprecationWarning)



# Initialize logistic regression model

lModel = LinearRegression()



# Train the model

yLabelsLog = np.log1p(yLabels)

lModel.fit(X = dataTrain,y = yLabelsLog)



# Make predictions

preds = lModel.predict(X= dataTrain)

print ("RMSLE Value For Linear Regression: ",rmsle(np.exp(yLabelsLog),np.exp(preds),False))
print(dataTrain.dtypes)

print(yLabelsLog.dtypes)