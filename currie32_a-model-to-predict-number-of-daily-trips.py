import pandas as pd

import numpy as np

from scipy.stats.stats import pearsonr  

from pandas.tseries.holiday import USFederalHolidayCalendar

from pandas.tseries.offsets import CustomBusinessDay

from datetime import datetime

from sklearn.model_selection import train_test_split

import math

import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest

from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA

from sklearn.grid_search import GridSearchCV

from sklearn.pipeline import FeatureUnion

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn.cross_validation import KFold

from sklearn.metrics import mean_squared_error, median_absolute_error

import xgboost as xgb
df = pd.read_csv("../input/trip.csv")

weather = pd.read_csv("../input/weather.csv")

stations = pd.read_csv("../input/station.csv")
df.head()
df.isnull().sum()
df.duration.describe()
#Change duration from seconds to minutes

df.duration /= 60
df.duration.describe()
#I want to remove major outliers from the data; trips longer than 6 hours. This will remove less than 0.5% of the data.

df['duration'].quantile(0.995)

df = df[df.duration <= 360]
df.shape
#Convert to datetime so that it can be manipulated more easily

df.start_date = pd.to_datetime(df.start_date, format='%m/%d/%Y %H:%M')
#Extract the year, month, and day from start_date

df['date'] = df.start_date.dt.date
#Each entry in the date feature is a trip. 

#By finding the total number of times a date is listed, we know how many trips were taken on that date.

dates = {}

for d in df.date:

    if d not in dates:

        dates[d] = 1

    else:

        dates[d] += 1
#Create the data frame that will be used for training, with the dictionary we just created.

df2 = pd.DataFrame.from_dict(dates, orient = "index")

df2['date'] = df2.index

df2['trips'] = df2.ix[:,0]

train = df2.ix[:,1:3]

train.reset_index(drop = True, inplace = True)
train
#All sorted now!

train = train.sort('date')

train.reset_index(drop=True, inplace=True)
weather.head()
weather.isnull().sum()
weather.date = pd.to_datetime(weather.date, format='%m/%d/%Y')
#The weather data frame is 5 times as long as the train data frame, 

#therefore there are 5 entries per date.

print (train.shape)

print (weather.shape)
#It seems we have one entry per zip code

weather.zip_code.unique()
#Let's see which zip code has the cleanest date.

for zc in weather.zip_code.unique():

    print (weather[weather.zip_code == zc].isnull().sum())

    print ()
#I used this zip code for my other report as well. It is missing only a bit of data and is formatted rather well.

weather = weather[weather.zip_code == 94107]
weather.events.unique()
weather.loc[weather.events == 'rain', 'events'] = "Rain"

weather.loc[weather.events.isnull(), 'events'] = "Normal"
weather.events
events = pd.get_dummies(weather.events)
weather = weather.merge(events, left_index = True, right_index = True)
#Remove features we don't need

weather = weather.drop(['events','zip_code'],1)
#max_wind and max_gust are well correlated, so we can use max_wind to help fill the null values of max_gust

print (pearsonr(weather.max_wind_Speed_mph[weather.max_gust_speed_mph >= 0], 

               weather.max_gust_speed_mph[weather.max_gust_speed_mph >= 0]))
#For each value of max_wind, find the median max_gust and use that to fill the null values.

weather.loc[weather.max_gust_speed_mph.isnull(), 'max_gust_speed_mph'] = weather.groupby('max_wind_Speed_mph').max_gust_speed_mph.apply(lambda x: x.fillna(x.median()))
weather.isnull().sum()
#Change this feature from a string to numeric.

#Use errors = 'coerce' because some values currently equal 'T' and we want them to become NAs.

weather.precipitation_inches = pd.to_numeric(weather.precipitation_inches, errors = 'coerce')
#Change null values to the median, of values > 0, because T, I think, means True. 

#Therefore we want to find the median amount of precipitation on days when it rained.

weather.loc[weather.precipitation_inches.isnull(), 

            'precipitation_inches'] = weather[weather.precipitation_inches.notnull()].precipitation_inches.median()
train = train.merge(weather, on = train.date)
#Need to remove the extra date columns, otherwise good!

train.head()
train['date'] = train['date_x']

train.drop(['date_y','date_x'],1, inplace= True)
stations.head()
#Good, each stations is only listed once

print (len(stations.name.unique()))

print (stations.shape)
stations.installation_date = pd.to_datetime(stations.installation_date, format = "%m/%d/%Y").dt.date
#The min date is before any in the train data frame, therefore stations were installed before the first trips (good).

#The max date is before the end of the train data frame, therefore the service has not been adding new stations recently.

print (stations.installation_date.min())

print (stations.installation_date.max())
#For each day in train.date, find the number of docks (parking spots for individual bikes) that were installed 

#on or before that day.

total_docks = []

for day in train.date:

    total_docks.append(sum(stations[stations.installation_date <= day].dock_count))
train['total_docks'] = total_docks
#Find all of the holidays during our time span

calendar = USFederalHolidayCalendar()

holidays = calendar.holidays(start=train.date.min(), end=train.date.max())
holidays
#Find all of the business days in our time span

us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())

business_days = pd.DatetimeIndex(start=train.date.min(), end=train.date.max(), freq=us_bd)
business_days
business_days = pd.to_datetime(business_days, format='%Y/%m/%d').date

holidays = pd.to_datetime(holidays, format='%Y/%m/%d').date
#A 'business_day' or 'holiday' is a date within either of the respected lists.

train['business_day'] = train.date.isin(business_days)

train['holiday'] = train.date.isin(holidays)
train.head()
#Convert True to 1 and False to 0

train.business_day = train.business_day.map(lambda x: 1 if x == True else 0)

train.holiday = train.holiday.map(lambda x: 1 if x == True else 0)
#Convert date to the important features, year, month, weekday (0 = Monday, 1 = Tuesday...)

#We don't need day because what it represents changes every year.

train['year'] = pd.to_datetime(train['date']).dt.year

train['month'] = pd.to_datetime(train['date']).dt.month

train['weekday'] = pd.to_datetime(train['date']).dt.weekday
labels = train.trips

train = train.drop(['trips', 'date'], 1)
X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=0.2, random_state = 2)
#15 fold cross validation. Multiply by -1 to make values positive.

#Used median absolute error to learn how many trips my predictions are off by.



def scoring(clf):

    scores = cross_val_score(clf, X_train, y_train, cv=15, n_jobs=1, scoring = 'neg_median_absolute_error')

    print (np.median(scores) * -1)
rfr = RandomForestRegressor(n_estimators = 55,

                            min_samples_leaf = 3,

                            random_state = 2)

scoring(rfr)
gbr = GradientBoostingRegressor(learning_rate = 0.12,

                                n_estimators = 150,

                                max_depth = 8,

                                min_samples_leaf = 1,

                                random_state = 2)

scoring(gbr)
dtr = DecisionTreeRegressor(min_samples_leaf = 3,

                            max_depth = 8,

                            random_state = 2)

scoring(dtr)
abr = AdaBoostRegressor(n_estimators = 100,

                        learning_rate = 0.1,

                        loss = 'linear',

                        random_state = 2)

scoring(abr)
import warnings

warnings.filterwarnings("ignore")



random_state = 2

params = {

        'eta': 0.15,

        'max_depth': 6,

        'min_child_weight': 2,

        'subsample': 1,

        'colsample_bytree': 1,

        'verbose_eval': True,

        'seed': random_state,

    }



n_folds = 15 #number of Kfolds

cv_scores = [] #The sum of the mean_absolute_error for each fold.

early_stopping_rounds = 100

iterations = 10000

printN = 50

fpred = [] #stores the sums of predicted values for each fold.



testFinal = xgb.DMatrix(X_test)



kf = KFold(len(X_train), n_folds=n_folds)



for i, (train_index, test_index) in enumerate(kf):

    print('\n Fold %d' % (i+1))

    Xtrain, Xval = X_train.iloc[train_index], X_train.iloc[test_index]

    Ytrain, Yval = y_train.iloc[train_index], y_train.iloc[test_index]

    

    xgtrain = xgb.DMatrix(Xtrain, label = Ytrain)

    xgtest = xgb.DMatrix(Xval, label = Yval)

    watchlist = [(xgtrain, 'train'), (xgtest, 'eval')] 

    

    xgbModel = xgb.train(params, 

                         xgtrain, 

                         iterations, 

                         watchlist,

                         verbose_eval = printN,

                         early_stopping_rounds=early_stopping_rounds

                        )

    

    scores_val = xgbModel.predict(xgtest, ntree_limit=xgbModel.best_ntree_limit)

    cv_score = median_absolute_error(Yval, scores_val)

    print('eval-MSE: %.6f' % cv_score)

    y_pred = xgbModel.predict(testFinal, ntree_limit=xgbModel.best_ntree_limit)

    print(xgbModel.best_ntree_limit)



    if i > 0:

        fpred = pred + y_pred #sum predictions

    else:

        fpred = y_pred

    pred = fpred

    cv_scores.append(cv_score)



xgb_preds = pred / n_folds #find the average values for the predictions

score = np.median(cv_scores)

print('Median error: %.6f' % score)
#Train and make predictions with the best models.

rfr = rfr.fit(X_train, y_train)

gbr = gbr.fit(X_train, y_train)



rfr_preds = rfr.predict(X_test)

gbr_preds = gbr.predict(X_test)



#Weight the top models to find the best prediction

final_preds = rfr_preds*0.32 + gbr_preds*0.38 + xgb_preds*0.3

print ("Daily error of trip count:", median_absolute_error(y_test, final_preds))
#A reminder of the range of values in number of daily trips.

labels.describe()
y_test.reset_index(drop = True, inplace = True)
fs = 16

plt.figure(figsize=(8,5))

plt.plot(final_preds)

plt.plot(y_test)

plt.legend(['Prediction', 'Acutal'])

plt.ylabel("Number of Trips", fontsize = fs)

plt.xlabel("Predicted Date", fontsize = fs)

plt.title("Predicted Values vs Actual Values", fontsize = fs)

plt.show()
#Create a plot that ranks the features by importance.

def plot_importances(model, model_name):

    importances = model.feature_importances_

    std = np.std([model.feature_importances_ for feature in model.estimators_],

                 axis=0)

    indices = np.argsort(importances)[::-1]    



    # Plot the feature importances of the forest

    plt.figure(figsize = (8,5))

    plt.title("Feature importances of " + model_name)

    plt.bar(range(X_train.shape[1]), importances[indices], color="r", align="center")

    plt.xticks(range(X_train.shape[1]), indices)

    plt.xlim([-1, X_train.shape[1]])

    plt.show()
# Print the feature ranking

print("Feature ranking:")



i = 0

for feature in X_train:

    print (i, feature)

    i += 1

    

plot_importances(rfr, "Random Forest Regressor")

plot_importances(gbr, "Gradient Boosting Regressor")