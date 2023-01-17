import pandas as pd
import statistics
from datetime import datetime
df = pd.read_csv("../input/trip.csv")
df.head()
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
df2['trips'] = df2.iloc[:,0]
df2.head()
train = pd.DataFrame(df2.date)
train['trips'] = df2['trips']
train.head()
train.reset_index(drop = True, inplace = True)
train.head()
train = train.sort_values(by='date')
train.head()
train.tail()
type(train.date[0])
weather = pd.read_csv("../input/weather.csv")
weather.head()
weather.events.unique()
weather.loc[weather["events"] == 'rain', 'events'] = 'Rain'
weather.events.unique()
weather.loc[weather["events"].isnull(), 'events'] = 'Normal'
weather.events.unique()
weather.zip_code.unique()
for zipcode in (weather.zip_code.unique()):

    print(zipcode)

    print(weather[weather.zip_code == zipcode].isnull().sum())

    print()
weather = weather[weather.zip_code == 94107]
weather = weather.drop(['zip_code'], axis=1)
weather.max_gust_speed_mph.describe()
weather.corr()
w1 = weather.loc[:, ('max_wind_Speed_mph', 'max_gust_speed_mph')]
w1.corr()
w1_null = w1[w1.max_gust_speed_mph.isnull()]
w1_null.head()
weather.loc[weather.max_gust_speed_mph.isnull(), 'max_gust_speed_mph'] = weather.max_wind_Speed_mph
weather.max_gust_speed_mph.isnull().sum()
weather.iloc[63]
for i in weather.precipitation_inches[0:5]:

    print(type(i))
weather.precipitation_inches = pd.to_numeric(weather.precipitation_inches, errors = 'coerce')
type(weather.precipitation_inches.iloc[1])
weather.precipitation_inches.describe()
statistics.median(weather[weather.precipitation_inches.notnull()].precipitation_inches)
weather.precipitation_inches.isnull().sum()
weather.loc[weather.precipitation_inches.isnull(), 'precipitation_inches'] = 0.0
weather.precipitation_inches.isnull().sum()
weather = weather.sort_values(by = 'date')
weather.reset_index(drop = True, inplace = True)
weather.date.head()
train = train.merge(weather, on = train.date)
train.head()
train.drop(['key_0', 'date_y'],1, inplace= True)
train = train.rename(columns={'date_x':'date'})
train.head()
stations = pd.read_csv("../input/station.csv")
stations.head()
stations.city.unique()
stations = stations[stations.city == 'San Francisco']
stations.reset_index(drop = True, inplace = True)
stations.shape
stations.head()
for i in stations.installation_date[0:5]:

    print(i, type(i))
stations.installation_date.shape
stations.installation_date = pd.to_datetime(stations.installation_date)
stations['installation_date'] = stations.installation_date.dt.date
for str in stations.installation_date[0:5]:

    print(type(str))
print (stations.installation_date.min())

print (stations.installation_date.max())
#For each day in train.date, find the number of docks (parking spots for individual bikes) that were installed 

#on or before that day.

total_docks = []

for day in train.date:

    total_docks.append(sum(stations[stations.installation_date <= day].dock_count))
train['total_docks'] = total_docks
train.total_docks.unique()
from pandas.tseries.holiday import USFederalHolidayCalendar
#Find all of the holidays during out time span

calendar = USFederalHolidayCalendar()

holidays = calendar.holidays(start=train.date.min(), end=train.date.max())
holidays
from pandas.tseries.offsets import CustomBusinessDay
#Find all of the business days in our time span

us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())

business_days = pd.DatetimeIndex(start=train.date.min(), end=train.date.max(), freq=us_bd)
business_days
business_days = pd.to_datetime(business_days, format = '%Y/%m/%d').date
# if train.date is a business day or not

train['business_days'] = train.date.isin(business_days)
train['business_days'].head()
holidays = pd.to_datetime(holidays, format = '%Y/%m/%d').date
# if train.date is a holiday or not

train['holidays'] = train.date.isin(holidays)
train['holidays'].head()
weekday = []

for i in train.date:

    wkday = i.weekday()

#    print(wkday)

    if wkday in range(0,5):

        weekday.append(1)

#        print(1)

    else:

        weekday.append(0)

#        print(0)
train['weekday'] = weekday
train.head()
train.business_days = [1 if i is True else 0 for i in train.business_days ]
train.holidays = [1 if i is True else 0 for i in train.holidays ]
train.head()
train['month'] = pd.to_datetime(train.date).dt.month
train.head()
labels = train.trips
train.drop(['date', 'trips'],1, inplace = True)
train.tail()
events = pd.get_dummies(train.events, drop_first = True)
train = train.merge(events, left_index = True, right_index = True)
train.head()
train.drop(['events'], axis = 1, inplace=True)
train.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=0.2, random_state = 1)
import math

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict
regressor = LinearRegression()
predicted = cross_val_predict(regressor, X_train, y_train, cv=15)
import matplotlib.pyplot as plt

fig,ax = plt.subplots()

ax.scatter(y_train, predicted, edgecolors = (0,0,0))

ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4)

ax.set_xlabel('Measured')

ax.set_ylabel('Predicted')

plt.show()
scoring = ['r2','neg_mean_squared_error','neg_mean_absolute_error']

for i in scoring:

    scores = cross_val_score(regressor, X_train, y_train, cv=15, scoring = i)

#    print(scores)

    if i == 'r2':

        print(i, ': ', scores.mean())

    elif i == 'neg_mean_squared_error':    

        x = -1*scores.mean()

        y = math.sqrt(x) 

        print('RMSE: ', "%0.2f" % y)

    elif i == 'neg_mean_absolute_error':

        x = -1*scores.mean()

        print(i, ": %0.2f (+/- %0.2f)" % (x, scores.std() * 2))   
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators = 55,

                            min_samples_leaf = 3,

                            random_state = 2, bootstrap=False)
predicted = cross_val_predict(rfr, X_train, y_train, cv=15)
fig,ax = plt.subplots()

ax.scatter(y_train, predicted, edgecolors = (0,0,0))

ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4)

ax.set_xlabel('Measured')

ax.set_ylabel('Predicted')

plt.show()
scoring = ['r2','neg_mean_squared_error','neg_mean_absolute_error']

for i in scoring:

    scores = cross_val_score(rfr, X_train, y_train, cv=15, scoring = i)

#    print(scores)

    if i == 'r2':

        print(i, ': ', scores.mean())

    elif i == 'neg_mean_squared_error':    

        x = -1*scores.mean()

        y = math.sqrt(x) 

        print('RMSE: ', "%0.2f" % y)

    elif i == 'neg_mean_absolute_error':

        x = -1*scores.mean()

        print(i, ": %0.2f (+/- %0.2f)" % (x, scores.std() * 2))   
rfr1 = RandomForestRegressor(n_estimators=60, criterion='mse', random_state=2)
predicted = cross_val_predict(rfr1, X_train, y_train, cv=15)
fig,ax = plt.subplots()

ax.scatter(y_train, predicted, edgecolors = (0,0,0))

ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4)

ax.set_xlabel('Measured')

ax.set_ylabel('Predicted')

plt.show()
scoring = ['r2','neg_mean_squared_error','neg_mean_absolute_error']

for i in scoring:

    scores = cross_val_score(rfr1, X_train, y_train, cv=15, scoring = i)

#    print(scores)

    if i == 'r2':

        print(i, ': ', scores.mean())

    elif i == 'neg_mean_squared_error':    

        x = -1*scores.mean()

        y = math.sqrt(x) 

        print('RMSE: ', "%0.2f" % y)

    elif i == 'neg_mean_absolute_error':

        x = -1*scores.mean()

        print(i, ": %0.2f (+/- %0.2f)" % (x, scores.std() * 2))   
from sklearn.neighbors import KNeighborsRegressor

neigh = KNeighborsRegressor(n_neighbors=2)

neigh.fit(X_train, y_train) 
scoring = ['r2','neg_mean_squared_error','neg_mean_absolute_error']

for i in scoring:

    scores = cross_val_score(neigh, X_train, y_train, cv=15, scoring = i)

#    print(scores)

    if i == 'r2':

        print(i, ': ', scores.mean())

    elif i == 'neg_mean_squared_error':    

        x = -1*scores.mean()

        y = math.sqrt(x) 

        print('RMSE: ', "%0.2f" % y)

    elif i == 'neg_mean_absolute_error':

        x = -1*scores.mean()

        print(i, ": %0.2f (+/- %0.2f)" % (x, scores.std() * 2))   
neigh1 = KNeighborsRegressor(n_neighbors=3)

neigh1.fit(X_train, y_train) 
scoring = ['r2','neg_mean_squared_error','neg_mean_absolute_error']

for i in scoring:

    scores = cross_val_score(neigh1, X_train, y_train, cv=15, scoring = i)

#    print(scores)

    if i == 'r2':

        print(i, ': ', scores.mean())

    elif i == 'neg_mean_squared_error':    

        x = -1*scores.mean()

        y = math.sqrt(x) 

        print('RMSE: ', "%0.2f" % y)

    elif i == 'neg_mean_absolute_error':

        x = -1*scores.mean()

        print(i, ": %0.2f (+/- %0.2f)" % (x, scores.std() * 2))   
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor(learning_rate = 0.12,

                                n_estimators = 150,

                                max_depth = 8,

                                min_samples_leaf = 1,

                                random_state = 2)
scoring = ['r2','neg_mean_squared_error','neg_mean_absolute_error']

for i in scoring:

    scores = cross_val_score(gbr, X_train, y_train, cv=15, scoring = i)

#    print(scores)

    if i == 'r2':

        print(i, ': ', scores.mean())

    elif i == 'neg_mean_squared_error':    

        x = -1*scores.mean()

        y = math.sqrt(x) 

        print('RMSE: ', "%0.2f" % y)

    elif i == 'neg_mean_absolute_error':

        x = -1*scores.mean()

        print(i, ": %0.2f (+/- %0.2f)" % (x, scores.std() * 2))   
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(min_samples_leaf = 3,

                            max_depth = 8,

                            random_state = 2)
scoring = ['r2','neg_mean_squared_error','neg_mean_absolute_error']

for i in scoring:

    scores = cross_val_score(dtr, X_train, y_train, cv=15, scoring = i)

#    print(scores)

    if i == 'r2':

        print(i, ': ', scores.mean())

    elif i == 'neg_mean_squared_error':    

        x = -1*scores.mean()

        y = math.sqrt(x) 

        print('RMSE: ', "%0.2f" % y)

    elif i == 'neg_mean_absolute_error':

        x = -1*scores.mean()

        print(i, ": %0.2f (+/- %0.2f)" % (x, scores.std() * 2))
from sklearn.ensemble import AdaBoostRegressor
abr = AdaBoostRegressor(n_estimators = 100,

                        learning_rate = 0.1,

                        loss = 'linear',

                        random_state = 2)

scoring = ['r2','neg_mean_squared_error','neg_mean_absolute_error']

for i in scoring:

    scores = cross_val_score(abr, X_train, y_train, cv=15, scoring = i)

#    print(scores)

    if i == 'r2':

        print(i, ': ', scores.mean())

    elif i == 'neg_mean_squared_error':    

        x = -1*scores.mean()

        y = math.sqrt(x) 

        print('RMSE: ', "%0.2f" % y)

    elif i == 'neg_mean_absolute_error':

        x = -1*scores.mean()

        print(i, ": %0.2f (+/- %0.2f)" % (x, scores.std() * 2))     
rfr1.fit(X_train, y_train )

predicted = rfr1.predict(X_test)
labels.describe()
y_test.reset_index(drop = True, inplace = True)
plt.figure(figsize=(10,7))

plt.plot(predicted)

plt.plot(y_test)

plt.legend(['Prediction', 'Acutal'])

plt.ylabel("Number of Trips", fontsize = 14)

plt.xlabel("Predicted Date", fontsize = 14)

plt.title("Predicted Values vs Actual Values", fontsize = 14)

plt.show()