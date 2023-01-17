# importing necessary libraries

import pandas as pd

import numpy as np





import matplotlib.pyplot as plt

import seaborn as sns

# to show the figures in the jupyter notebook itself

%matplotlib inline 
train = pd.read_csv('../input/bike-sharing-demand/train.csv', parse_dates=['datetime']) # loading the training data

# looking at the training data from Top 5

train.head()
# looking at the training data from end

train.tail() 
train.plot.scatter(x = 'season', y = 'count') # plotting the counts based on the season
train.plot.scatter(x = 'holiday', y = 'count') # plotting the counts based on the holidays
train.plot.scatter(x = 'workingday', y = 'count') # plotting the counts based on working day
train.plot.scatter(x = 'weather', y = 'count') # plotting the counts based on the weather
train.plot.scatter(x = 'temp', y = 'count') # plotting the counts based on the temparature
train.plot.scatter(x = 'atemp', y = 'count') # plotting the counts based on atemp (atemp - "feels like" temperature in Celsius)
train.plot.scatter(x = 'humidity', y = 'count')# plotting the counts based on humidity
train.plot.scatter(x = 'windspeed', y = 'count') # plotting the counts based on windspeed
train.plot.scatter(x = 'casual', y = 'count')# plotting the counts based casual user
train.info() # observing the data types of the columns
 # Generate descriptive statistics that summarize the central tendency,dispersion and shape of a dataset's distribution

train.describe()
test = pd.read_csv('../input/bike-sharing-demand/test.csv') # loading the test data

test.head()  #looking at the 1st 5 rows of the test data

test.tail() # last 5 rows of the test data
test.info()# observing the data types of the columns for test data

test.describe()
# installing the pandas profiling library. It is used for a deeper understanding than the normal Dataframe.describe() method

!pip install pandas-profiling
import pandas_profiling
train.profile_report()
print("count samples & features: ", train.shape) # printing the number of rows and columns

print("Are there missing values: ", train.isnull().values.any()) # printing if dataset has any NaN value

# method for creating the count plot based on hour for a given year 

def plot_by_hour(data, year=None, agg='sum'):

    dd = data

    if year: dd = dd[ dd.datetime.dt.year == year ]

    # extracting the hour data if the year in the data is equal to the year passed as argument    

    dd.loc[:, ('hour')] = dd.datetime.dt.hour

    

    # groupby hour and working day

    by_hour = dd.groupby(['hour', 'workingday'])['count'].agg(agg).unstack() 

    # returning the figure grouped by hour

    return by_hour.plot(kind='bar', ylim=(0, 80000), figsize=(15,5), width=0.9, title="Year = {0}".format(year)) 





plot_by_hour(train, year=2011) # plotting the count plot based on hour for 2011 

plot_by_hour(train, year=2012) # plotting the count plot based on hour for 2012
# method for creating the count plot based on year 

def plot_by_year(agg_attr, title):

    # extracting the required fields

    dd = train.copy()

    dd['year'] = train.datetime.dt.year # extratcing the year

    dd['month'] = train.datetime.dt.month # extratcing the month

    dd['hour'] = train.datetime.dt.hour # extratcing the hour

    

    by_year = dd.groupby([agg_attr, 'year'])['count'].agg('sum').unstack() # groupby year

    return by_year.plot(kind='bar', figsize=(15,5), width=0.9, title=title) # returning the figure grouped by year





plot_by_year('month', "Rent bikes per month in 2011 and 2012") # plotting monthly bike rentals based on year

plot_by_year('hour', "Rent bikes per hour in 2011 and 2012") # plotting hourls bike rentals based  on year

# method to plot a graph for count per hour

def plot_hours(data, message = ''):

    dd = data.copy()

    dd['hour'] = data.datetime.dt.hour # extratcing the hour

    

    hours = {}

    for hour in range(24):

        hours[hour] = dd[ dd.hour == hour ]['count'].values



    plt.figure(figsize=(20,10))

    plt.ylabel("Count rent")

    plt.xlabel("Hours")

    plt.title("count vs hours\n" + message)

    plt.boxplot( [hours[hour] for hour in range(24)] )

    

    axis = plt.gca()

    axis.set_ylim([1, 1100])

 
plot_hours( train[train.datetime.dt.year == 2011], 'year 2011') # box plot for hourly count for the mentioned year

plot_hours( train[train.datetime.dt.year == 2012], 'year 2012') # box plot for hourly count for the mentioned year
dt = pd.to_datetime(train["datetime"]) # converting the column to datetime for train dataset

train["hour"] = dt.map(lambda x: x.hour) # adding the hour column for train dataset

train.head()
dt_test = pd.to_datetime(test["datetime"]) # converting the column to datetime for test dataset

test["hour"] = dt_test.map(lambda x: x.hour) # adding the hour column for test dataset

test.head()


plot_hours( train[train.workingday == 1], 'working day') # plotting hourly count of rented bikes for working days for a given year

plot_hours( train[train.workingday == 0], 'non working day') # plotting hourly count of rented bikes for non-working days for a given year
# method to convert categorical data to numerical data

def categorical_to_numeric(x):

    if 0 <=  x < 6:

        return 0

    elif 6 <= x < 13:

        return 1

    elif 13 <= x < 19:

        return 2

    elif 19 <= x < 24:

        return 3
train['hour'] = train['hour'].apply(categorical_to_numeric)# applying the above conversion logic to training data

train.head()

test['hour'] = test['hour'].apply(categorical_to_numeric) # applying the above conversion logic to test data

test.head()

# drop unnecessary columns



train = train.drop(['datetime'], axis=1)

test = test.drop(['datetime'], axis=1)
train.head()
# an Hour vs Count Graph depicting average bike demand based on the hour 

figure,axes = plt.subplots(figsize = (10, 5))

hours = train.groupby(["hour"]).agg("mean")["count"]  

hours.plot(kind="line", ax=axes) 

plt.title('Hours VS Counts')

axes.set_xlabel('Time in Hours')

axes.set_ylabel('Average of the Bike Demand')

plt.show()
# count of different temp values

a = train.groupby('temp')[['count']].mean()

a
a.plot()

plt.show()
# count of different atemp values

a = train.groupby('atemp')[['count']].mean()

a
a.plot()

plt.show()
# count based on holiday

a = train.groupby('holiday')[['count']].mean()

a.plot()

plt.show()
# method to  select the features. If a feature is not in the blacklist, it gets selected

def select_features(data):

    black_list = ['casual', 'registered', 'count', 'is_test', 'datetime', 'count_log']

    return [feat for feat in data.columns if feat not in black_list]

# a method to show results of various model and their predictions

def _simple_modeling(X_train, X_test, y_train, y_test):

    # sepcifying the model names

    models = [

        ('dummy-mean', DummyRegressor(strategy='mean')),

        ('dummy-median', DummyRegressor(strategy='median')),

        ('random-forest', RandomForestRegressor(random_state=0)),

    ]

    

    results = []



    for name, model in models:

        model.fit(X_train, y_train)# fitting the training data to model

        y_pred = model.predict(X_test) # doing predictions using the model

        

        results.append((name, y_test, y_pred)) # creating the list of predictions from various models

        

    return results



# a method to return the performance metric of the model used in the above method

def simple_modeling(X_train, X_test, y_train, y_test):

    results = _simple_modeling(X_train, X_test, y_train, y_test) # using the function defined above to caluclate the predictions

    

    return [ (r[0], rmsle(r[1], r[2]) ) for r in results] # returning the performance metrics



from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score

forest_reg = RandomForestRegressor(n_estimators=100) # instantiating the random Forest Regressor



score = cross_val_score(forest_reg, train, train, cv=4) # calcuating the cross validation score

print (score)
