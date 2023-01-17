# libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import datetime

import pytz

from sklearn import linear_model



%matplotlib inline
# import data

data = pd.read_csv('../input/SolarPrediction.csv')



# read few lines

data.head()
# time is in reverse order, therefore order ascending

data = data.sort_values(by='UNIXTime', ascending=True).reset_index(drop=True)

data.head()
# set Hawaii tz

HItz = pytz.timezone(zone='US/Hawaii')



# create a column which is type datetime

datetimeHI = data['UNIXTime'].apply(lambda x: 

                                 datetime.datetime.utcfromtimestamp(x).replace(tzinfo=pytz.utc).astimezone(HItz))

# add to df

data['DatetimeHI'] = datetimeHI
# show data to predict

plt.plot(data['DatetimeHI'], data['Radiation'])

plt.title('Radiation Between Sept 2016 and Dec 2016')

plt.xticks(rotation=45);
# extract one week of data

weekendmarker = datetime.datetime(2016,9, 8).replace(tzinfo=HItz)

weekonedata = data[data['DatetimeHI'] < weekendmarker]

plt.plot(weekonedata['DatetimeHI'], weekonedata['Radiation'])

plt.title('Radiation 9/1/2016 - 9/8/2016')

plt.xticks(rotation=45)

plt.ylabel('Radiation Level')

plt.xlabel('Date');
def abstract_week_plot(ax, dates, col, colname):

    # function to take in column of data and plot the

    # week's worth of data

    # returns an axis so can be added to a larger plot

    

    # color radiation so it is easy to identify as the dependent var

    if colname == 'Radiation':

        plt_color = 'red'

    else:

        plt_color = 'blue'

    

    # plot the data

    ax.plot(dates, col, c=plt_color)

    

    # format

    ax.set_title('{colname} 9/1/2016 - 9/8/2016'.format(colname=colname))

    ax.set_ylabel('{colname} Level'.format(colname=colname))

    ax.set_xlabel('Date')

    

    # rotation

    for tick in ax.get_xticklabels():

        tick.set_rotation(45)

    

    return ax
# make plot larger

plt.rcParams['figure.figsize'] = 16, 12



# loop over all columns important in data

ts_cols = ['Radiation', 'Temperature', 'Pressure', 'Humidity', 'WindDirection(Degrees)', 'Speed']

fig, axes = plt.subplots(len(ts_cols), sharex=True)

for i, ax in enumerate(axes):

    ax = abstract_week_plot(ax, weekonedata['DatetimeHI'], weekonedata[ts_cols[i]], ts_cols[i])



# prevent squashing

fig.tight_layout()
def abstract_subsection_plot(ax, data, colname, start_dt, end_dt):

    # function to take in column of data and plot the

    # week's worth of data

    # returns an axis so can be added to a larger plot

    

    # subset the data

    subset_data = data[(data['DatetimeHI'] > start_dt) & (data['DatetimeHI'] < end_dt)]

    dates = subset_data['DatetimeHI']

    col = subset_data[colname]

    

    # turn start date and end date into strings

    srt = start_dt.strftime('%m/%d/%Y')

    end = end_dt.strftime('%m/%d/%Y')

    

    # color radiation so it is easy to identify as the dependent var

    if colname == 'Radiation':

        plt_color = 'red'

    else:

        plt_color = 'blue'

    

    # plot the data

    ax.plot(dates, col, c=plt_color)

    

    # format

    ax.set_title('{colname} {srt} - {end}'.format(colname=colname, srt=srt, end=end))

    ax.set_ylabel('{colname} Level'.format(colname=colname))

    ax.set_xlabel('Date')

    

    # rotation

    for tick in ax.get_xticklabels():

        tick.set_rotation(45)

    

    return ax
# loop over all columns important in data

fig, axes = plt.subplots(len(ts_cols), sharex=True)

for i, ax in enumerate(axes):

    ax = abstract_subsection_plot(

        ax,

        data,

        ts_cols[i],

        datetime.datetime(2016,11, 27).replace(tzinfo=HItz),

        datetime.datetime(2016,12, 15).replace(tzinfo=HItz),

    )

# prevent squashing

fig.tight_layout()
# loop over all columns important in data

fig, axes = plt.subplots(len(ts_cols), sharex=True)

for i, ax in enumerate(axes):

    ax = abstract_subsection_plot(

        ax,

        data,

        ts_cols[i],

        datetime.datetime(2016,9, 8).replace(tzinfo=HItz),

        datetime.datetime(2016,10, 3).replace(tzinfo=HItz),

    )

# prevent squashing

fig.tight_layout()
start_train = datetime.datetime(2016, 9, 1).replace(tzinfo=HItz)

end_train = datetime.datetime(2016,11, 29).replace(tzinfo=HItz)

start_test = datetime.datetime(2016,12, 9).replace(tzinfo=HItz)

end_test = datetime.datetime(2016,12, 31).replace(tzinfo=HItz)
def is_day(row):

    sun_rise = datetime.datetime.strptime(row['TimeSunRise'], '%H:%M:%S').time()

    sun_set = datetime.datetime.strptime(row['TimeSunSet'], '%H:%M:%S').time()

    if ((sun_set > row['DatetimeHI'].time()) & (sun_rise < row['DatetimeHI'].time())):

        return 1

    else:

        return 0

    

day_bool = np.empty(data.shape[0])



for i in np.arange(data.shape[0]):

    day_bool[i] = is_day(data.iloc[i])
day_bool
data['Day'] = day_bool
# add interaction terms for Day

data['Day x Temperature'] = data['Temperature'] * day_bool

data['Day x Pressure'] = data['Pressure'] * day_bool

data['Day x Humidity'] = data['Humidity'] * day_bool

data['Day x WindDirection(Degrees)'] = data['WindDirection(Degrees)'] * day_bool

data['Day x Speed'] = data['Speed'] * day_bool
data.head()
from datetime import timedelta

import time



# create a series of number of hours since sunrise

# if not Day, then 0

hour_of_day = np.empty((data.shape[0], ))



for ix in data.index:

    # sunrise, sunset; probably a fiddler on the roof joke in there somewhere

    sr = datetime.datetime.strptime(data.loc[ix, 'TimeSunRise'], '%H:%M:%S').replace(tzinfo=HItz)

    ss = datetime.datetime.strptime(data.loc[ix, 'TimeSunSet'], '%H:%M:%S').replace(tzinfo=HItz)

    

    # if night, 0

    if ((data.loc[ix, 'DatetimeHI'].time() > ss.time()) | (data.loc[ix, 'DatetimeHI'].time() < sr.time())):

        hour_of_day[ix] = 0.

    else:

        time_ix = data.loc[ix, 'DatetimeHI'].time()

    

        # need to account for minutes

        # sunrise of 6:59 and time of 7:01 is closer to 0 hours apart than 1

        if (time_ix.hour - sr.hour > 0) & (time_ix.minute - sr.minute < 30):

            hour_of_day[ix] = time_ix.hour - sr.hour - 1

        else:

            hour_of_day[ix] = time_ix.hour - sr.hour
hour_of_day
from scipy import stats

stats.describe(hour_of_day)
# have a look at dataframe to add

pd.get_dummies(hour_of_day).head()
# add this to the full data frame

data = pd.concat([data, pd.get_dummies(hour_of_day)], axis=1)
TRAIN = data[(data['DatetimeHI'] > start_train) & (data['DatetimeHI'] < end_train)]

TEST = data[(data['DatetimeHI'] > start_test) & (data['DatetimeHI'] < end_test)]
# split training and testing into X and y for compatibility with sklearn

X_train = TRAIN.drop(['UNIXTime', 'Data', 'Time', 'Radiation', 'TimeSunRise', 'TimeSunSet', 'DatetimeHI'], axis=1)

X_test = TEST.drop(['UNIXTime', 'Data', 'Time', 'Radiation', 'TimeSunRise', 'TimeSunSet', 'DatetimeHI'], axis=1)

y_train = TRAIN['Radiation']

y_test = TEST['Radiation']
lin_reg = linear_model.LinearRegression()

lin_reg.fit(X_train, y_train)
lin_reg.score(X=X_test, y=y_test)
y_pred = lin_reg.predict(X_test)
# resize

plt.rcParams['figure.figsize'] = 10, 8



# plot results

plt.plot(TEST['DatetimeHI'], y_pred, c='blue', label='Predicted')

plt.plot(TEST['DatetimeHI'], y_test, c='red', label='Observed')

plt.title('Observed vs Predicted')

plt.ylabel('Radiation')

plt.xlabel('Date');
# perform a backwards stepwise regression

from sklearn.feature_selection import RFE



# fit reduced model

reg = linear_model.LinearRegression()

reduced_reg = RFE(reg)

reduced_reg.fit(X_train, y_train)
reduced_reg.score(X=X_test, y=y_test)
# predict using the reduced model

y_pred = reduced_reg.predict(X_test)



# plot results

plt.plot(TEST['DatetimeHI'], y_pred, c='blue', label='Predicted')

plt.plot(TEST['DatetimeHI'], y_test, c='red', label='Observed')

plt.title('Observed vs Predicted')

plt.ylabel('Radiation')

plt.xlabel('Date');
from sklearn.tree import DecisionTreeRegressor as DTR



# fit random forest

dt = DTR()

dt.fit(X_train, y_train)
dt.score(X=X_test, y=y_test)
# predict using the random forest model

y_pred = dt.predict(X_test)



# plot results

plt.plot(TEST['DatetimeHI'], y_pred, c='blue', label='Predicted')

plt.plot(TEST['DatetimeHI'], y_test, c='red', label='Observed')

plt.title('Observed vs Predicted')

plt.ylabel('Radiation')

plt.xlabel('Date');
from sklearn.ensemble import RandomForestRegressor as RFR



# set seed for consistency

np.random.seed(171)



# fit random forest

random_forest = RFR()

random_forest.fit(X_train, y_train)
random_forest.score(X=X_test, y=y_test)
# predict using the random forest model

y_pred = random_forest.predict(X_test)



# plot results

plt.plot(TEST['DatetimeHI'], y_pred, c='blue', label='Predicted')

plt.plot(TEST['DatetimeHI'], y_test, c='red', label='Observed')

plt.title('Observed vs Predicted')

plt.ylabel('Radiation')

plt.xlabel('Date');
# depths to search

rf_depths = np.arange(5) + 1



# number of esimators to search

rf_estimators = np.linspace(1, 301, num=16)
rf_score_grid = np.empty((len(rf_depths), len(rf_estimators)))



# loop through sc

for i, depth in enumerate(rf_depths):

    for j, est in enumerate(rf_estimators):

        rf_ = RFR(max_depth=int(rf_depths[i]), n_estimators=int(rf_estimators[j]))

        rf_.fit(X_train, y_train)

        rf_score_grid[i, j] = rf_.score(X_test, y_test)



# display results

rf_score_grid
import seaborn as sns



# place in dataframe

rf_score_grid = pd.DataFrame(

    rf_score_grid,

    columns=[str(i) for i in rf_estimators],

    index=[str(i) for i in rf_depths]

)



# display as heatmap

sns.heatmap(rf_score_grid)

plt.title('Heatmap of Score for Tuned Random Forest');
rf_score_grid
# predict using the tuned random forest model

rf = RFR(max_depth=5, n_estimators=161)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)



# resize

plt.rcParams['figure.figsize'] = 14, 12



# plot results

plt.plot(TEST['DatetimeHI'], y_pred, c='blue', label='Predicted')

plt.plot(TEST['DatetimeHI'], y_test, c='red', label='Observed')

plt.title('Observed vs Predicted Tuned Random Forest')

plt.ylabel('Radiation')

plt.xlabel('Date');
from sklearn.ensemble import AdaBoostRegressor



# fit boosted tree

boosted_tree = AdaBoostRegressor(DTR(max_depth=3), n_estimators=200)

boosted_tree.fit(X_train, y_train)



# score

boosted_tree.score(X=X_test, y=y_test)
# predict using the random forest model

y_pred = boosted_tree.predict(X_test)



# plot results

plt.plot(TEST['DatetimeHI'], y_pred, c='blue', label='Predicted')

plt.plot(TEST['DatetimeHI'], y_test, c='red', label='Observed')

plt.title('Observed vs Predicted')

plt.ylabel('Radiation')

plt.xlabel('Date');
from tqdm import tqdm, trange



boost_depths = rf_depths 

boost_estimators = rf_estimators



boost_score_grid = np.empty((len(boost_depths), len(boost_estimators)))



# loop through sc ; tqdm will give a progress bar

for i, depth in enumerate(tqdm(boost_depths, total=len(boost_depths))):

    for j, est in enumerate(boost_estimators):

        boost_ = AdaBoostRegressor(DTR(max_depth=int(boost_depths[i])), n_estimators=int(boost_estimators[j]))

        boost_.fit(X_train, y_train)

        boost_score_grid[i, j] = boost_.score(X_test, y_test)



# display results

boost_score_grid
# place in dataframe

boost_score_grid = pd.DataFrame(

    boost_score_grid,

    columns=[str(i) for i in boost_estimators],

    index=[str(i) for i in boost_depths]

)



# display as heatmap

sns.heatmap(boost_score_grid)

plt.title('Heatmap of Score for Boosted Decision Tree');
# predict using the boosted

bt = AdaBoostRegressor(DTR(max_depth=5), n_estimators=181)

bt.fit(X_train, y_train)

y_pred = bt.predict(X_test)



# plot results

plt.plot(TEST['DatetimeHI'], y_pred, c='blue', label='Predicted')

plt.plot(TEST['DatetimeHI'], y_test, c='red', label='Observed')

plt.title('Observed vs Predicted')

plt.ylabel('Radiation')

plt.xlabel('Date');