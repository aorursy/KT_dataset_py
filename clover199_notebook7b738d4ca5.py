# load python modules 

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

from time import time

from datetime import datetime

import pandas as pd



from statsmodels.api import OLS as lm

from sklearn.ensemble import RandomForestRegressor as rf

from pandas.tseries.holiday import USFederalHolidayCalendar
# load data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
print("shape (train): {:d} obs. {:d} features in train".format(*train.shape))

print("shape (test): {:d} obs. {:d} features in test".format(*test.shape))
print("\t\t\t", "train\t\t\t","test")

print("starting time:", pd.to_datetime(train['pickup_datetime']).min(), \

     pd.to_datetime(test['pickup_datetime']).min())

print("ending time:  ", pd.to_datetime(train['dropoff_datetime']).max())
# define some functions that will be used later



def Manhattan_distance(data, direct):

    """ claculate direct distance (optional) and Manhattan distance

        and add column log10(distance) to data """

    lat = (data['pickup_latitude']+data['dropoff_latitude'])/2

    dx = (data['dropoff_longitude'] - data['pickup_longitude'])*np.cos(lat)

    dy = data['dropoff_latitude'] - data['pickup_latitude']

    theta = np.pi*30/180

    rotate = np.array([[np.cos(theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]])

    loc = np.dot(np.array([dx,dy]).T, rotate)

    if direct:

        data['distance'] = np.log10(np.sqrt((loc[:,0])**2 + (loc[:,1])**2)+1e-6)

    data['Manhattan_dist'] = np.log10(np.abs(loc[:,0]) + np.abs(loc[:,1])+1e-6)

    

def speed(data):

    """ calculate average speed and add column log10(speed) to data """

    Manhattan_distance(data, False)

    data['speed'] = data['Manhattan_dist']-np.log10(data['trip_duration'])

    

def separate_time(data):

    """ extract weekday, time from datetime and add corresponding columns to data """

    pickup_time = pd.DatetimeIndex(pd.to_datetime(data["pickup_datetime"]))

    data['pickup_date'] = pickup_time.weekday

    data['pickup_hour'] = pickup_time.hour

    try:

        data['dropoff_hour'] = pd.DatetimeIndex(pd.to_datetime(data["dropoff_datetime"])).hour

    except KeyError:

        pass

    

def remove_exotic(data):

    """ remove exotic data according to criteria in the section about outliers """

    Manhattan_distance(data, False)

    select = (data['passenger_count']>0) & (data["Manhattan_dist"]>-5.9)

    try:

        return data[(data['trip_duration']<3600*8) & select].copy()

    except KeyError:

        pass

    return data[select].copy()



def preprocess(data):

    """ remove exotic data, change data format and add new features necessary """

    Manhattan_distance(data, False)

    data['store_and_fwd_flag'] = np.where(data['store_and_fwd_flag']=="Y", 1, 0).ravel()

    separate_time(data)

    try:

        data['log_duration'] = np.log(data['trip_duration']+1)

        data['speed'] = data['Manhattan_dist']-np.log10(data['trip_duration'])

    except KeyError:

        pass
# first look at the training data

train.head()
test.head()
# check vendor_id

counts = pd.DataFrame({"train": train['vendor_id'].value_counts(), \

                       "test": test['vendor_id'].value_counts()})

counts.plot.bar(stacked=True)

plt.title("vendor_id")

plt.show()
# check passenger_count

counts = pd.DataFrame({"train": train['passenger_count'].value_counts(), \

                       "test": test['passenger_count'].value_counts()})

counts.plot.bar(stacked=True)

for i in range(counts.shape[0]):

    plt.text(i, counts.sum(axis=1).iat[i], int(counts.sum(axis=1).iat[i]), \

             rotation=0, ha='center', va='bottom')

plt.title("passenger_count")

plt.show()
# check store_and_fwd_flag

counts = pd.DataFrame({"train": train['store_and_fwd_flag'].value_counts(), \

                       "test": test['store_and_fwd_flag'].value_counts()})

counts.plot.bar(stacked=True)

for i in range(counts.shape[0]):

    plt.text(i, counts.sum(axis=1).iat[i], int(counts.sum(axis=1).iat[i]), \

             ha='center', va='bottom')

plt.title("store_and_fwd_flag")

plt.show()
plt.figure(figsize=(4,8))

plotdata = test[(-74.02<test['pickup_longitude']) &  (test['pickup_longitude']<-73.92) & \

                 (40.7<test['pickup_latitude']) & (test['pickup_latitude']<40.85)]

locy = np.array(plotdata['pickup_latitude'])

locx = (np.array(plotdata['pickup_longitude'])+74)*np.cos(locy*np.pi/180)

loc = np.array([locx, locy]).T

theta = np.pi*30/180

rotate = np.array([[np.cos(theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]])

loc_new = np.dot(loc, rotate)

plt.scatter(loc_new[:,0], loc_new[:,1], s=0.1, c='y')

plt.xlim([-20.39,-20.34])

plt.ylim([35.24,35.37])

plt.xticks([])

plt.yticks([])

plt.show()
# trip duration distribution

plt.figure(figsize=(12,4))

plt.title("distribution of log10(trip duration)")

plt.hist(np.log10(train['trip_duration']), bins=200)

plt.show()
# trip distance distribution

Manhattan_distance(train, True)

plt.figure(figsize=(12,4))

plt.title("distribution of log10(trip distance)")

plt.hist([train['distance']+np.log10(2)/2,train['Manhattan_dist']], bins=100, \

         label=['direct*sqrt(2)','Manhattan'])

plt.legend(loc='best')

plt.show()
# trip speed distribution

speed(train)

plt.figure(figsize=(12,4))

plt.title("distribution of log10(trip speed)")

plt.hist(train['speed'], bins=300)

plt.show()
train.sample(n=100000).plot.scatter(x='Manhattan_dist', y='speed', alpha=0.1)

plt.show()
separate_time(train)

temp = train.loc[train['pickup_hour']==train['dropoff_hour'], ['speed','pickup_hour']]

plt.figure()

ax = plt.subplot(111, projection='polar')

ax.set_rticks([-4.4,-4.2])

ax.set_rlim([-4.6,-4.1])

ax.set_theta_offset(np.pi/2)

ax.set_theta_direction(-1)

ax.set_xticks(np.linspace(0,2*np.pi,6, endpoint=False))

ax.set_xticklabels(np.linspace(0,24,6, endpoint=False))

x = np.linspace(0,2*np.pi, 25, endpoint=True)[:24]

s = temp.groupby('pickup_hour').count().values.ravel()/temp.shape[0]*24*50

ax.scatter(x, temp.groupby('pickup_hour').mean().values.ravel(), s=s)

ax.set_title("average speed during a day\n(shape represents counts)\n")

plt.show()
def plot(temp, title):

    fig = plt.figure(figsize=(8,4))

    ax = fig.add_axes([0,0,1,1])

    ax.set_xlim([-0.5,6.5])

    ax.set_ylim([-4.9,-3.7])

    ax.set_xticks(np.arange(7))

    ax.set_ylabel("log10(speed)")

    ax.set_xticklabels(['Sun','Mon','Tue','Wed','Thu','Fri','Sat'])

    ax.axhline(temp['speed'].median(), linestyle='--', color='k')

    ax.plot(np.arange(7), temp.groupby('pickup_date').median(), 'o-', label='Median')

    ax.plot(np.arange(7), temp.groupby('pickup_date').mean(), 'o-', label='Mean')

    ax.set_title(title)

    ax.legend()

    temp2 = temp[(temp['speed']<-3.7)&(temp['speed']>-4.9)]

    for i in range(6,-1,-1):

        ax = fig.add_axes([i/7,0,0.2,1])

        ax.axis('off')

        ax.patch.set_alpha(0)

        ax.set_xlim([0,temp.shape[0]/7.0/10])

        ax.set_ylim([-4.9,-3.7])

        ax.hist(temp2.loc[temp2['pickup_date']==i, 'speed'], \

                          orientation='horizontal', bins=80, alpha=0.5)

#         sns.kdeplot(temp.loc[temp['pickup_date']==i,'speed'], \

#                     shade=True, legend=False, vertical=True)

    plt.show()

temp = train.loc[train['trip_duration']<3600, ['speed','pickup_date']]

plot(temp, "Trips of duration less than 1h (sample size {:d})".format(temp.shape[0]))

temp = train.loc[train['trip_duration']>3600, ['speed','pickup_date']]

plot(temp, "Trips of duration more than 1h (sample size {:d})".format(temp.shape[0]))
new_train = remove_exotic(train)

new_test = remove_exotic(test)

print("size of new data v.s. original data")

print("train: {}/{} \t {:2f}%".format(new_train.shape[0], train.shape[0], \

                                     100.0*new_train.shape[0]/train.shape[0]))

print("test:  {}/{} \t {:2f}%".format(new_test.shape[0], test.shape[0], \

                                     100.0*new_test.shape[0]/test.shape[0]))
preprocess(new_train)

preprocess(test)
columns = ['vendor_id', 'passenger_count', 'store_and_fwd_flag', \

           'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', \

           'Manhattan_dist', 'pickup_date', 'pickup_hour']

X = new_train[columns]

y = new_train['log_duration']

lm_model = lm(y, X)

lm_result = lm_model.fit()

lm_result.summary()
# predicted values

duration = np.exp(lm_result.predict(test[columns]))-1

duration = np.where(duration<0, 0, duration)
n = X.shape[0]

shuffle = np.random.permutation(n)



rf_model = rf(n_estimators=300)

rf_model.fit(X.values[shuffle[:250000]], y.values[shuffle[:250000]])

feature_importance = rf_model.feature_importances_
plt.figure()

pos = np.arange(len(columns))

plt.barh(pos, feature_importance)

plt.yticks(pos, columns)

plt.xlabel("feature importance")

plt.show()
pred = rf_model.predict(X.values[shuffle[100000:]])

print("Error is:", np.sqrt(np.mean((y.values[shuffle][100000:]-pred)**2)))