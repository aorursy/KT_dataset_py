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

# settings

import warnings

warnings.filterwarnings("ignore")
#importing the required library files

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from fbprophet import Prophet

from sklearn.metrics import mean_squared_error, mean_absolute_error
#loading the data 

chk = pd.read_csv('../input/mobile-usage-dataset-individual-person/CheckDevice.csv')

ph_usage = pd.read_csv('../input/mobile-usage-dataset-individual-person/phone_usage.csv')

chk.head()
ph_usage.head()
print("info of CHECK COUNT")

print('--'*20)

chk.info()
#renaming the columns name

chk.rename(columns={'Check phone count': 'check_phn_count', 'Screen on time': 'screen_on_time'}, inplace=True)
#droping the NaN columns

chk.dropna(axis=0, inplace =True)
chk['duration'] = chk['screen_on_time'].str.split(':').apply(lambda x: int(x[0]) *60 + int(x[1])  )
chk.describe()
chk.loc[chk['duration'] > 660, 'duration'] = chk['duration'].median()
chk.describe()
chk["Date"]= pd.to_datetime(chk["Date"]) 

#Bar plot with respect ot date and the phone check count

plt.figure(figsize=(20,6))

sns.barplot(x="Date", y="check_phn_count", data=chk)

plt.title('Phone check count')

plt.xticks(rotation=90)

plt.show()
#Bar plot with respect to date and the phone usage duration everyday

plt.figure(figsize=(20,6))

sns.barplot(x="Date", y="duration", data=chk)

plt.title('Phone usage each day in minutes')

plt.xticks(rotation=90)

plt.show()
## converting the date column from object to time series

chk['Date'] = pd.to_datetime(chk['Date'])
chk['day_of_week'] = chk['Date'].dt.dayofweek

chk
chk.groupby('day_of_week').sum().nlargest(20,'duration').reset_index()
plt.figure(figsize=(15,6))

data = chk.groupby('day_of_week').sum().nlargest(20,'duration').reset_index()

sns.barplot(x='day_of_week',y='duration',data=data)

plt.title('DAY OF THE WEEK')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(15,6))

data = chk.groupby('day_of_week').sum().nlargest(20,'check_phn_count').reset_index()

sns.barplot(x='day_of_week',y='check_phn_count',data=data)

plt.title('DAY OF THE WEEK')

plt.xticks(rotation=90)

plt.show()
chk['categories'] = chk['day_of_week'].apply(lambda x: 'weekday' if x < 5 else 'weekend')



chk['weekday'] = chk['categories'].apply(lambda x: '0' if x == 'weekday' else '1')

chk.drop(columns='categories', inplace=True)

chk
chk['month'] = pd.DatetimeIndex(chk['Date']).month

chk.head()
ph_usage.head(10)
#renaming the columns name

ph_usage.rename(columns={'App name': 'App_name'}, inplace=True)
ph_usage.info()
ph_usage.describe()
ph_usage.shape
#droping the NaN columns

ph_usage.dropna(axis=0, inplace =True)
ph_usage.columns
#making a new copy of data frame

ph_usg = ph_usage
# Creating a new column of datetime (timestamp)

ph_usg['DateTime']= pd.to_datetime(ph_usg['Date'] +" " + ph_usg['Time'],format='%d/%m/%Y %H:%M:%S')

ph_usg.head()
# Converting the duration into seconds.

ph_usg['usage_seconds'] = ph_usg['Duration'].str.split(':').apply(lambda x: int(x[0]) *3600 + int(x[1]) * 60 + int(x[2]))

ph_usg
# to find the number of days

ph_usg['DateTime'].max() - ph_usg['DateTime'].min()
#Filtering the system apps and system usage

system_tracker = ['Screen on (unlocked)','Screen off (locked)','Screen on (locked)', 'Screen off','Permission controller','System UI','Package installer',

'Device shutdown','Call Management']

service_app = ph_usg[ph_usg['App_name'].isin(system_tracker)]

service_app
#Getting all the user apps.

all_apps = ph_usg[~ph_usg['App_name'].isin(system_tracker)]



all_apps
#sorting the usage seconds in descending order

test = service_app.sort_values(by='usage_seconds',ascending=0)

sns.scatterplot(x='App_name', y='usage_seconds', data=test[test['usage_seconds'] > 3600])
plt.figure(figsize=(15,6))

sns.countplot(test['App_name'])

plt.title('APP name count')

plt.xticks(rotation=90)

plt.show()
sleep = ['Screen off (locked)','Screen on (locked)', 'Screen off']

sleep_duration = service_app[service_app['App_name'].isin(sleep)]

sleep_duration
sns.scatterplot(x='App_name', y='usage_seconds', data=sleep_duration[sleep_duration['usage_seconds'] > 18000])
new = sleep_duration[sleep_duration['usage_seconds'] > 18000]

new


plt.figure(figsize=(15,6))

sns.scatterplot(x='Date', y='usage_seconds', data=new)

plt.title('User sleep pattern')

plt.xticks(rotation=90)

plt.show()
#The user approximately sleeps 6.7 hours everyday.. since the screen off was filtered more than 5 hours. all the time are showing around 10PM to 1AM

new.usage_seconds.mean()
# Getting the screen on unlocked alone

wake = ['Screen on (unlocked)']

wake_up = service_app[service_app['App_name'].isin(wake)]

wake_up.head()
wake_up.tail()
#Grouping the datetime on the basis of frequency day and getting the minimum time of the day

wakeup_time= wake_up.set_index('DateTime').groupby(pd.Grouper(freq='D')).min()

wakeup_time.tail(50)

#Filtering the app usage seconds more than 10 seconds. Assuming that user use apps more than 10 seconds. 

all_apps = all_apps[(all_apps.usage_seconds > 10)]

all_apps
#All apps access count

all_apps['App_name'].value_counts()
plt.figure(figsize=(15,6))

sns.countplot(x = 'App_name',

              data = all_apps,

              order = all_apps['App_name'].value_counts().index)



plt.title('APP name count')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(15,6))

s = all_apps['App_name'].value_counts().head(25)

ax= s.plot.bar(width=.8) 



for i, v in s.reset_index().iterrows():

    ax.text(i, v.App_name + 0.2 , v.App_name, color='red')
all_apps['usage_minutes'] = all_apps['usage_seconds']//60

plt.figure(figsize=(15,6))

data = all_apps.groupby('App_name').sum().nlargest(20,'usage_minutes').reset_index()

sns.barplot(x='App_name',y='usage_minutes',data=data)

plt.title('Top 20 apps used')

plt.xticks(rotation=90)

plt.show()
all_apps.groupby('App_name').sum().nlargest(20,'usage_minutes').reset_index()
def dateFeatures(all_apps):

    features = ['day','week','dayofweek','month','weekofyear']

    for col in features:

        all_apps[col] = getattr(all_apps['DateTime'].dt,col) * 1
dateFeatures(all_apps)

all_apps
plt.figure(figsize=(15,6))

all_apps.groupby(['weekofyear'])['usage_minutes'].sum().plot(kind='bar')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(15,6))

all_apps.groupby(['month','App_name']).sum().nlargest(6,'usage_minutes')['usage_minutes'].plot(kind='bar')

plt.title('User spends more minutes on the app for each month')

plt.xticks(rotation=90)

plt.show()
all_apps
train = all_apps.copy()
def f(x):

    if (x > 5) and (x <= 8):

        return 'Early_Morn'

    elif (x > 8) and (x <= 12 ):

        return 'Morn'

    elif (x > 12) and (x <= 16):

        return'Noon'

    elif (x > 16) and (x <= 20) :

        return 'Eve'

    elif (x > 20) and (x <= 24):

        return'Night'

    elif (x <= 4):

        return'Late_Night'
# make a session

train['hour'] = train['DateTime'].dt.hour

train['session'] = train['hour'].apply(f)

train.drop(['weekofyear','usage_seconds'],axis=1, inplace=True)
train
# Grouping based on the session, date and app name, to find out which counts of app on each session each day

train.groupby(['session','Date','App_name']).size().reset_index()

sns.pairplot(train,

             hue='hour',

             x_vars=['hour','dayofweek','week','session'],

             y_vars='usage_minutes',

             height=5,

             plot_kws={'alpha':0.15, 'linewidth':0}

            )

plt.suptitle('Phone usage minutes, hour, Day of Week, week and session')

plt.show()
train.set_index('DateTime',inplace=True)
train
split_date = '30/10/2019'

f_train = train.loc[train.index <= split_date].copy()

f_test  = train.loc[train.index > split_date].copy()
f_train
plt.style.use('fivethirtyeight') # For plots

# Color pallete for plotting

color_pal = ["#F8766D", "#D39200", "#93AA00",

             "#00BA38", "#00C19F", "#00B9E3",

             "#619CFF", "#DB72FB"]

train.plot(style='.', figsize=(20,6), color=color_pal, title='Usage plot')

plt.show()
# Format data for prophet model using ds and y

f_train.reset_index().rename(columns={'DateTime':'ds','usage_minutes':'y'}).head()
# Setup and train model and fit

model = Prophet()

model.fit(f_train.reset_index().rename(columns={'DateTime':'ds','usage_minutes':'y'}))
# Predict on training set with model

f_test_fcst = model.predict(df=f_test.reset_index().rename(columns={'DateTime':'ds'}))
f_test_fcst.head()
# Plot the forecast

f, ax = plt.subplots(1)

f.set_figheight(5)

f.set_figwidth(15)

fig = model.plot(f_test_fcst,ax=ax)

plt.show()
# Plot the components of the model

fig = model.plot_components(f_test_fcst)
# Plot the forecast with the actuals

f, ax = plt.subplots(1)

f.set_figheight(5)

f.set_figwidth(15)

ax.scatter(f_test.index, f_test['usage_minutes'], color='r')

fig = model.plot(f_test_fcst, ax=ax)
mean_squared_error(y_true=f_test['usage_minutes'],

                   y_pred=f_test_fcst['yhat'])
mean_absolute_error(y_true=f_test['usage_minutes'],

                   y_pred=f_test_fcst['yhat'])