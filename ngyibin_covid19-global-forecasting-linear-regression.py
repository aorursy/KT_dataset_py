# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from datetime import timedelta

from sklearn.linear_model import LinearRegression



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
N = 5 # Number of previous data points to use to forecast
def get_preds_lin_reg(series, pred_min, H):

    """

    Given a dataframe, get prediction at timestep t using values from t-1, t-2, ..., t-N.

    Inputs

        series     : series to forecast

        pred_min   : all predictions should be >= pred_min

        H          : forecast horizon

    Outputs

        result: the predictions. The length of result is H. numpy array of shape (H,)

    """

    # Create linear regression object

    regr = LinearRegression(fit_intercept=True)



    pred_list = []



    X_train = np.array(range(len(series))) # e.g. [0 1 2 3 4]

    y_train = np.array(series) # e.g. [2944 3088 3226 3335 3436]

    X_train = X_train.reshape(-1, 1)     # e.g X_train = 

                                             # [[0]

                                             #  [1]

                                             #  [2]

                                             #  [3]

                                             #  [4]]

    # X_train = np.c_[np.ones(N), X_train]              # add a column

    y_train = y_train.reshape(-1, 1)

    regr.fit(X_train, y_train)            # Train the model

    pred = regr.predict(np.array(range(len(series),len(series)+H)).reshape(-1,1))

    pred = pred.reshape(H,)

    

    # If the values are < pred_min, set it to be pred_min

    pred[pred < pred_min] = pred_min

        

    return np.around(pred)
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')



# Change column names to lower case

train.columns = [col.lower() for col in train.columns]



# Change to date format

train['date'] = pd.to_datetime(train['date'], format='%Y-%m-%d')



train
test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')



# Change column names to lower case

test.columns = [col.lower() for col in test.columns]



# Change to date format

test['date'] = pd.to_datetime(test['date'], format='%Y-%m-%d')



test
submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/submission.csv')

submission
# Count number of nulls for each column

train.isnull().sum(axis=0)
# Get the province_states

print(len(train['province_state'].unique()))

train['province_state'].unique()
# Get the country_regions

print(len(train['country_region'].unique()))

train['country_region'].unique()
# Get amount of data per country

train['country_region'].value_counts()
train[train['country_region']=='Singapore']
# Plot the confirmed cases in Singapore, Malaysia, Indonesia, Thailand

countries_list = ['Singapore', 'Malaysia', 'Indonesia', 'Thailand', 'Philippines', 'Brunei', 'Laos', 'Cambodia', 'New Zealand']

color_list = ['r', 'g', 'b', 'k', 'c', 'y', 'm', '0.75', '0.25']



ax = train[train['country_region']==countries_list[0]].plot(x='date', y='confirmedcases', style = 'r.-', grid=True, figsize=(10, 6))



i = 1

for country in countries_list[1:]:

    ax = train[train['country_region']==country].plot(x='date', y='confirmedcases', color=color_list[i], marker='.', grid=True, ax=ax, figsize=(10, 6))

    i = i + 1

    

ax.set_xlabel("date")

ax.set_ylabel("confirmedcases")

ax.legend(countries_list)
# Plot the fatalities in Singapore, Malaysia, Indonesia, Thailand

ax = train[train['country_region']==countries_list[0]].plot(x='date', y='fatalities', style = 'r.-', grid=True, figsize=(10, 6))



i = 1

for country in countries_list[1:]:

    ax = train[train['country_region']==country].plot(x='date', y='fatalities', color=color_list[i], marker='.', grid=True, ax=ax, figsize=(10, 6))

    i = i + 1

    

ax.set_xlabel("date")

ax.set_ylabel("fatalities")

ax.legend(countries_list)
# Plot the confirmed cases in China, US, India

ax = train[train['country_region']=='China'].groupby("date").agg({"confirmedcases": "sum"}).plot(marker='.', figsize=(10, 6), grid=True)

ax = train[train['country_region']=='US'].groupby("date").agg({"confirmedcases": "sum"}).plot(marker='.', figsize=(10, 6), grid=True, ax=ax)

ax = train[train['country_region']=='India'].groupby("date").agg({"confirmedcases": "sum"}).plot(marker='.', figsize=(10, 6), grid=True, ax=ax)

ax = train[train['country_region']=='Italy'].groupby("date").agg({"confirmedcases": "sum"}).plot(marker='.', figsize=(10, 6), grid=True, ax=ax)

ax = train[train['country_region']=='France'].groupby("date").agg({"confirmedcases": "sum"}).plot(marker='.', figsize=(10, 6), grid=True, ax=ax)

ax = train[train['country_region']=='Iran'].groupby("date").agg({"confirmedcases": "sum"}).plot(marker='.', figsize=(10, 6), grid=True, ax=ax)



ax.set_xlabel("date")

ax.set_ylabel("confirmedcases")

ax.legend(['China', 'US', 'India', 'Italy', 'France', 'Iran'])
# Plot the fatalities in China, US, India

ax = train[train['country_region']=='China'].groupby("date").agg({"fatalities": "sum"}).plot(marker='.', figsize=(10, 6), grid=True)

ax = train[train['country_region']=='US'].groupby("date").agg({"fatalities": "sum"}).plot(marker='.', figsize=(10, 6), grid=True, ax=ax)

ax = train[train['country_region']=='India'].groupby("date").agg({"fatalities": "sum"}).plot(marker='.', figsize=(10, 6), grid=True, ax=ax)

ax = train[train['country_region']=='Italy'].groupby("date").agg({"fatalities": "sum"}).plot(marker='.', figsize=(10, 6), grid=True, ax=ax)

ax = train[train['country_region']=='France'].groupby("date").agg({"fatalities": "sum"}).plot(marker='.', figsize=(10, 6), grid=True, ax=ax)

ax = train[train['country_region']=='Iran'].groupby("date").agg({"fatalities": "sum"}).plot(marker='.', figsize=(10, 6), grid=True, ax=ax)



ax.set_xlabel("date")

ax.set_ylabel("fatalities")

ax.legend(['China', 'US', 'India', 'Italy', 'France', 'Iran'])
# Get global number of cases

ax = train.groupby("date").agg({"confirmedcases": "sum"}).plot(marker='.', figsize=(10, 6), grid=True)

train.groupby("date").agg({"fatalities": "sum"}).plot(marker='.', figsize=(10, 6), grid=True, ax=ax)
# Fill nans in province_state with ''

train['province_state'] = train['province_state'].fillna(value = 'nil')

train.head()
# Fill nans in province_state with ''

test['province_state'] = test['province_state'].fillna(value = 'nil')

test.head()
# Get unique combinations of province_state and country_region

ps_cr_unique = train[['province_state', 'country_region']].drop_duplicates()

ps_cr_unique
# Get number of days we need to predict

date_max_train = train[(train['province_state']=='nil') & 

                 (train['country_region']=='Singapore')]['date'].max()



date_max_test = test[(test['province_state']=='nil') & 

                 (test['country_region']=='Singapore')]['date'].max()



pred_days = (date_max_test - date_max_train).days

print(date_max_train, date_max_test, pred_days)
# Specify the country here

ps = 'nil'

cr = 'Singapore'
train_sgp = train[(train['province_state']==ps) & (train['country_region']==cr)]

train_sgp[-5:]
# Get predictions 

preds = get_preds_lin_reg(train_sgp['confirmedcases'][-N:], 0, pred_days)

preds
# Put into dataframe

date_list = []

date = pd.date_range(date_max_train+timedelta(days=1), date_max_test)

results = pd.DataFrame({'date': date, 'preds':preds})

results.head()
# Plot the confirmed cases in Singapore and the predictions

ax = train[train['country_region']==cr].plot(x='date', y='confirmedcases', style = 'r.-', grid=True, figsize=(10, 6))

ax = results.plot(x='date', y='preds', style = 'r.', grid=True, figsize=(10, 6), ax=ax)

    



ax.set_xlabel("date")

ax.set_ylabel("fatalities")

ax.legend([cr])
# Predict for confirmedcases

ps_list = []

cr_list = []

date_list = []

confirmedcasespred_list = []



for index, row in ps_cr_unique.iterrows():

    train_temp = train[(train['province_state']==row['province_state']) & (train['country_region']==row['country_region'])]

    preds = get_preds_lin_reg(train_temp['confirmedcases'][-N:], 0, pred_days)

    

    ps_list = ps_list + ([row['province_state']]*pred_days)

    cr_list = cr_list + ([row['country_region']]*pred_days)

    date_list = date_list + list(pd.date_range(date_max_train+timedelta(days=1), date_max_test).strftime("%Y-%m-%d"))

    confirmedcasespred_list = confirmedcasespred_list + list(preds)



results = pd.DataFrame({'province_state': ps_list,

                        'country_region': cr_list,

                        'date': date_list,

                        'confirmedcases': confirmedcasespred_list})

results['date'] = pd.to_datetime(results['date'], format='%Y-%m-%d')

results
# Merge test with the existing values in train

test_merged = test.merge(train[['province_state', 'country_region', 'date', 'confirmedcases', 'fatalities']], 

                         left_on=['province_state', 'country_region', 'date'], 

                         right_on=['province_state', 'country_region', 'date'], 

                         how='left') 

test_merged
# Merge test with the predictions

test_merged2 = test_merged.merge(results, 

                                left_on=['province_state', 'country_region', 'date'], 

                                right_on=['province_state', 'country_region', 'date'], 

                                how='left') 

test_merged2
# Create column confirmedcases

test_merged2['confirmedcases'] = test_merged2.apply(lambda row: row['confirmedcases_x'] if pd.isnull(row['confirmedcases_y']) else row['confirmedcases_y'], axis=1)

test_merged2.drop(['confirmedcases_x', 'confirmedcases_y'], axis=1, inplace=True)

test_merged2
# Predict for fatalities

ps_list = []

cr_list = []

date_list = []

fatalities_list = []



for index, row in ps_cr_unique.iterrows():

    train_temp = train[(train['province_state']==row['province_state']) & (train['country_region']==row['country_region'])]

    preds = get_preds_lin_reg(train_temp['fatalities'][-N:], 0, pred_days)

    

    ps_list = ps_list + ([row['province_state']]*pred_days)

    cr_list = cr_list + ([row['country_region']]*pred_days)

    date_list = date_list + list(pd.date_range(date_max_train+timedelta(days=1), date_max_test).strftime("%Y-%m-%d"))

    fatalities_list = fatalities_list + list(preds)



results = pd.DataFrame({'province_state': ps_list,

                        'country_region': cr_list,

                        'date': date_list,

                        'fatalities': fatalities_list})

results['date'] = pd.to_datetime(results['date'], format='%Y-%m-%d')

results
# Merge with the predictions

test_merged3 = test_merged2.merge(results, 

                                left_on=['province_state', 'country_region', 'date'], 

                                right_on=['province_state', 'country_region', 'date'], 

                                how='left') 

test_merged3
# Create column fatalities

test_merged3['fatalities'] = test_merged3.apply(lambda row: row['fatalities_x'] if pd.isnull(row['fatalities_y']) else row['fatalities_y'], axis=1)

test_merged3.drop(['fatalities_x', 'fatalities_y'], axis=1, inplace=True)

test_merged3
# Form the submission dataset

submission = test_merged3.copy()

submission.drop(['country_region', 'province_state', 'date'], axis=1, inplace=True)

submission.rename(columns={'forecastid': 'ForecastId',

                           'fatalities': 'Fatalities', 

                           'confirmedcases': 'ConfirmedCases'}, inplace=True)

submission
# Test submission

submission.to_csv("submission.csv", index=False)