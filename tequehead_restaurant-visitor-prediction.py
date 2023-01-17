# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# import ggplot

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



data = {

    'tra': pd.read_csv('../input/air_visit_data.csv'),

    'as': pd.read_csv('../input/air_store_info.csv'),

    'hs': pd.read_csv('../input/hpg_store_info.csv'),

    'ar': pd.read_csv('../input/air_reserve.csv'),

    'hr': pd.read_csv('../input/hpg_reserve.csv'),

    'id': pd.read_csv('../input/store_id_relation.csv'),

    'tes': pd.read_csv('../input/sample_submission.csv'),

    'hol': pd.read_csv('../input/date_info.csv').rename(columns={'calendar_date': 'visit_date'})

}



# Any results you write to the current directory are saved as output.



# print(data)
air_visit_count = data['tra'].groupby(data['tra']['visit_date'])['visitors'].sum()

plt.figure(figsize = (15, 7))

plt.plot(air_visit_count.index, air_visit_count)

plt.ylabel('Number of visitors', fontsize = 20)

plt.legend()



air_visit_data = data['tra']

air_visit_data['visit_date'] = pd.to_datetime(air_visit_data['visit_date'])

air_visit_data['day_of_week'] = air_visit_data['visit_date'].dt.dayofweek

median_visitors_per_day = air_visit_data.groupby(['day_of_week'])['visitors'].median()



air_visit_data['month'] = air_visit_data['visit_date'].dt.month

median_visitors_per_month = air_visit_data.groupby(['month'])['visitors'].median()



fig, (ax1, ax2) = plt.subplots(ncols = 2, sharey = True, figsize = (14, 4))

sns.barplot(x = median_visitors_per_day.index, y = median_visitors_per_day, ax = ax1)

sns.barplot(x = median_visitors_per_month.index, y = median_visitors_per_month, ax = ax2)





# air_visit_data['visit_date'] = pd.to_datetime(air_visit_data['visit_date'])

# print(data['tra'])

air_total_visits = data['tra'].groupby('visit_date').sum()

air_mean_visits = data['tra'].groupby('visit_date').mean()

air_median_visits = data['tra'].groupby('visit_date').median()

# air_total_visits.describe()

# air_total_visits.groups['2016-01-01']

# air_total_visits.groups.keys()

# for key in air_total_visits.groups:

# tmp = air_total_visits.groups['2016-01-01']

plt.figure('Total visits per day')

plt.plot(air_total_visits)

plt.figure('Mean visits per day')

plt.plot(air_mean_visits)

plt.figure('Median visits per day')

plt.plot(air_median_visits)

# Setting up a Gradient Boosting Regressor model

params = {'n_estimators': 100, # Number of boosting estimators ?

         'max_depth': 5, # Maximum depth of individual regression estimators

         'min_samples_split': 200, # Minimum # of samples to split an internal node

         'min_samples_leaf': 50, # Minimum # of samples to be a leaf node

         'learning_rate': 0.005, # Shrinks the contribution of each tree by learning_rate

         'max_features': 9, # Number of features to consider when looking for a best split

         'subsample': .8, # Fraction of samples to be used for fitting the individual base learners

         'loss': 'ls'} # Least squares














