%matplotlib inline
%matplotlib inline

from __future__ import division
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, probplot
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, f_classif, RFECV
from sklearn.svm import LinearSVC, SVC
from sklearn import preprocessing

pd.set_option('display.max_columns', 1800)
pd.set_option('display.width', 1800)
train = pd.read_csv('../input/train.csv')
#train = DataSet(train_data, target_column='y', id_column='id')
print(train[:5])

store = pd.read_csv('../input/store.csv')
print(store[:5])

all_data = pd.merge(train, store, on='Store', how='left')

test = pd.read_csv('../input/test.csv')
# clean StateHoliday column
all_data['StateHoliday'][all_data['StateHoliday'] == 0 ] = '0'
# a couple of statistics
print('number of stores: {}'.format(train.Store.size))

# count open stores by week day
print(train.groupby(['DayOfWeek']).sum())
print(all_data[['DayOfWeek', 'Open', 'Sales', 'Customers']].groupby(['DayOfWeek', 'Open']).agg([np.sum, np.mean, np.std]))
avg_per_store = all_data[['Sales', 'Store']].groupby('Store').mean()
avg_per_store.reset_index().plot(kind='scatter', x='Store', y='Sales')
avg_per_weekday = all_data[['Sales', 'DayOfWeek']].groupby('DayOfWeek').mean()
avg_per_weekday.reset_index().plot(kind='bar', x='DayOfWeek', y='Sales')
all_data[['Customers', 'Sales']].plot(kind='scatter', x='Customers', y='Sales')
np.log(all_data[['Customers', 'Sales']]).plot(kind='scatter', x='Customers', y='Sales')
avg_promotion = all_data[['Sales', 'Customers', 'Promo']].groupby('Promo').mean()
avg_promotion.plot(kind='bar')
avg_stateholiday = all_data[['Sales', 'Customers', 'StateHoliday']].groupby('StateHoliday').mean()
avg_stateholiday.plot(kind='bar')
avg_stateholiday = all_data[['Sales', 'Customers', 'SchoolHoliday']].groupby('SchoolHoliday').mean()
avg_stateholiday.plot(kind='bar')
avg_stateholiday = all_data[['Sales', 'Customers', 'StoreType']].groupby('StoreType').mean()
avg_stateholiday.plot(kind='bar')
avg_stateholiday = all_data[['Sales', 'Customers', 'Assortment']].groupby('Assortment').mean()
avg_stateholiday.plot(kind='bar')
all_data[['CompetitionDistance', 'Sales']].plot(kind='scatter', x='CompetitionDistance', y='Sales')
all_data.hist('CompetitionDistance')
# Bin the competition distance with 10 bins...
bins = np.linspace(all_data['CompetitionDistance'].min(), all_data.CompetitionDistance.max(), 10)

competition_bins = all_data[['Sales', 'Customers']].groupby(np.digitize(all_data['CompetitionDistance'], bins))
competition_avg = competition_bins.mean()
competition_avg.plot(kind='bar')
competition_bins = all_data[['Sales', 'Customers']].groupby(pd.cut(all_data['CompetitionDistance'], bins))
competition_avg = competition_bins.mean()
competition_avg.plot(kind='bar')
competition_bins = all_data[['Sales', 'Customers']].groupby(np.digitize(all_data['CompetitionDistance'], bins))
competition_avg = competition_bins.count()
competition_avg.plot(kind='bar')
competition_bins = all_data[['Sales', 'Customers']].groupby(pd.cut(all_data['CompetitionDistance'], bins))
competition_avg = competition_bins.count()
competition_avg.plot(kind='bar')
# average sales and customers by month 
all_data['MMYYYY'] = all_data['Date'].map(lambda x: x[:7])


avg_hist_by_month = all_data[['Sales', 'Customers', 'MMYYYY']].groupby('MMYYYY').mean()
avg_hist_by_month.plot(kind='bar')
# average sales and customers by month
all_data['Months'] = all_data['Date'].map(lambda x: x[5:7])

avg_hist_by_month = all_data[['Sales', 'Customers', 'Months']].groupby('Months').mean()
avg_hist_by_month.plot(kind='bar')
# average sales and customers by month day
all_data['MonthDay'] = all_data['Date'].map(lambda x: x[8:])

avg_hist_by_month = all_data[['Sales', 'Customers', 'MonthDay']].groupby('MonthDay').mean()
avg_hist_by_month.plot(kind='bar')
# average sales and customers by week day
all_data['WeekDay'] = all_data['Date'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").date().weekday())
avg_hist_by_month = all_data[['Sales', 'Customers', 'WeekDay']].groupby('WeekDay').mean()
avg_hist_by_month.plot(kind='bar')
# average sales by week day by promo

avg_hist_by_month = all_data[['Sales', 'Customers', 'Promo', 'WeekDay']].groupby(['WeekDay', 'Promo']).mean()
sns.barplot(x="WeekDay", y="Sales", hue="Promo", order=[0, 1, 2, 3, 4, 5, 6], data=all_data)
# unique values in train
train.apply(lambda x: len(x.unique()))
# unique values in test
test.apply(lambda x: len(x.unique()))
# since there's 3 open unique values
# let's check which store has the extra value
test[pd.isnull(test.Open)]
print('''
        unique train stores: {} 
        unique test stores: {} 
        are alltest stores in train stores: {}
        extra train stores: {}
      '''.format(
        len(pd.unique(train.Store)),
        len(pd.unique(test.Store)),
        'yes' if set(pd.unique(test.Store)) - set(pd.unique(train.Store)) else 'no',
        len(set(pd.unique(train.Store)) - set(pd.unique(test.Store)))
    ))
# the fraction of open/closed stores train
open_close_frac = pd.value_counts(train.Open) / train.Open.size
open_close_frac.plot(kind='bar')
open_close_frac
# the fraction of open/closed stores test
open_close_frac = pd.value_counts(test.Open) / test.Open.size
open_close_frac.plot(kind='bar')
open_close_frac
# the fraction of promo/no promo stores
promo_frac = pd.value_counts(train.Promo) / train.Promo.size
promo_frac.plot(kind='bar')
promo_frac
# the fraction of promo/no promo stores
promo_frac = pd.value_counts(test.Promo) / test.Promo.size
promo_frac.plot(kind='bar')
promo_frac
print ('sales when stores are open')
print (pd.value_counts(train[train.Open == 1].Sales > 0))
print ('----')
print ('sales when stores are closes')
print (pd.value_counts(train[train.Open == 0].Sales > 0))
print ('sales when stores are on promo')
print (pd.value_counts(train[train.Promo == 1].Sales > 0))
print ('----')
print ('sales when stores have no promo')
print (pd.value_counts(train[train.Promo == 0].Sales > 0))