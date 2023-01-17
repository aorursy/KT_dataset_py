# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import gc

import operator

import numpy as np

import pandas as pd

from datetime import datetime

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split, TimeSeriesSplit

import matplotlib.pyplot as plt

%matplotlib inline
def to_weight(y):

    w = np.zeros(y.shape, dtype=float)

    ind = y != 0

    w[ind] = 1./(y[ind]**2)

    return w



def rmspe(y, yhat):

    w = to_weight(y)

    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))

    return rmspe



def str_date_to_ordinal(date):

    return datetime.strptime(date, '%Y-%m-%d').date().toordinal()



def date_to_ordinal(date):

    return date.toordinal()



def str_to_date(date):

    return datetime.strptime(date, '%Y-%m-%d').date()



SEED = 42
train_df = pd.read_csv('../input/train.csv', sep=',', parse_dates=['Date'], date_parser=str_to_date,

                      low_memory=False)

test_df = pd.read_csv('../input/test.csv', sep=',', parse_dates=['Date'], date_parser=str_to_date,

                      low_memory=False)
train_df.info()
test_df.info()
test_df[pd.isnull(test_df.Open)]
test_df.loc[pd.isnull(test_df.Open), 'Open'] = 1
all_df = pd.concat([train_df, test_df], axis=0)

all_df.head()
all_df.info()
unique_StateHoliday_values = np.sort(all_df.StateHoliday.unique())

print ("Train data StateHoliday unique values:", np.sort(train_df.StateHoliday.unique()))

print ("Test data StateHoliday unique values: ", np.sort(test_df.StateHoliday.unique()))

print ("All data StateHoliday unique values:  ", unique_StateHoliday_values)
all_df['StateHoliday'] = all_df['StateHoliday'].astype('category').cat.codes
feature_name = 'Date'

dispatcher = {'DayOfMonth': pd.Index(all_df[feature_name]).day, 

              'WeekOfYear': pd.Index(all_df[feature_name]).week,

              'MonthOfYear': pd.Index(all_df[feature_name]).month,

              'Year': pd.Index(all_df[feature_name]).year,

              'DayOfYear': pd.Index(all_df[feature_name]).dayofyear

             }



for new_feat_suffx, mapping in dispatcher.items():

    all_df[feature_name + new_feat_suffx] = mapping



all_df[feature_name] = all_df[feature_name].apply(date_to_ordinal)



all_df.head()
store_df = pd.read_csv('../input/store.csv', sep=',')

store_df.head()
store_df.info()
unique_StoreType_values =np.sort(store_df['StoreType'].unique())

unique_Assortment_values = np.sort(store_df['Assortment'].unique())

print ("Unique values of StoreType: ", unique_StoreType_values)

print ("Unique values of Assortment:", unique_Assortment_values)
store_df['StoreType'] = store_df['StoreType'].astype('category').cat.codes

store_df['Assortment'] = store_df['Assortment'].astype('category').cat.codes

print ("Unique values of StoreType: ", np.sort(store_df['StoreType'].unique()))

print ("Unique values of Assortment:", np.sort(store_df['Assortment'].unique()))
def CompetitionOpenSince_to_ordinal(row):

    try:

        date = '%d-%d' % (int(row['CompetitionOpenSinceYear']), int(row['CompetitionOpenSinceMonth']))

        return datetime.strptime(date, '%Y-%m').date().toordinal()

    except:

        return np.nan
store_df['CompetitionOpenSinceDate'] = store_df.apply(CompetitionOpenSince_to_ordinal, axis=1)

mean_CompetitionOpenSince = store_df['CompetitionOpenSinceDate'].mean()

store_df['CompetitionOpenSinceDate'] = store_df['CompetitionOpenSinceDate'].fillna(

                                        mean_CompetitionOpenSince).astype(np.int64)

store_df.head()
def Promo2Since_to_ordinal(row):

    try:

        date = '%d-W%d' % (int(row['Promo2SinceYear']), int(row['Promo2SinceWeek']))

        return datetime.strptime(date + '-1', '%Y-W%W-%w').date().toordinal()

    except:

        return np.nan
store_df['Promo2SinceDate'] = store_df.apply(Promo2Since_to_ordinal, axis=1)

mean_Promo2SinceDate = store_df['Promo2SinceDate'].mean()

store_df['Promo2SinceDate'] = store_df['Promo2SinceDate'].fillna(mean_Promo2SinceDate).astype(np.int64)

store_df.loc[store_df['Promo2'] == 0, 'Promo2SinceDate'] = 0

store_df.head()
mode_PromoInterval = store_df[store_df['PromoInterval'].notnull()]['PromoInterval'].mode()[0]

promo_intervals = {'Jan,Apr,Jul,Oct': (1,4,7,10), 

                   'Feb,May,Aug,Nov': (2,5,8,11), 

                   'Mar,Jun,Sept,Dec': (3,6,9,12)

                  } # ()}

promo_intervals[np.nan] = promo_intervals[mode_PromoInterval]

store_df['PromoInterval'] = store_df['PromoInterval'].apply(lambda x: promo_intervals[x])

store_df.head()
columns_to_drop = ['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear',

                   'Promo2SinceWeek', 'Promo2SinceYear']

store_df.drop(columns_to_drop, axis=1, inplace=True)

store_df.head()
store_df.info()
mean_CompetitionDistance = store_df.CompetitionDistance.mean()

store_df['CompetitionDistance'] = store_df['CompetitionDistance'].fillna(mean_CompetitionDistance)

store_df.info()
df = pd.merge(all_df, store_df, how='left', on=['Store'])

df.head().T
df.info()
def is_promo2_today(row):

    return int(row['Promo2'] == 1 and row['Promo2SinceDate'] <= row['Date'] \

           and row['DateMonthOfYear'] in row['PromoInterval'])



def is_competition_open(row):

    return int(row['CompetitionOpenSinceDate'] <= row['Date'])
df['Promo2Today'] = df.apply(is_promo2_today, axis=1)

df['CompetitionIsOpen'] = df.apply(is_competition_open, axis=1)

avg_checks = pd.DataFrame(df.groupby('Store')['Sales'].sum().astype(np.float64) \

             / df.groupby('Store')['Customers'].sum().astype(np.float64), 

                          columns=['AverageCheck']).reset_index()

df = df.merge(avg_checks, on='Store', how='left')

df.head().T
columns_to_drop = ['PromoInterval', 'Date', 'CompetitionOpenSinceDate', 'Promo2SinceDate']

df.drop(columns_to_drop, axis=1, inplace=True)
train_df = df[(df['Id'].isnull()) & (df['Open'] == 1)].drop(['Id'], axis=1)

train_df.info()
test_df = df[df['Id'].notnull()].drop(['Sales', 'Customers'], axis=1)

test_df.info()
X_train = train_df[train_df.columns.drop(['Customers', 'Sales'])].values

y_train = train_df['Sales'].values

print (X_train.shape, y_train.shape)
X_test = test_df[test_df.columns.drop(['Id'])].values

X_test.shape
#X_train_train, X_train_test, y_train_train, y_train_test = train_test_split(X_train, y_train,

                                                                           #train_size=0.08,

                                                                           #test_size=0.02,

                                                                           #random_state=SEED)

#print X_train_train.shape, X_train_test.shape, y_train_train.shape, y_train_test.shape
columns = train_df.columns.drop(['Sales', 'Customers'])

del train_df

gc.collect()
'''

param_grid = {'n_estimators': (10, 50, 80, 100),

              'criterion': ('mse',),

              'max_depth': (5, 10, 15, 20, None)}

best_n_estimators = None

best_criterion = None

best_max_depth = None

best_rsmpe_score = 100.0

for criterion in param_grid['criterion']:

    for n_estimators in param_grid['n_estimators']:

        for max_depth in param_grid['max_depth']:

            print ('n_estimators =', n_estimators, 'criterion =', criterion, 'max_depth =', max_depth)

            scores = []

            tss = TimeSeriesSplit(n_splits=5)

            for train, test in tss.split(X_train_train):

                rf_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,

                                                 criterion=criterion, random_state=SEED, n_jobs=1)

                rf_model.fit(X_train_train[train], y_train_train[train])

                scores.append(rmspe(y_train_train[test], rf_model.predict(X_train_train[test])))

                del rf_model

                gc.collect()

            cur_rsmpe_score = np.mean(np.array(scores))

            print ('rsmpe_score =', cur_rsmpe_score)

            if cur_rsmpe_score < best_rsmpe_score:

                best_rsmpe_score = cur_rsmpe_score

                best_n_estimators = n_estimators

                best_criterion = criterion

                best_max_depth = max_depth

            del scores, tss

            gc.collect()

            print '----------------------------------------------------------'

        

print ('best n_estimators:', best_n_estimators, 'best criterion:', best_criterion)

print ('rmspe for best params:', best_rsmpe_score)

'''
rf = RandomForestRegressor(n_estimators=50, random_state=SEED)

rf.fit(X_train, y_train)
# check RMSPE on test data to verify regression is not too bad

#print rmspe(y_train_test, rf.predict(X_train_test))
test_predicted = rf.predict(X_test)

test_predicted.shape
submission = pd.concat([test_df.Id.astype(int), pd.DataFrame(test_predicted, 

                                                 columns=['Sales'], index=test_df.index)], axis=1)

submission.head()
submission[['Id', 'Sales']].to_csv('submission_6.csv', index=False)
features = {}

for i, column in enumerate(columns):

    features[column] = rf.feature_importances_[i] 

features = sorted(features.items(), key=operator.itemgetter(1), reverse=True)

features