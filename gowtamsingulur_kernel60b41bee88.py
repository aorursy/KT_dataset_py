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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns





df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')

df_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv')

print(df_train.shape)

print(df_test.shape)
df_train.columns
df_train['Province/State'] = df_train.apply(

    lambda row: row['Country/Region'] if pd.isnull(row['Province/State']) else row['Province/State'],

    axis=1

)

df_test['Province/State'] = df_test.apply(

    lambda row: row['Country/Region'] if pd.isnull(row['Province/State']) else row['Province/State'],

    axis=1

)
df_train['Date'] = df_train.apply(

    lambda row: pd.Timestamp(row['Date']).value//10**9,

    axis=1

)

df_test['Date'] = df_test.apply(

    lambda row: pd.Timestamp(row['Date']).value//10**9,

    axis=1

)
import matplotlib.pyplot as plt

df = df_train.sort_values('Date')

plt.plot(df['Date'],np.log2(df['ConfirmedCases'])/np.log2(1.5))

plt.show()

plt.plot(df['Date'],np.log2(df['Fatalities'])/np.log2(1.5))

plt.show()
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.preprocessing import Normalizer

vectorizer = CountVectorizer(binary=True)

vectorizer.fit(df_train['Province/State'])

state_train = vectorizer.transform(df_train['Province/State'])

state_test = vectorizer.transform(df_test['Province/State'])



vectorizer = CountVectorizer(binary=True)

vectorizer.fit(df_train['Country/Region'])

country_train = vectorizer.transform(df_train['Country/Region'])

country_test = vectorizer.transform(df_test['Country/Region'])
normalizer = Normalizer()

normalizer.fit(df_train['Lat'].values.reshape(1, -1))

lat_train = normalizer.transform(df_train['Lat'].values.reshape(1, -1))

lat_test = normalizer.transform(df_test['Lat'].values.reshape(1, -1))



normalizer = Normalizer()

normalizer.fit(df_train['Long'].values.reshape(1, -1))

long_train = normalizer.transform(df_train['Long'].values.reshape(1, -1))

long_test = normalizer.transform(df_test['Long'].values.reshape(1, -1))



normalizer = Normalizer()

normalizer.fit(df_train['Date'].values.reshape(1, -1))

date_train = normalizer.transform(df_train['Date'].values.reshape(1, -1))

date_test = normalizer.transform(df_test['Date'].values.reshape(1, -1))
print(state_train.shape)

print(country_train.shape)

print(lat_train.reshape(-1, 1).shape)

print(long_train.shape)

print(date_train.shape)
from scipy.sparse import hstack

data_train = hstack([state_train, country_train, lat_train.reshape(-1, 1), long_train.reshape(-1, 1), date_train.reshape(-1, 1)])

data_test = hstack([state_test, country_test, lat_test.reshape(-1, 1), long_test.reshape(-1, 1), date_test.reshape(-1, 1)])

#data_train=data_train.todense()

#data_test=data_test.todense()
#print(data_test[0])
from sklearn import linear_model

from sklearn.svm import SVR

from sklearn.ensemble import GradientBoostingRegressor

from xgboost import XGBRegressor

clf = XGBRegressor(base_score=0.5, booster=None, colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,

             importance_type='gain', interaction_constraints=None,

             learning_rate=0.300000012, max_delta_step=0, max_depth=6,

             min_child_weight=1, monotone_constraints=None,

             n_estimators=1000, n_jobs=0, num_parallel_tree=1,

             objective='reg:squarederror', random_state=0, reg_alpha=0,

             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method=None,

             validate_parameters=False, verbosity=None)

clf.fit(data_train.todense(), df_train['ConfirmedCases'])

#conf_cased_pred_tr = clf.predict(data_test.todense())

conf_cased_pred = clf.predict(data_test.todense())

normalizer = Normalizer()

normalizer.fit(df_train['ConfirmedCases'].values.reshape(1, -1))

conf_cased_pred_train = normalizer.transform(df_train['ConfirmedCases'].values.reshape(1, -1))

conf_cased_pred_test = normalizer.transform(conf_cased_pred.reshape(1, -1))

data_train_with_conf = hstack([state_train, country_train, lat_train.reshape(-1, 1), long_train.reshape(-1, 1), date_train.reshape(-1, 1), conf_cased_pred_train.reshape(-1,1)])

data_test_with_conf = hstack([state_test, country_test, lat_test.reshape(-1, 1), long_test.reshape(-1, 1), date_test.reshape(-1, 1),conf_cased_pred_test.reshape(-1, 1)])
print(np.mean(conf_cased_pred))
clf = XGBRegressor(base_score=0.5, booster=None, colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,

             importance_type='gain', interaction_constraints=None,

             learning_rate=0.300000012, max_delta_step=0, max_depth=6,

             min_child_weight=1, monotone_constraints=None,

             n_estimators=1000, n_jobs=0, num_parallel_tree=1,

             objective='reg:squarederror', random_state=0, reg_alpha=0,

             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method=None,

             validate_parameters=False, verbosity=None)

clf.fit(data_train_with_conf.todense(), np.log(df_train['Fatalities']+0.000000001))

fatalities_pred = clf.predict(data_test_with_conf.todense())

#fatalities_pred = np.exp(fatalities_pred)

print(np.mean(fatalities_pred))
def make_submission(conf, fat, sub_name):

    my_submission = pd.DataFrame({'ForecastId':pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv').ForecastId,'ConfirmedCases':conf, 'Fatalities':fat})

    my_submission.to_csv('{}.csv'.format(sub_name),index=False)

    print('A submission file has been made')

make_submission(conf_cased_pred,fatalities_pred,'submission')