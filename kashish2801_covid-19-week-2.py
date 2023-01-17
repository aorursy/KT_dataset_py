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
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import xgboost as xgb

from xgboost import plot_importance, plot_tree

from sklearn.metrics import mean_squared_error, mean_absolute_error
train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv")

test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/test.csv")

submission = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/submission.csv")
country_dict= dict()

for itr in range(len(train)):

    if train.loc[itr]['Country_Region'] not in country_dict.keys():

        country_dict[train.loc[itr]['Country_Region']]= []

    else:

        if len(country_dict[train.loc[itr]['Country_Region']])>=69:

            continue

    country_dict[train.loc[itr]['Country_Region']].append([[train.loc[itr]['Date']],[train.loc[itr]['ConfirmedCases']],[train.loc[itr]['Fatalities']]])    
def split_sequence(sequence, n_steps):

    X, Y = list(), list()

    for i in range(len(sequence)):

        end_ix = i + n_steps

        if end_ix > len(sequence)-1:

            break

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]

        X.append(seq_x)

        Y.append(seq_y)

    return np.array(X),np.array(Y)
# prediction_dict = dict()

# output1 = dict()

# country_count = 1

# for country in country_dict: #

#     case_dict = {'ConfirmedCases':1,'Fatalities':2}

#     time_series = country_dict[country]

#     time_series = [x[case_dict[case]][0] for x in time_series]

#     train_x,train_y = split_sequence(time_series,7)

#     reg = xgb.XGBRegressor(n_estimators=100000)

#     reg.fit(train_x,train_y,verbose = True)

#     output1[country]=[]

#     output1[country].extend(time_series[57:])

#     time_series = [time_series[-7:]]

#     time_series = np.array(time_series)

#     x = []

#     pred = reg.predict(time_series)

#     output1[country].append(int(pred[0]))

#     x.extend(list(time_series[0][1:]))

#     x.append(int(pred[0]))

#     time_series[0]=x

#     for i in range(30):

#         time_series = np.array(time_series)

#         x =[]

#         pred = reg.predict(time_series)

#         output1[country].append(int(pred[0]))

#         x.extend(list(time_series[0][1:]))

#         x.append(int(pred[0]))

#         time_series[0]=x

#     out = pd.Series(output1[country], index =date_list) 

#     # Confirmed Cases

#     time_series = country_dict[country]

#     time_series = [x[1][0] for x in time_series]

#     train_x,train_y = split_sequence(time_series,7)

#     reg = xgb.XGBRegressor(n_estimators=100000)

#     reg.fit(train_x,train_y,verbose = True)

#     prediction_dict_confirmedcases[country]=[]

#     prediction_dict_confirmedcases[country].extend(time_series[57:])

#     time_series = [time_series[60:67]]

#     time_series = np.array(time_series)

#     x = []

#     pred = reg.predict(time_series)

#     prediction_dict_confirmedcases[country].append(int(pred[0]))

#     x.extend(list(time_series[0][1:]))

#     x.append(int(pred[0]))

#     time_series[0]=x

#     for i in range(32):

#         time_series = np.array(time_series)

#         x =[]

#         pred = reg.predict(time_series)

#         prediction_dict_confirmedcases[country].append(int(pred[0]))

#         x.extend(list(time_series[0][1:]))

#         x.append(int(pred[0]))

#         time_series[0]=x

#     country_count = country_count+1
from datetime import timedelta, date



def daterange(date1, date2):

    for n in range(int ((date2 - date1).days)+1):

        yield date1 + timedelta(n)
date_list = []

for  i in daterange(pd.to_datetime('2020-03-19'),pd.to_datetime('2020-04-30')):

    date_list.append(i)
import itertools

import statsmodels.api as sm

from scipy import stats

import warnings

warnings.filterwarnings('ignore')
time_series_dict = dict()

for country in country_dict.keys():

    for case in ['ConfirmedCases','Fatalities']:

        tsz=train.loc[(train['Country_Region']==country)]

        tsz=tsz[['Date',case]]

        x = []

        for itr in tsz.index:

            x.append([pd.to_datetime(tsz.loc[itr]['Date']),tsz.loc[itr][case]])

        tsz = pd.DataFrame(x,columns = ['Date',case])

        tsz=tsz.set_index('Date')

        tsz

        if country not in time_series_dict.keys():

            time_series_dict[country] = dict()

        time_series_dict[country][case] = tsz
prediction_country_list = dict()

count = 0

for country in country_dict.keys():

    prediction_country_list[country] = dict()

    for case in ['ConfirmedCases','Fatalities']:

        start = 0

        end = 69

        prediction_country_list[country][case] = []

        len(time_series_dict[country][case])//69

        for i in range(len(time_series_dict[country][case])//69):

            mod = sm.tsa.statespace.SARIMAX(time_series_dict[country][case].iloc[start:end],

                                            order=(1,0,1),

                                            trend= [1,0,1],

                                            enforce_stationarity=True,

                                            enforce_invertibility=False)

            results = mod.fit()

            pred = results.get_prediction(start=pd.to_datetime('2020-03-19'),end=pd.to_datetime('2020-04-30'),dynamic=True )

            prediction_country_list[country][case].append(pred.predicted_mean)

            start = start + 69

            end = end + 69

    count = count+1
forecastid = 1

submission_out = []

for country in country_dict.keys():

    for itr in range(len(prediction_country_list[country]['ConfirmedCases'])):

        for index in prediction_country_list[country]['ConfirmedCases'][itr].index:

            submission_out.append([forecastid,prediction_country_list[country]['ConfirmedCases'][itr][index],prediction_country_list[country]['Fatalities'][itr][index]])

            forecastid = forecastid +1
for i in range(len(submission_out)):

    submission_out[i][1] = round(submission_out[i][1])

    submission_out[i][2] = round(submission_out[i][2])
submission.head()
submission_file = pd.DataFrame(submission_out,columns=['ForecastId','ConfirmedCases','Fatalities'])
submission_file.to_csv('submission.csv',index = False)