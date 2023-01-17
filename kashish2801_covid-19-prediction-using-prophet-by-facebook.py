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

import itertools

import statsmodels.api as sm

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

from matplotlib import rcParams

rcParams['figure.figsize'] = 18,8
train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")

test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/test.csv")

submission = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/submission.csv")
country_dict= dict()

province_list = []

for itr in range(len(train)):

    if train.loc[itr]['Country_Region'] not in country_dict.keys():

        country_dict[train.loc[itr]['Country_Region']]= dict()

    if str(train.iloc[itr]['Province_State']) != 'nan':

        province_list.append(train.iloc[itr]['Province_State'])

        if train.loc[itr]['Province_State'] not in country_dict[train.loc[itr]['Country_Region']].keys():

            country_dict[train.loc[itr]['Country_Region']][train.loc[itr]['Province_State']] = dict()

            country_dict[train.loc[itr]['Country_Region']][train.loc[itr]['Province_State']]['ConfirmedCases'] = []

            country_dict[train.loc[itr]['Country_Region']][train.loc[itr]['Province_State']]['Fatalities'] = []

        country_dict[train.loc[itr]['Country_Region']][train.loc[itr]['Province_State']]['ConfirmedCases'].append([train.loc[itr]['Date'],train.loc[itr]['ConfirmedCases']])

        country_dict[train.loc[itr]['Country_Region']][train.loc[itr]['Province_State']]['Fatalities'].append([train.loc[itr]['Date'],train.loc[itr]['Fatalities']])

    if str(train.loc[itr]['Province_State']) == 'nan':

        province_list.append(train.iloc[itr]['Country_Region'])

        if train.loc[itr]['Country_Region'] not in country_dict[train.loc[itr]['Country_Region']].keys():

            country_dict[train.loc[itr]['Country_Region']][train.loc[itr]['Country_Region']] = dict()

            country_dict[train.loc[itr]['Country_Region']][train.loc[itr]['Country_Region']]['ConfirmedCases'] = []

            country_dict[train.loc[itr]['Country_Region']][train.loc[itr]['Country_Region']]['Fatalities'] = []

        country_dict[train.loc[itr]['Country_Region']][train.loc[itr]['Country_Region']]['ConfirmedCases'].append([train.loc[itr]['Date'],train.loc[itr]['ConfirmedCases']])

        country_dict[train.loc[itr]['Country_Region']][train.loc[itr]['Country_Region']]['Fatalities'].append([train.loc[itr]['Date'],train.loc[itr]['Fatalities']])
for country in country_dict.keys():

    for province in country_dict[country].keys():

        for case in country_dict[country][province].keys():

            for itr in range(len(country_dict[country][province][case])):

                country_dict[country][province][case][itr][0] = pd.to_datetime(country_dict[country][province][case][itr][0])
for country in country_dict.keys():

    for province in country_dict[country].keys():

        for case in country_dict[country][province].keys():

            country_dict[country][province][case] = pd.DataFrame(country_dict[country][province][case],columns=['ds','y'])
test_dates = pd.DataFrame(set(test['Date']),columns=['ds'])
import pandas as pd

from fbprophet import Prophet
for country in country_dict.keys():

    for province in country_dict[country].keys():

        for case in country_dict[country][province].keys():

            m = Prophet()

            m.fit(country_dict[country][province][case])

            forecast = m.predict(test_dates)

            country_dict[country][province][case] = forecast[['ds','yhat']]
submission_list = []

forecastId = 1

for country in country_dict.keys():

    for province in country_dict[country].keys():

        for itr in range(len(country_dict[country][province][case])):

            submission_list.append([forecastId,round(country_dict[country][province]['ConfirmedCases'].iloc[itr]['yhat']),round(country_dict[country][province]['Fatalities'].iloc[itr]['yhat'])])

            forecastId = forecastId+1
submission_file = pd.DataFrame(submission_list,columns =['ForecastId','ConfirmedCases','Fatalities'])

submission_file.to_csv('submission.csv',index = False)