# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

train_df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv", index_col = 'Id')

test_df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/test.csv", index_col = 'ForecastId')

pd.set_option('display.max_columns', 150)

pd.set_option('display.max_rows', 150)



# Any results you write to the current directory are saved as output.
# Remove columns we do not need

cols = ['Date','Fatalities']

times_series_cntr_pr = train_df.drop(cols, axis=1).fillna('NA')



#Double exponential smoothing for Confirmed cases



countries = train_df['Country_Region'].unique()

days_to_predict = len([x for x in train_df.Date.unique() if x not in test_df.Date.unique()]) + test_df.Date.nunique() - train_df.Date.nunique() 



def double_exponential_smoothing(df, alpha, beta):

    """

        series - dataset with timeseries

        alpha - float [0.0, 1.0], smoothing parameter for level

        beta - float [0.0, 1.0], smoothing parameter for trend

    """

    result =[]

    cntr = []

    prov=[]

    for c in countries:

        for p in df.loc[df['Country_Region'] == c]['Province_State'].unique():

            if p is not np.nan :

                series = df.loc[(df['Province_State'] == p) & (df['Country_Region'] == c)].ConfirmedCases

                series = list(series)

                #result.append(series[0])

                for n in range(1, len(series)+days_to_predict +1):

                    if n == 1:

                        level, trend = series[0], series[1] - series[0]

                    if n >= len(series): # forecasting

                        value = result[-1]

                    else:

                        value = series[n]

                    last_level, level = level, alpha*value + (1-alpha)*(level+trend)

                    trend = beta*(level-last_level) + (1-beta)*trend

                    result.append(int(level+trend))

                    prov.append(p)

                    cntr.append(c)



    return result, cntr, prov
t = double_exponential_smoothing(times_series_cntr_pr,0.15, 0.9)

full_cc = pd.DataFrame([t[0],t[1],t[2]], index = ['ConfirmedCases','Country_Region','Province_State'], columns= np.arange(1, len(t[0]) + 1)).T

full_cc.loc[(full_cc['ConfirmedCases'] < 0) ,'ConfirmedCases'] = 0

full_cc = full_cc.sort_values(['Country_Region','ConfirmedCases','Province_State'])

full_cc.shape
# Remove training data



total_days = len([x for x in train_df.Date.unique() if x not in test_df.Date.unique()]) + test_df.Date.nunique() #100

indeces = []

for j in range(0,294):

    for i in range(1,len([x for x in train_df.Date.unique() if x not in test_df.Date.unique()]) + 1): # 57 + 1

        indeces.append((i+j*total_days))



pred_cc = full_cc.drop(indeces).reset_index().ConfirmedCases
pred_cc.shape[0] 
# Remove columns we do not need

cols = ['Date','ConfirmedCases']

times_series_cntr_f = train_df.drop(cols, axis=1).fillna('NA')



#Double exponential smoothing for Fatalities



countries = train_df['Country_Region'].unique()

days_to_predict = len([x for x in train_df.Date.unique() if x not in test_df.Date.unique()]) + test_df.Date.nunique() - train_df.Date.nunique() 



def double_exponential_smoothing_ft(df, alpha, beta):

    """

        series - dataset with timeseries

        alpha - float [0.0, 1.0], smoothing parameter for level

        beta - float [0.0, 1.0], smoothing parameter for trend

    """

    result =[]

    cntr = []

    prov=[]

    for c in countries:

        for p in df.loc[df['Country_Region'] == c]['Province_State'].unique():

            if p is not np.nan :

                series = df.loc[(df['Province_State'] == p) & (df['Country_Region'] == c)].Fatalities

                series = list(series)

                #result.append(series[0])

                for n in range(1, len(series)+days_to_predict +1):

                    if n == 1:

                        level, trend = series[0], series[1] - series[0]

                    if n >= len(series): # forecasting

                        value = result[-1]

                    else:

                        value = series[n]

                    last_level, level = level, alpha*value + (1-alpha)*(level+trend)

                    trend = beta*(level-last_level) + (1-beta)*trend

                    result.append(int(level+trend))

                    prov.append(p)

                    cntr.append(c)



    return result, cntr, prov
f = double_exponential_smoothing_ft(times_series_cntr_f,0.15, 0.9)

full_f = pd.DataFrame([f[0],f[1],f[2]], index = ['Fatalities','Country_Region','Province_State'], columns= np.arange(1, len(f[0]) + 1)).T

full_f.loc[(full_f['Fatalities'] < 0) ,'Fatalities'] = 0



full_f = full_f.sort_values(['Country_Region','Fatalities','Province_State'])

pred_f = full_f.drop(indeces).reset_index().Fatalities
predicted_df = pd.DataFrame([pred_cc, pred_f], index = ['ConfirmedCases','Fatalities']).T

predicted_df.index += 1 

predicted_df.to_csv('submission.csv', index_label = "ForecastId")