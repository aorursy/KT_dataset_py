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
import plotly.express as px

import plotly.graph_objs as go

from plotly.subplots import make_subplots

import plotly

plotly.offline.init_notebook_mode() # For not show up chart error



import matplotlib.pyplot as plt

import matplotlib.animation as animation

from IPython.display import HTML

%matplotlib inline



from tqdm import tqdm



def RMSLE(pred,actual):

    return np.sqrt(np.mean(np.power((np.log(pred+1)-np.log(actual+1)),2)))
pd.set_option('mode.chained_assignment', None)

test = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")

train = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")

train['Province_State'].fillna('', inplace=True)

test['Province_State'].fillna('', inplace=True)

train['Date'] =  pd.to_datetime(train['Date'])

test['Date'] =  pd.to_datetime(test['Date'])

train = train.sort_values(['Country_Region','Province_State','Date'])

test = test.sort_values(['Country_Region','Province_State','Date'])
# Fix error in train data



train[['ConfirmedCases', 'Fatalities']] = train.groupby(['Country_Region', 'Province_State'])[['ConfirmedCases', 'Fatalities']].transform('cummax') 

import warnings

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt



feature_day = [1,20,50,100,200,500,1000]



def CreateInput(data):

    feature = []

    for day in feature_day:

        #Get information in train data

        data.loc[:,'Number day from ' + str(day) + ' case'] = 0

        if (train[(train['Country_Region'] == country) & (train['Province_State'] == province) & (train['ConfirmedCases'] < day)]['Date'].count() > 0):

            fromday = train[(train['Country_Region'] == country) & (train['Province_State'] == province) & (train['ConfirmedCases'] < day)]['Date'].max()        

        else:

            fromday = train[(train['Country_Region'] == country) & (train['Province_State'] == province)]['Date'].min()       

        for i in range(0, len(data)):

            if (data['Date'].iloc[i] > fromday):

                day_denta = data['Date'].iloc[i] - fromday

                data['Number day from ' + str(day) + ' case'].iloc[i] = day_denta.days 

        feature = feature + ['Number day from ' + str(day) + ' case']

    

    return data[feature]

pred_data_all = pd.DataFrame()

with tqdm(total=len(train['Country_Region'].unique())) as pbar:

    for country in train['Country_Region'].unique():

    #for country in ['Vietnam']:

        for province in train[(train['Country_Region'] == country)]['Province_State'].unique():

            with warnings.catch_warnings():

                warnings.filterwarnings("ignore")

                df_train = train[(train['Country_Region'] == country) & (train['Province_State'] == province)]

                df_test = test[(test['Country_Region'] == country) & (test['Province_State'] == province)]

                X_train = CreateInput(df_train)

                y_train_confirmed = df_train['ConfirmedCases'].ravel()

                y_train_fatalities = df_train['Fatalities'].ravel()

                X_pred = CreateInput(df_test)



                # Define feature to use by X_pred

                feature_use = X_pred.columns[0]

                for i in range(X_pred.shape[1] - 1,0,-1):

                    if (X_pred.iloc[0,i] > 0):

                        feature_use = X_pred.columns[i]

                        break

                idx = X_train[X_train[feature_use] == 0].shape[0]   



                adjusted_X_train = X_train[idx:][feature_use].values.reshape(-1, 1)

                adjusted_y_train_confirmed = y_train_confirmed[idx:]

                adjusted_y_train_fatalities = y_train_fatalities[idx:] #.values.reshape(-1, 1)



                pred_data = test[(test['Country_Region'] == country) & (test['Province_State'] == province)]

                max_train_date = train[(train['Country_Region'] == country) & (train['Province_State'] == province)]['Date'].max()

                min_test_date = pred_data['Date'].min()

                #The number of day forcast

                #pred_data[pred_data['Date'] > max_train_date].shape[0]

                #model = SimpleExpSmoothing(adjusted_y_train_confirmed).fit()

                #model = Holt(adjusted_y_train_confirmed).fit()

                #model = Holt(adjusted_y_train_confirmed, exponential=True).fit()

                #model = Holt(adjusted_y_train_confirmed, exponential=True, damped=True).fit()



                model = ExponentialSmoothing(adjusted_y_train_confirmed, trend = 'additive').fit()

                y_hat_confirmed = model.forecast(pred_data[pred_data['Date'] > max_train_date].shape[0])

                y_train_confirmed = train[(train['Country_Region'] == country) & (train['Province_State'] == province) & (train['Date'] >=  min_test_date)]['ConfirmedCases'].values

                y_hat_confirmed = np.concatenate((y_train_confirmed,y_hat_confirmed), axis = 0)



                #model = Holt(adjusted_y_train_fatalities).fit()



                model = ExponentialSmoothing(adjusted_y_train_fatalities, trend = 'additive').fit()

                y_hat_fatalities = model.forecast(pred_data[pred_data['Date'] > max_train_date].shape[0])

                y_train_fatalities = train[(train['Country_Region'] == country) & (train['Province_State'] == province) & (train['Date'] >=  min_test_date)]['Fatalities'].values

                y_hat_fatalities = np.concatenate((y_train_fatalities,y_hat_fatalities), axis = 0)





                pred_data['ConfirmedCases_hat'] =  y_hat_confirmed

                pred_data['Fatalities_hat'] = y_hat_fatalities

                pred_data_all = pred_data_all.append(pred_data)

        pbar.update(1)



df_val = pd.merge(pred_data_all,train[['Date','Country_Region','Province_State','ConfirmedCases','Fatalities']],on=['Date','Country_Region','Province_State'], how='left')

df_val.loc[df_val['Fatalities_hat'] < 0,'Fatalities_hat'] = 0

df_val.loc[df_val['ConfirmedCases_hat'] < 0,'ConfirmedCases_hat'] = 0

df_val_2 = df_val.copy()
df_val = df_val_2

submission = df_val[['ForecastId','ConfirmedCases_hat','Fatalities_hat']]

submission.columns = ['ForecastId','ConfirmedCases','Fatalities']

submission = submission.round({'ConfirmedCases': 0, 'Fatalities': 0})

submission.to_csv('submission.csv', index=False)

submission