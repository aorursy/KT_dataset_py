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
#importation des librairies

import pandas as pd

import numpy as np

import itertools

import matplotlib.pyplot 

import statsmodels.api as sm

%matplotlib inline
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')

train.info()
#Data imputation

def impute(df):

    df['Province_State'] = df['Province_State'].mask(df['Province_State'].isnull(), df['Country_Region'])

    return df

train_imputed = impute(train)

train_imputed.info()
#train dataset for confirmed cases forecasting

train_cc = train_imputed.drop(['Id','Fatalities'], 1)

train_cc["Date"]= pd.to_datetime(train_cc["Date"])

train_cc.set_index(['Country_Region','Date'], inplace=True)



#train dataset for fatalities forecasting

train_f = train_imputed.drop(['Id', 'ConfirmedCases'], 1)

train_f["Date"]= pd.to_datetime(train_f["Date"])

train_f.set_index(['Country_Region','Date'], inplace=True)
#preparing parameters for SARIMAX model

p = d = q = range(0, 2)

pdq = list(itertools.product(p, d, q))

seasonal_pdq = [(x[0], x[1], x[2], 7) for x in list(itertools.product(p, d, q))]



def param(df) :

    liste = dict()

    for param in pdq:

        for param_seasonal in seasonal_pdq:

            try:

                mod = sm.tsa.statespace.SARIMAX(y,

                                            order=param,

                                            seasonal_order=param_seasonal,

                                            enforce_stationarity=False,

                                            enforce_invertibility=False)

                results = mod.fit()

                key = str(param)+','+str(param_seasonal)

                liste[key] = results.aic

                

            except:

                continue

    key_min = min(liste, key=liste.get)

    k = key_min.replace('(','').replace(')','').split(',')

    i = [int(x) for x in k]

    par = tuple(i[:3])

    par_seas = tuple(i[3:])

    return par, par_seas
#extracting countries list

liste_pays = train_cc.index.get_level_values(0).unique()
#training model for confirmed cases forecasting

rmsle_cc_pays=dict()

mle_cc_retval=dict()

list_cc_results = dict()

predictions_cc = dict()

list_cc_y = dict()



for elmt in liste_pays:

    df = train_cc.loc[(elmt)]

    if len(df['Province_State'].unique())==1 :

        y = df['ConfirmedCases'].resample('D').mean()

        list_cc_y[elmt+' '+elmt] = y

        par, par_seas = param(y)

        mod = sm.tsa.statespace.SARIMAX(y,

                                order=par,

                                seasonal_order=par_seas,

                                enforce_stationarity=False,

                                enforce_invertibility=False)

        results = mod.fit()

        list_cc_results[elmt+' '+elmt] = results

        pred = results.get_prediction(start=pd.to_datetime('2020-01-22'), dynamic=False)

        predictions_cc[elmt+' '+elmt] = pred

        y_forecasted = pred.predicted_mean

        y_truth = y.copy()

        rmsle =np.sqrt(np.square(np.log(y_forecasted + 1) - np.log(y_truth + 1)).mean())

        rmsle_cc_pays[elmt+' '+elmt] = round(rmsle,2)

        mle_cc_retval[elmt+' '+elmt] = results.mle_retvals

    else :

        for elt in df['Province_State'].unique():

            d = df.loc[df['Province_State']==elt]['ConfirmedCases'].resample('D').mean()

            list_cc_y[elmt+' '+elt] = d

            par, par_seas = param(d)

            mod = sm.tsa.statespace.SARIMAX(d,

                                order=par,

                                seasonal_order=par_seas,

                                enforce_stationarity=False,

                                enforce_invertibility=False)

            results = mod.fit()

            list_cc_results[elmt+' '+elt] = results

            pred = results.get_prediction(start=pd.to_datetime('2020-01-22'), dynamic=False)

            predictions_cc[elmt+' '+elt] = pred

            y_forecasted = pred.predicted_mean

            y_truth = d.copy()

            rmsle =np.sqrt(np.square(np.log(y_forecasted + 1) - np.log(y_truth + 1)).mean())

            rmsle_cc_pays[elmt+' '+elt] = round(rmsle,2)

            mle_cc_retval[elmt+' '+elt] = results.mle_retvals
rmsle_f_pays=dict()

mle_f_retval=dict()

list_f_results = dict()

predictions_f = dict()

list_f_y = dict()

for elmt in liste_pays:

    df = train_f.loc[(elmt)]

    if len(df['Province_State'].unique())==1 :

        y = df['Fatalities'].resample('D').mean()

        list_f_y[elmt+' '+elt] = y

        par, par_seas = param(y)

        mod = sm.tsa.statespace.SARIMAX(y,

                                order=par,

                                seasonal_order=par_seas,

                                enforce_stationarity=False,

                                enforce_invertibility=False)

        results = mod.fit()

        list_f_results[elmt+' '+elt] = results

        pred = results.get_prediction(start=pd.to_datetime('2020-01-22'), dynamic=False)

        predictions_f[elmt+' '+elt] = pred

        y_forecasted = pred.predicted_mean

        y_truth = y.copy()

        rmsle =np.sqrt(np.square(np.log(y_forecasted + 1) - np.log(y_truth + 1)).mean())

        rmsle_f_pays[elmt+' '+elt] = round(rmsle,2)

        mle_f_retval[elmt+' '+elt] = results.mle_retvals

    else :

        for elt in df['Province_State'].unique():

            d = df.loc[df['Province_State']==elt]['Fatalities'].resample('D').mean()

            list_f_y[elmt+' '+elt] = d

            par, par_seas = param(d)

            mod = sm.tsa.statespace.SARIMAX(d,

                                order=par,

                                seasonal_order=par_seas,

                                enforce_stationarity=False,

                                enforce_invertibility=False)

            results = mod.fit()

            list_f_results[elmt+' '+elt] = results

            pred = results.get_prediction(start=pd.to_datetime('2020-01-22'), dynamic=False)

            predictions_f[elmt+' '+elt] = pred

            y_forecasted = pred.predicted_mean

            y_truth = d.copy()

            rmsle =np.sqrt(np.square(np.log(y_forecasted + 1) - np.log(y_truth + 1)).mean())

            rmsle_f_pays[elmt+' '+elt] = round(rmsle,2)

            mle_f_retval[elmt+' '+elt] = results.mle_retvals
#rmsle cc

error = np.array(list(rmsle_cc_pays.values())).mean()

error
#rmsle f

error = np.array(list(rmsle_f_pays.values())).mean()

error
#Représentation du forecast de la RCI

pred_ci = predictions_cc["Cote d'Ivoire Cote d'Ivoire"].conf_int()

ax = train_cc.loc[("Cote d'Ivoire")].plot(label='observed')

predictions_cc["Cote d'Ivoire Cote d'Ivoire"].predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))

ax.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')

ax.set_ylabel('Confirmed Cases')

plt.legend()

plt.show()
# Diagnostic forecasting RCI

print(list_cc_results["Cote d'Ivoire Cote d'Ivoire"].summary().tables[1])

list_cc_results["Cote d'Ivoire Cote d'Ivoire"].plot_diagnostics(figsize=(16, 8))

plt.show()
holdout = impute(test)

holdout.head()
#forecasting pour les données tests (confirmed cases)

cc_holdout_res_list = dict()

cc_holdout_pred_list = dict()

holdout_cc_y = dict()



for elmt in liste_pays:

    df = train_cc.loc[(elmt)]

    if len(df['Province_State'].unique())==1 :

        y = df['ConfirmedCases'].resample('D').mean()

        holdout_cc_y[elmt+' '+elmt] = y

        par, par_seas = param(y)

        mod = sm.tsa.statespace.SARIMAX(y,

                                order=par,

                                seasonal_order=par_seas,

                                enforce_stationarity=False,

                                enforce_invertibility=False)

        results = mod.fit()

        cc_holdout_res_list[elmt+' '+elmt] = results

        pred = results.get_prediction(start=pd.to_datetime('2020-04-02'), end=pd.to_datetime('2020-05-14'), dynamic=False)

        cc_holdout_pred_list[elmt+' '+elmt] = pred

        

    else :

        for elt in df['Province_State'].unique():

            d0 = df.loc[df['Province_State']==elt]

            d=d0['ConfirmedCases'].resample('D').mean()

            holdout_cc_y[elmt+' '+elt] = d

            par, par_seas = param(d)

            mod = sm.tsa.statespace.SARIMAX(d,

                                order=par,

                                seasonal_order=par_seas,

                                enforce_stationarity=False,

                                enforce_invertibility=False)

            results = mod.fit()

            cc_holdout_res_list[elmt+' '+elt] = results

            pred = results.get_prediction(start=pd.to_datetime('2020-04-02'), end=pd.to_datetime('2020-05-14'), dynamic=False)

            cc_holdout_pred_list[elmt+' '+elt] = pred
#forecasting pour les données tests (fatalities)

f_holdout_res_list = dict()

f_holdout_pred_list = dict()

holdout_f_y = dict()



for elmt in liste_pays:

    df = train_f.loc[(elmt)]

    if len(df['Province_State'].unique())==1 :

        y = df.loc['2020-04-02':,'Fatalities'].resample('D').mean()

        holdout_f_y[elmt+' '+elmt] = y

        par, par_seas = param(y)

        mod = sm.tsa.statespace.SARIMAX(y,

                                order=par,

                                seasonal_order=par_seas,

                                enforce_stationarity=False,

                                enforce_invertibility=False)

        results = mod.fit()

        f_holdout_res_list[elmt+' '+elmt] = results

        pred = results.get_prediction(start=pd.to_datetime('2020-04-02'), end=pd.to_datetime('2020-05-14'), dynamic=False)

        f_holdout_pred_list[elmt+' '+elmt] = pred

        

    else :

        for elt in df['Province_State'].unique():

            d0 = df.loc[df['Province_State']==elt]

            d=d0.loc['2020-04-02':,'Fatalities'].resample('D').mean()

            holdout_f_y[elmt+' '+elt] = d

            par, par_seas = param(d)

            mod = sm.tsa.statespace.SARIMAX(d,

                                order=par,

                                seasonal_order=par_seas,

                                enforce_stationarity=False,

                                enforce_invertibility=False)

            results = mod.fit()

            f_holdout_res_list[elmt+' '+elt] = results

            pred = results.get_prediction(start=pd.to_datetime('2020-04-02'), end=pd.to_datetime('2020-05-14'), dynamic=False)

            f_holdout_pred_list[elmt+' '+elt] = pred
#forecasting from april 4th to may 15th RCI (confirmed cases)

cc_holdout_pred_list["Cote d'Ivoire Cote d'Ivoire"].predicted_mean
#forecasting from april 4th to may 15th RCI (fatalities)

cc_holdout_pred_list["Cote d'Ivoire Cote d'Ivoire"].predicted_mean
#construction du fichier submissions.csv

confirmed_cases = list()

for elmt in cc_holdout_pred_list.keys():

    confirmed_cases.append(cc_holdout_pred_list[elmt].predicted_mean)

    

fatalities = list()

for elmt in f_holdout_pred_list.keys():

    fatalities.append(f_holdout_pred_list[elmt].predicted_mean)



flat_confirmed_cases = list()

flat_confirmed_cases = [item for sublist in confirmed_cases for item in sublist]



flat_fatalities = list()

flat_fatalities = [item for sublist in fatalities for item in sublist]
len(flat_confirmed_cases)
len(flat_fatalities)
forecastId = list(holdout['ForecastId'])

len(forecastId)
df3 = pd.DataFrame(list(zip(forecastId, flat_confirmed_cases, flat_fatalities)), 

               columns =['ForecastId', 'ConfirmedCases', 'Fatalities']) 
df3.to_csv("submission.csv", index=False)
df4 = pd.read_csv('submission.csv')

df4.info()