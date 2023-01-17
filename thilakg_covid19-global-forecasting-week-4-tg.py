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

import datetime as dt

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

import pickle
import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")

test = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")
submission = pd.read_csv("../input/covid19-global-forecasting-week-4/submission.csv")
train.columns = ['ForecastId', 'Province_State', 'Country_Region', 'Date', 'ConfirmedCases', 'Fatalities']
train_date_end = train['Date'].max()

test_date_st = test['Date'].min()
test = test[test['Date'] > train_date_end]
master = train.append(test)
master['Province_State'] = master['Province_State'].fillna('')

master['Country_State'] = master['Country_Region'].str.cat(master['Province_State'],sep=" ")
lockdown_dates = pd.read_csv("../input/lockdown-dates-upd1/lockdown_dates.csv")

#self ccurated file from information found on Wikipedia and news channels. It is not complete for all Countries. So wherever not available or not added, I have defaulted the lockdown dates to 10 Mar 2020
lockdown_dates['Country_State'] = lockdown_dates['Country_Region'].str.cat(lockdown_dates.Province_State.fillna(''),sep=" ")
gov_stringency_data = pd.read_csv("../input/stringency-index/CSVDownload")

#lockdown strigency data provided by University of Oxford https://www.bsg.ox.ac.uk/research/research-projects/coronavirus-government-response-tracker
gov_stringency_data_use = gov_stringency_data[['CountryName','Date','StringencyIndex']]

gov_stringency_data_use['Date'] = pd.to_datetime(gov_stringency_data_use['Date'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
gov_stringency_data_use['CountryName'][gov_stringency_data_use['CountryName'] == 'United States'] = 'US'

gov_stringency_data_use['CountryName'][gov_stringency_data_use['CountryName'] == 'South Korea'] = 'Korea South'

gov_stringency_data_use['CountryName'][gov_stringency_data_use['CountryName'] == 'Myanmar'] = 'Burma'
gov_stringency_data_agg = gov_stringency_data_use[gov_stringency_data_use['StringencyIndex'] >= 50.0].groupby(['CountryName'])['StringencyIndex'].max().reset_index()

#considering an index of above 50 as a serious initiative for lockdown
gov_stringency_data_use = pd.merge(gov_stringency_data_use, gov_stringency_data_agg

                                   ,how='inner', on=['CountryName'])

gov_stringency_data_use['StringencyIndex_diff_max'] = gov_stringency_data_use['StringencyIndex_x'] - gov_stringency_data_use['StringencyIndex_y']
gov_stringency_data_agg_use = gov_stringency_data_use[gov_stringency_data_use['StringencyIndex_diff_max'] >= -10.0].groupby(['CountryName'])['Date'].min().reset_index()

gov_stringency_data_agg_use.columns = ['Country_Region', 'Stringency_Date']
lockdown_dates_upd = pd.merge(lockdown_dates, gov_stringency_data_agg_use

                              ,how='left', on=['Country_Region'])
lockdown_dates_upd['Lockdown_date_use'] = ''

lockdown_dates_upd['Lockdown_date_use'][lockdown_dates_upd['Country_Region'].isin(['US','China','Australia']) == True] = lockdown_dates_upd['Lockdown_date']

lockdown_dates_upd['Lockdown_date_use'][lockdown_dates_upd['Country_Region'].isin(['US','China','Australia']) == False] = lockdown_dates_upd['Stringency_Date'].combine_first(lockdown_dates_upd['Lockdown_date'])

#for some countries, state level lockdown dates are more accurate, hence using information from the self curated file instead of Stringency index
def cc_model_poly(dataset,start_date,end_date,start_value,Country_State,knot_number):

    dataset = dataset[(dataset['Date'] >= start_date) & (dataset['Date'] <= end_date)]

    dataset['Date_diff'] = (((pd.to_datetime(dataset['Date']) - pd.to_datetime(start_date)).dt.days))

    dataset['Date_diff_p2'] = dataset['Date_diff']**2

    dataset['Date_diff_p3'] = dataset['Date_diff']**3

    input_columns = ['Date_diff','Date_diff_p2','Date_diff_p3']

    

    if len(dataset.dropna()) > 5:

        start_value = dataset['ConfirmedCases_inc'].values[0]

        dataset['ConfirmedCases_inc_use'] = dataset['ConfirmedCases_inc_sm']

        reg = LinearRegression().fit(dataset.dropna()[input_columns], dataset.dropna()['ConfirmedCases_inc_use'])

        dataset['ConfirmedCases_pred'] = reg.predict(dataset[input_columns])

#         print(reg.score(dataset.dropna()[input_columns], dataset.dropna()['ConfirmedCases_inc_use']))

#         print(reg.coef_, reg.intercept_)

        filename = 'cc_'+Country_State+'_'+knot_number+'.sav'

        pickle.dump(reg, open(filename, 'wb'))

    else:

        print('..picking fallback cc model for knot '+knot_number+' ..')

        reg = pickle.load(open('cc_China Hubei'+'_'+knot_number+'.sav', 'rb'))

#         print(reg.coef_, loaded_model.intercept_)

        dataset['ConfirmedCases_pred'] = reg.predict(dataset[input_columns]) - reg.intercept_ + start_value



#     outlier treatment for middle knot

    if knot_number == '2':

        cap_upper = dataset['ConfirmedCases_inc'].mean() + (dataset['ConfirmedCases_inc'].std())

        cap_lower = dataset['ConfirmedCases_inc'].mean() - (dataset['ConfirmedCases_inc'].std())

        dataset['ConfirmedCases_pred'] = dataset['ConfirmedCases_pred'].clip(lower=cap_lower, upper=cap_upper)

    

    dataset['ConfirmedCases_pred'][dataset['ConfirmedCases_pred'] < 0] = 0

    return(dataset)
def cc_model_exp(dataset,start_date,end_date,start_value,Country_State,knot_number):

    dataset = dataset[(dataset['Date'] >= start_date) & (dataset['Date'] <= end_date)]

    dataset['Date_diff'] = (((pd.to_datetime(dataset['Date']) - pd.to_datetime(start_date)).dt.days))

#     input_columns = [['Date_diff']]

    

    if len(dataset.dropna()) > 5:

        dataset['ConfirmedCases_inc_use'] = dataset['ConfirmedCases_inc_sm'] + 0.0001 # to avoid infinity values from log(0)

        reg = LinearRegression().fit(dataset.dropna()[['Date_diff']], np.log(dataset.dropna()['ConfirmedCases_inc_use']))

        dataset['ConfirmedCases_pred'] = np.exp(reg.predict(dataset[['Date_diff']]))

#         print(reg.score(dataset.dropna()[['Date_diff']], dataset.dropna()['ConfirmedCases_inc_use']))

#         print(reg.coef_, reg.intercept_)

        filename = 'cc_'+Country_State+'_'+knot_number+'.sav'

        pickle.dump(reg, open(filename, 'wb'))

    else:

        print('..picking fallback cc model for knot '+knot_number+' ..')

        reg = pickle.load(open('cc_China Hubei'+'_'+knot_number+'.sav', 'rb'))

#         print(reg.coef_, reg.intercept_)

        dataset['ConfirmedCases_pred'] = np.exp(reg.predict(dataset[['Date_diff']]) - reg.intercept_ + np.log(start_value))

    

    if reg.coef_ > 0:

        # instances where confirmed cases are increasing even after significant lockdown, it is not ideal to fit the last knot into an exponential model, hence returning an avg of last three values

        print('..skipping exp model as the curve is not declining for knot '+knot_number+' ..')

        dataset['ConfirmedCases_pred'] = dataset['ConfirmedCases_inc'][dataset['ConfirmedCases'] > 0].mean()



    return(dataset)
def fa_model_lnr(dataset,Country_State):

    dataset['ConfirmedCases_lagn'] = dataset['ConfirmedCases'].shift(periods=5)

    dataset['ConfirmedCases_pred_lagn'] = dataset['ConfirmedCases_pred'].shift(periods=5).fillna(0)

    

    if len(dataset.dropna()) > 10:

        reg = LinearRegression(fit_intercept=False).fit(dataset.dropna()[['ConfirmedCases_lagn']], dataset.dropna()['Fatalities'])

        dataset['Fatalities_pred'] = reg.predict(dataset[['ConfirmedCases_pred_lagn']])

#         print(reg.score(dataset.dropna()[['ConfirmedCases_lagn']], dataset.dropna()['Fatalities']))

#         print(reg.coef_, reg.intercept_)

        filename = 'fa_'+Country_State+'_'+'.sav'

        pickle.dump(reg, open(filename, 'wb'))

    else:

        print('..picking fallback fa model..')

        loaded_model = pickle.load(open('fa_China Hubei'+'_'+'.sav', 'rb'))

#         print(loaded_model.coef_, loaded_model.intercept_)

        dataset['Fatalities_pred'] = loaded_model.predict(dataset[['ConfirmedCases_pred_lagn']])



    return(dataset)
def cc_plot_outputs(dataset):

    fig = plt.figure()

    plt.ioff()

    fig.set_size_inches(18.5, 10.5)

    plt.plot(dataset['Date'], dataset['ConfirmedCases_inc'], marker='o', color='lightgreen')

    plt.plot(dataset['Date'], dataset['ConfirmedCases_pred'], color='r')

#     fig.savefig('plots/cc/'+Country_State+'.jpg')

#     plt.close(fig)
def fa_plot_outputs(dataset):

    fig = plt.figure()

    plt.ioff()

    fig.set_size_inches(18.5, 10.5)

    plt.plot(dataset['Date'], dataset['Fatalities'], marker='o', color='lightgreen')

    plt.plot(dataset['Date'], dataset['Fatalities_pred'], color='r')

#     fig.savefig('plots/fa/'+Country_State+'.jpg')

#     plt.close(fig)
test_update = pd.DataFrame(columns = ['Province_State', 'Country_Region', 'Date', 'ConfirmedCases_pred' ,'Fatalities_pred'])

for Country_State in ['China Hubei'] + master['Country_State'].unique().tolist():

    print('running for: '+Country_State)

    master_use = master[master['Country_State'] == Country_State]

    master_use['ConfirmedCases_inc'] = master_use['ConfirmedCases'] - master_use['ConfirmedCases'].shift(periods=1).fillna(0)

    master_use['ConfirmedCases_inc_sm'] = master_use['ConfirmedCases_inc'].rolling(window=5,center=True,min_periods=1).mean()



    st_dt = master_use['Date'][master_use['ConfirmedCases'] > 0].min()

    mx_dt = master_use['Date'].max()

    ld_dt = lockdown_dates_upd['Lockdown_date_use'][lockdown_dates_upd['Country_State'] == Country_State].values[0]

    

    try:    

        if (ld_dt != '9999-12-31'): # countries with lockdown imposed

            add_wdw = 3

            kt_1p = (pd.to_datetime(ld_dt)+dt.timedelta(days=14+add_wdw)).strftime("%Y-%m-%d")

            kt_1m = (pd.to_datetime(ld_dt)+dt.timedelta(days=14-add_wdw)).strftime("%Y-%m-%d")

            kt_2p = (pd.to_datetime(ld_dt)+dt.timedelta(days=28+add_wdw)).strftime("%Y-%m-%d")

            kt_2m = (pd.to_datetime(ld_dt)+dt.timedelta(days=28-add_wdw)).strftime("%Y-%m-%d")



            a = cc_model_poly(master_use, st_dt, kt_1p, 0, Country_State, '1')

            b = cc_model_poly(master_use, kt_1m, kt_2p, a.tail(1)['ConfirmedCases_pred'].values[0], Country_State, '2')

            c = cc_model_exp(master_use, kt_2m, mx_dt, b.tail(1)['ConfirmedCases_pred'].values[0], Country_State, '3')

            master_use = a.append(b).append(c)

            master_use = master_use.groupby(['Province_State', 'Country_Region', 'Date', 'Country_State'])['ConfirmedCases', 'ConfirmedCases_pred', 'ConfirmedCases_inc', 'Fatalities'].mean().reset_index()

#             cc_plot_outputs(master_use)



        else: # countries with no lockdown. uses only one model

            master_use = cc_model_poly(master_use, st_dt, mx_dt, 0, Country_State, '1')

#             cc_plot_outputs(master_use)



        # update dataset for fatalities model

        master_use['ConfirmedCases_pred'] = master_use['ConfirmedCases_inc'].combine_first(master_use['ConfirmedCases_pred'])

        master_use['ConfirmedCases_pred'] = master_use['ConfirmedCases_pred'].cumsum()



        master_use = fa_model_lnr(master_use, Country_State)

#         fa_plot_outputs(master_use)

        

        master_use['Fatalities_pred'] = master_use['Fatalities'].combine_first(master_use['Fatalities_pred'])

        test_use = master_use[master_use['Date'] >= test_date_st]

        test_update = test_update.append(test_use[['Province_State', 'Country_Region', 'Date', 'ConfirmedCases_pred', 'Fatalities_pred']])

    

    except:

        print('exiting due to error')

        pass
missing_inst = list(set(master['Country_State'].unique())-set(test_update['Country_Region'].str.cat(test_update['Province_State'],sep=" ").unique()))
for Country_State in missing_inst:

    print('running for: '+Country_State)

    master_use = master[master['Country_State'] == Country_State]

    

    cc_ext = master_use['ConfirmedCases'].max()

    fa_ext = master_use['Fatalities'].max()

    master_use['ConfirmedCases_pred'] = cc_ext

    master_use['Fatalities_pred'] = fa_ext

    

    master_use['ConfirmedCases_pred'] = master_use['ConfirmedCases'].combine_first(master_use['ConfirmedCases_pred'])

    test_use = master_use[master_use['Date'] >= test_date_st]

    test_update = test_update.append(test_use[['Province_State', 'Country_Region', 'Date', 'ConfirmedCases_pred', 'Fatalities_pred']])
test_update = test_update.groupby(['Province_State', 'Country_Region', 'Date'])['ConfirmedCases_pred','Fatalities_pred'].max().reset_index()
test_orig = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")

test_orig['Province_State'] = test_orig['Province_State'].fillna('')
sub = pd.merge(test_orig[['ForecastId','Province_State','Country_Region', 'Date']], test_update[['Province_State','Country_Region', 'Date', 'ConfirmedCases_pred','Fatalities_pred']], how='left', on=['Province_State','Country_Region', 'Date'])
sub['ConfirmedCases_pred'] = sub['ConfirmedCases_pred'].fillna(0)

sub['Fatalities_pred'] = sub['Fatalities_pred'].fillna(0)
sub = sub[['ForecastId','ConfirmedCases_pred','Fatalities_pred']]
sub.columns = submission.columns
sub.to_csv("submission.csv",index=False)
for dirname, _, filenames in os.walk('..'):

    for filename in filenames:

        if filename.lower().endswith('.sav'):

            os.remove(os.path.join(dirname, filename))