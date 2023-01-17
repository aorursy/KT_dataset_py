import numpy as np

import pandas as pd

import os

import holoviews as hv

from holoviews import opts

hv.extension('bokeh')

from sklearn import linear_model

from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')

print('Train Set Shape = {}'.format(train.shape))

train.head()
test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv')

print('Test Set Shape = {}'.format(test.shape))

test.head()
print(f'Total reported cases are {len(train)}.')

print(f'Total confirmed cases are {int(train["ConfirmedCases"].sum())}.')

print(f'Total fatality cases are {int(train["Fatalities"].sum())}.')

print(f'Total countries are {len(train["Country/Region"].unique())}.')

print(f'Total provinces/states are {len(train.dropna(subset=["Province/State"])["Province/State"].unique())}.')
reported_cases_coutry = train["Country/Region"].value_counts().sort_values()

hv.Bars(reported_cases_coutry).opts(title="The rank of Reported Cases in each countries", color="red",invert_axes=True, width=800, height=1200,tools=['hover'])
confirmed_cases_country = train.groupby('Country/Region').sum()['ConfirmedCases'].sort_values()

hv.Bars(confirmed_cases_country).opts(title="The rank of Confirmed Cases in each countries", color="blue",invert_axes=True, width=800, height=1200,tools=['hover'])
fatality_cases_country = train.groupby('Country/Region').sum()['Fatalities'].sort_values()

hv.Bars(fatality_cases_country).opts(title="The rank of Fatality Cases in each countries", color="green",invert_axes=True, width=800, height=1200,tools=['hover'])
hasstate = train.dropna(subset=['Province/State'])

hasstate.head()
confirmed_cases_coutry_state = hasstate.groupby('Country/Region').sum()['ConfirmedCases'].sort_values()

hv.Bars(confirmed_cases_coutry_state).opts(title="The rank of Confirmed Cases in each countries with State", color="red",invert_axes=True, width=800, height=1200,tools=['hover'])
china = hasstate[hasstate['Country/Region']=='China'].groupby('Province/State').sum()["ConfirmedCases"].sort_values()

china_g = hv.Bars(china).opts(title="The rank of Confirmed Cases in each state of China", color="blue")

china_g.opts(opts.Bars(invert_axes=True, width=800, height=1000,tools=['hover']))
us = hasstate[hasstate['Country/Region']=='US'].groupby('Province/State').sum()["ConfirmedCases"].sort_values()

us_g = hv.Bars(us).opts(title="The rank of Confirmed Cases in each state of US", color="red")

france = hasstate[hasstate['Country/Region']=='France'].groupby('Province/State').sum()["ConfirmedCases"].sort_values()

france_g = hv.Bars(france).opts(title="The rank of Confirmed Cases in each state of France", color="yellow")

uk = hasstate[hasstate['Country/Region']=='United Kingdom'].groupby('Province/State').sum()["ConfirmedCases"].sort_values()

uk_g = hv.Bars(uk).opts(title="The rank of Confirmed Cases in each state of UK", color="green")



(us_g + france_g + uk_g).opts(opts.Bars(invert_axes=True, width=600, height=1000,tools=['hover']))
confirmed_cases_state = hasstate.groupby('Province/State').sum()['ConfirmedCases'].sort_values()

hv.Bars(confirmed_cases_state).opts(title="The rank of Confirmed Cases in each state of all countries", color="blue",invert_axes=True, width=800, height=1200,tools=['hover'])
cases_each_day = train.groupby('Date').sum()

confirmed_cases_each_day = hv.Curve((cases_each_day.index, cases_each_day['ConfirmedCases']),'time','cases').opts(title="The time-series of Confirmed Cases of all countries", color="blue")

fatality_cases_each_day = hv.Curve((cases_each_day.index, cases_each_day['Fatalities']),'time','cases').opts(title="The time-series of Fatality Cases of all countries", color="red")



(confirmed_cases_each_day + fatality_cases_each_day).opts(opts.Curve(width=1500, height=500,tools=['hover'],xrotation=45),opts.Layout(shared_axes=False)).cols(1)
cases_each_day_china = train[train['Country/Region']=='China'].groupby('Date').sum()

cases_each_day_us = train[train['Country/Region']=='US'].groupby('Date').sum()

cases_each_day_france = train[train['Country/Region']=='France'].groupby('Date').sum()

cases_each_day_uk = train[train['Country/Region']=='United Kingdom'].groupby('Date').sum()

cases_each_day_italy = train[train['Country/Region']=='Italy'].groupby('Date').sum()

cases_each_day_spain = train[train['Country/Region']=='Spain'].groupby('Date').sum()

cases_each_day_korea = train[train['Country/Region']=='Korea_South'].groupby('Date').sum()

cases_each_day_japan = train[train['Country/Region']=='Japan'].groupby('Date').sum()



confirmed_cases_each_day_china = hv.Curve((cases_each_day_china.index, cases_each_day_china['ConfirmedCases']),'time','cases',label='China').opts(color="red")

confirmed_cases_each_day_us = hv.Curve((cases_each_day_us.index, cases_each_day_us['ConfirmedCases']),'time','cases',label='US').opts(color="orange")

confirmed_cases_each_day_france = hv.Curve((cases_each_day_france.index, cases_each_day_france['ConfirmedCases']),'time','cases',label='France').opts(color="green")

confirmed_cases_each_day_uk = hv.Curve((cases_each_day_uk.index, cases_each_day_uk['ConfirmedCases']),'time','cases',label='UK').opts(color="blue")

confirmed_cases_each_day_italy = hv.Curve((cases_each_day_italy.index, cases_each_day_italy['ConfirmedCases']),'time','cases',label='Italy').opts(color="purple")

confirmed_cases_each_day_spain = hv.Curve((cases_each_day_spain.index, cases_each_day_spain['ConfirmedCases']),'time','cases',label='Spain').opts(color="pink")

confirmed_cases_each_day_korea = hv.Curve((cases_each_day_korea.index, cases_each_day_korea['ConfirmedCases']),'time','cases',label='South Korea').opts(color="brown")

confirmed_cases_each_day_japan = hv.Curve((cases_each_day_japan.index, cases_each_day_japan['ConfirmedCases']),'time','cases',label='Japan').opts(color="black")



fatality_cases_each_day_china = hv.Curve((cases_each_day_china.index, cases_each_day_china['Fatalities']),'time','cases',label='China').opts(color="red")

fatality_cases_each_day_us = hv.Curve((cases_each_day_us.index, cases_each_day_us['Fatalities']),'time','cases',label='US').opts(color="orange")

fatality_cases_each_day_france = hv.Curve((cases_each_day_france.index, cases_each_day_france['Fatalities']),'time','cases',label='France').opts(color="green")

fatality_cases_each_day_uk = hv.Curve((cases_each_day_uk.index, cases_each_day_uk['Fatalities']),'time','cases',label='UK').opts(color="blue")

fatality_cases_each_day_italy = hv.Curve((cases_each_day_italy.index, cases_each_day_italy['Fatalities']),'time','cases',label='Italy').opts(color="purple")

fatality_cases_each_day_spain = hv.Curve((cases_each_day_spain.index, cases_each_day_spain['Fatalities']),'time','cases',label='Spain').opts(color="pink")

fatality_cases_each_day_korea = hv.Curve((cases_each_day_korea.index, cases_each_day_korea['Fatalities']),'time','cases',label='South Korea').opts(color="brown")

fatality_cases_each_day_japan = hv.Curve((cases_each_day_japan.index, cases_each_day_japan['Fatalities']),'time','cases',label='Japan').opts(color="black")



((confirmed_cases_each_day_china * confirmed_cases_each_day_us * confirmed_cases_each_day_france * fatality_cases_each_day_uk * confirmed_cases_each_day_italy 

  * confirmed_cases_each_day_spain * confirmed_cases_each_day_korea * confirmed_cases_each_day_japan).opts(title="The time-series of Confirmed Cases") 

 + (fatality_cases_each_day_china * fatality_cases_each_day_us * fatality_cases_each_day_france * fatality_cases_each_day_uk * fatality_cases_each_day_italy 

    * fatality_cases_each_day_spain * fatality_cases_each_day_korea * fatality_cases_each_day_japan).opts(title="The time-series of Fatality Cases")).opts(opts.Curve(width=1500, height=500,tools=['hover'],xrotation=45),opts.Layout(shared_axes=False)).cols(1)
for country in train['Country/Region'].unique():

    print ('training model for country ==>'+country)

    country_pd_train = train[train['Country/Region']==country]

    country_pd_test = test[test['Country/Region']==country]

    if country_pd_train['Province/State'].isna().unique()==True:

        x = np.array(range(len(country_pd_train))).reshape((-1,1))

        y = country_pd_train['ConfirmedCases']

        model = Pipeline([('poly', PolynomialFeatures(degree=2)),

                         ('linear', LinearRegression(fit_intercept=False))])

        model = model.fit(x, y)

        predict_x = (np.array(range(len(country_pd_test)))+50).reshape((-1,1))

        test.loc[test['Country/Region']==country,'ConfirmedCases'] = model.predict(predict_x)

    else:

        for state in country_pd_train['Province/State'].unique():

            state_pd = country_pd_train[country_pd_train['Province/State']==state] 

            state_pd_test = country_pd_test[country_pd_test['Province/State']==state] 

            x = np.array(range(len(state_pd))).reshape((-1,1))

            y = state_pd['ConfirmedCases']

            model = Pipeline([('poly', PolynomialFeatures(degree=2)),

                         ('linear', LinearRegression(fit_intercept=False))])

            model = model.fit(x, y)

            predict_x = (np.array(range(len(state_pd_test)))+50).reshape((-1,1))

            test.loc[(test['Country/Region']==country)&(test['Province/State']==state),'ConfirmedCases'] = model.predict(predict_x)
for country in train['Country/Region'].unique():

    print ('training model for country ==>'+country)

    country_pd_train = train[train['Country/Region']==country]

    country_pd_test = test[test['Country/Region']==country]

    if country_pd_train['Province/State'].isna().unique()==True:

        x = np.array(range(len(country_pd_train))).reshape((-1,1))

        y = country_pd_train['Fatalities']

        model = Pipeline([('poly', PolynomialFeatures(degree=2)),

                         ('linear', LinearRegression(fit_intercept=False))])

        model = model.fit(x, y)

        predict_x = (np.array(range(len(country_pd_test)))+50).reshape((-1,1))

        test.loc[test['Country/Region']==country,'Fatalities'] = model.predict(predict_x)

    else:

        for state in country_pd_train['Province/State'].unique():

            state_pd = country_pd_train[country_pd_train['Province/State']==state] 

            state_pd_test = country_pd_test[country_pd_test['Province/State']==state] 

            x = np.array(range(len(state_pd))).reshape((-1,1))

            y = state_pd['Fatalities']

            model = Pipeline([('poly', PolynomialFeatures(degree=2)),

                         ('linear', LinearRegression(fit_intercept=False))])

            model = model.fit(x, y)

            predict_x = (np.array(range(len(state_pd_test)))+50).reshape((-1,1))

            test.loc[(test['Country/Region']==country)&(test['Province/State']==state),'Fatalities'] = model.predict(predict_x)
submit = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/submission.csv')

submit['Fatalities'] = test['Fatalities'].astype('int')

submit['ConfirmedCases'] = test['ConfirmedCases'].astype('int')

submit.to_csv('submission.csv',index=False)

submit.head()