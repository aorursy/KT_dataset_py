import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

import warnings

warnings.filterwarnings("ignore")
train = pd.read_csv('../input/covid19-global-forecasting-week-2/train.csv')

train.info()
train.head()
train.tail()
train.describe()
train.corr()
train.isnull().sum()
test = pd.read_csv('../input/covid19-global-forecasting-week-2/test.csv')

test.info()
test.head()
test.tail()
test.describe()
test.isnull().sum()
#Changing dtype for dates from object to datetime

train['Date'] = pd.to_datetime(train['Date'])

test['Date'] = pd.to_datetime(test['Date'])
train[train['Province_State'].isnull()]['Country_Region'].unique()
train[train['Province_State'].notnull()]['Country_Region'].unique()
train['Province_State'] = np.where(train['Province_State'].isnull(), train['Country_Region'], train['Province_State'])

test['Province_State'] = np.where(test['Province_State'].isnull(), test['Country_Region'], test['Province_State'])
train[train['Province_State'] == 'Diamond Princess']
train[train['Province_State'] == 'Diamond Princess']['Country_Region'].unique()
df = train.append(test)
group = df.groupby(['Province_State', 'Country_Region'])['Date'].count().reset_index()

group
group[group['Province_State'].duplicated()]
df[df['Province_State'] == 'Georgia']['Country_Region'].unique()
#Distinguishing Province/State Georgia according to Country/Region

df['Province_State'] = np.where((df['Country_Region'] == 'Georgia') & (df['Province_State'] == 'Georgia'), 

                                'Country Georgia', df['Province_State'])
#Viewing the total number of confirmeed cases and fatalities worldwide

world = train.groupby('Date')['ConfirmedCases', 'Fatalities'].sum().reset_index()



plt.plot(world['Date'], world['ConfirmedCases'], label = 'Confirmed Cases')

plt.plot(world['Date'], world['Fatalities'], label = 'Fatalities')

plt.legend()

plt.title('Total number of Confirmed Cases and Fatalities Worldwide')

plt.xticks(rotation = 30)

plt.show();
#Plotting the number of confirmed cases and fatalities for each country

country = train.groupby('Country_Region')['ConfirmedCases', 'Fatalities'].sum().reset_index()



fig = plt.figure(figsize = (15, 25))

ax = fig.add_subplot(111)

ax.barh(country['Country_Region'], country['ConfirmedCases'],label = 'Confirmed Cases')

ax.barh(country['Country_Region'], country['Fatalities'],label = 'Fatalities')

ax.legend()

ax.set_title('Total Confirmed Cases and Fatalities by Country');
#Viewing the top 15 countries with the most confirmed cases

ranked = country.sort_values(by = 'ConfirmedCases', ascending = False)[:15]

ranked
#Plotting confirmed cases and fatalities for the 15 countries with the most cases

countries = ['China', 'Italy', 'US', 

             'Spain', 'Germany', 'Iran', 

             'France', 'Korea, South', 'United Kingdom', 

             'Switzerland', 'Netherlands', 'Belgium', 

             'Austria', 'Turkey', 'Canada']



for c in countries:

    group = train[train['Country_Region'] == c].groupby('Date')['ConfirmedCases', 'Fatalities'].sum().reset_index()

    group['ConfirmedCases'].plot(label = 'Confirmed Cases')

    group['Fatalities'].plot(label = 'Fatalities')

    plt.legend()

    plt.title(c)

    plt.show();
from statsmodels.tsa.seasonal import seasonal_decompose



def trends(country, case):

    group = train[train['Country_Region'] == country].groupby('Date')['ConfirmedCases', 'Fatalities'].sum().reset_index()

    decomposition = seasonal_decompose(group[case], freq = 3)

    trend = decomposition.trend

    seasonal = decomposition.seasonal

    residual = decomposition.resid

    

    plt.subplot(411)

    plt.plot(group[case], label= case)

    plt.legend(loc='best')

    plt.title('Original')

    plt.subplot(412)

    plt.plot(trend, label=case)

    plt.legend(loc='best')

    plt.title('Trend')

    plt.subplot(413)

    plt.plot(seasonal,label=case)

    plt.legend(loc='best')

    plt.title('Seasonality')

    plt.subplot(414)

    plt.plot(residual, label=case)

    plt.legend(loc='best')

    plt.title('Residual')

    plt.tight_layout();
trends('China', 'ConfirmedCases')
trends('China', 'Fatalities')
trends('Italy', 'ConfirmedCases')
trends('Italy', 'Fatalities')
trends('US', 'ConfirmedCases')
trends('US', 'Fatalities')
from statsmodels.tsa.stattools import adfuller

def stationarity_test(country, case):

    timeseries = train[train['Country_Region'] == country].groupby('Date')['ConfirmedCases', 'Fatalities'].sum().reset_index()

    #Perform Dickey-Fuller test:

    print('Results of Dickey-Fuller Test:')

    dftest = adfuller(timeseries[case], autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value',

                                             '#Lags Used',

                                             'Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    print (dfoutput)
stationarity_test('US', 'ConfirmedCases')
from statsmodels.tsa.arima_model import ARIMA

from sklearn.metrics import mean_squared_error
def comb_p_d_q(pVals,dVals,qVals):

    return [(p,d,q) for p in pVals for d in dVals for q in qVals]
#List of combinations for pdq 

pdq_results = comb_p_d_q([0,1,2],[0,1,2],[0,1,2])

pdq_results
df.drop_duplicates(subset = ['Date', 'Province_State'], keep = 'last', inplace = True)
#Finding the best pdq combination using aic

def aic_finder(province, case):

    train_df = df[df['Province_State'] == province][:70]

    a = 9999

    

    for pdq in pdq_results:

        try:

            model = ARIMA(train_df[case], order = pdq, dates = train_df['Date'], freq = 'D')

            model_fit = model.fit()

            aicval = model_fit.aic

            

            if aicval < a:

                a = aicval

                param = pdq

                print(param, a)

        except:

            pass
aic_finder('Switzerland', 'ConfirmedCases')
from statsmodels.tsa.arima_model import ARIMA
def model_eval(case):

    state = ['Italy']

    for s in state:

        train_ts = train[train['Province_State'] == s][:50]

        test_ts = train[train['Province_State'] == s][50:]

        a = 9999

        

        for pdq in pdq_results:

            try:

                model = ARIMA(train_ts[case], order = pdq, dates = train_ts['Date'], freq = 'D')

                model_fit = model.fit()

                aicval = model_fit.aic

            

                if aicval < a:

                    a = aicval

                    param = pdq

            except:

                pass

        

        model = ARIMA(train_ts[case], order = param, dates = train_ts['Date'], freq = 'D')

        model_fit = model.fit()

        model_fit.plot_predict(start = int(len(train_ts) * 0.3), end = int(len(train_ts) * 1.4))

        pred = model_fit.forecast(steps = int(len(test_ts)))[0]

            

        
model_eval('ConfirmedCases')
model_eval('Fatalities')
def model(case):

    state = df['Province_State'].unique()

    confirmed = []

    for s in state:

        train_ts = df[df['Province_State'] == s][:57]

        pred_ts = df[df['Province_State'] == s][57:]

        a = 9999

        

        for pdq in pdq_results:

            try:

                model = ARIMA(train_ts[case], order = pdq, dates = train_ts['Date'], freq = 'D')

                model_fit = model.fit()

                aicval = model_fit.aic

            

                if aicval < a:

                    a = aicval

                    param = pdq

            except:

                pass

        

        try:

            model = ARIMA(train_ts[case], order = param, dates = train_ts['Date'], freq = 'D')

            model_fit = model.fit()

            pred = model_fit.forecast(steps = int(len(pred_ts)))[0]

            confirmed = np.append(confirmed, pred.tolist())

        except:

            confirmed = np.append(confirmed, np.repeat(0, 43))

            continue

            

    test[case] = confirmed
model('ConfirmedCases')
model('Fatalities')
results = test[['ForecastId', 'ConfirmedCases', 'Fatalities']]

results.to_csv('submission.csv', index = False)