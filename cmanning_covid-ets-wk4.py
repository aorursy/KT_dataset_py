# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing

from sklearn.decomposition import TruncatedSVD

from multiprocessing import Pool



from warnings import catch_warnings

from warnings import filterwarnings

filterwarnings('ignore')



import matplotlib.pyplot as plt

plt.style.use('ggplot')

plt.rcParams["figure.figsize"] = (20, 6)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def auto_ets(df, seasonal_periods=[None], trend=['add', 'mul'], damped=[True], seasonal=[None], use_boxcox=[False]):

    min_ic = np.inf

    best_model = None

    params = [(sp, t, d, s, b) for sp in seasonal_periods for t in trend for d in damped for s in seasonal for b in use_boxcox]

    for sp, t, d, s, b in params:

        try:

            with catch_warnings():

                filterwarnings('ignore')

                ets = ExponentialSmoothing(df, seasonal_periods=sp, trend=t, damped=d, seasonal=s).fit(use_boxcox=b, remove_bias=False)

            if ets.aicc < min_ic:

                min_ic = ets.aicc

                best_model = ets

        except:

            pass

    return best_model



def naive(df, periods):

    return df.shift(periods, freq='D')



def fit_predict(data, forecast_period=43):

    if sum(data != 0) > 28:

        first_idx = data.index[data != 0][0]

        dat = data[first_idx:]

    else:

        dat = data.copy()

    model = auto_ets(dat)

    fcast = model.forecast(forecast_period)

    return fcast.dropna(), model



def fit_predict_pool(data, forecast_period=43):

    return fit_predict(data, forecast_period)[0]



def predict_all(dat, n_components=None):

    pool = Pool()

    f = pool.map(fit_predict_pool, [dat.loc[i, :] for i in dat.index])

    fcast = pd.DataFrame(index=dat.index, columns=f[0].index)

    for i in range(len(dat.index)):

        fcast.iloc[i, :] = f[i]

    return fcast
train = pd.read_csv('../input/covid19-global-forecasting-week-4/train.csv', parse_dates=['Date'])

test = pd.read_csv('../input/covid19-global-forecasting-week-4/test.csv', parse_dates=['Date'])

submission = pd.read_csv('../input/covid19-global-forecasting-week-4/submission.csv')
train['key'] = train['Country_Region'].astype('str') + " " + train['Province_State'].astype('str')

test['key'] = test['Country_Region'].astype('str') + " " + test['Province_State'].astype('str')
test
len(set(train.key)), len(set(test.key))
submission
fatalities = train.pivot('key', 'Date', 'Fatalities')

cases = train.pivot('key', 'Date', 'ConfirmedCases')

cases
cases.sum().plot(label='Confirmed cases', legend=True, logy=True)

fatalities.sum().plot(label='Fatalities', legend=True, title='COVID19 Global Confirmed Cases and Fatalities (log scale)');
new_cases = cases.diff(axis=1).dropna(axis=1)

new_fatalities = fatalities.diff(axis=1).dropna(axis=1)

new_cases
f, m = fit_predict(new_cases.sum())

new_cases.sum()[-43:].plot(title='Aggregated global new cases model')

f[:29].plot()

m.summary()
f, m = fit_predict(new_fatalities.sum())

new_fatalities.sum()[-43:].plot(title='Aggregated global new fatalities model')

f[:29].plot()

m.summary()
forecast_new_cases = predict_all(new_cases)

new_cases.sum()[-43:].plot()

forecast_new_cases.sum()[:29].plot()
forecast_new_cases.sum()[:29]/1000
forecast_new_fatalities = predict_all(new_fatalities)

new_fatalities.sum()[-43:].plot()

forecast_new_fatalities.sum()[:29].plot()
forecast_new_fatalities.sum()[:29]/1000
forecast_cases = cases.iloc[:, -1].values[:, None] + forecast_new_cases.cumsum(axis=1)

forecast_cases
forecast_fatalities = fatalities.iloc[:, -1].values[:, None] + forecast_new_fatalities.cumsum(axis=1)

forecast_fatalities
forecast_cases.iloc[:, :29].sum().plot(title='Cumulative Global Confirmed Cases (millions)')

cases.sum()[-43:].plot();
forecast_fatalities.iloc[:, :29].sum().plot(title='Cumulative Global Fatalities')

fatalities.sum()[-43:].plot();
(forecast_fatalities.iloc[:, :29].sum() / forecast_cases.iloc[:, :28].sum()).plot(title='Global fatalities as proportion of confirmed cases')

(fatalities.sum() / cases.sum())[-43:].plot();
cases_melt = forecast_cases.reset_index().melt('key', var_name='Date', value_name='ConfirmedCases')

fatalities_melt = forecast_fatalities.reset_index().melt('key', var_name='Date', value_name='Fatalities')
test = test.merge(cases_melt, how='left', on=['key', 'Date'])

test = test.merge(fatalities_melt, how='left', on=['key', 'Date'])

test
us_cases = test[test.Country_Region == 'US'].pivot('Province_State', 'Date', 'ConfirmedCases').dropna(axis=1)

us_cases
train[train['Country_Region'] == 'US'].pivot('Province_State', 'Date', 'ConfirmedCases').dropna(axis=1).sum()[-14:].plot()

us_cases.sum()[:29].plot(title='United States Cumulative Confirmed Cases');
test[test.Country_Region == 'US'].pivot('Province_State', 'Date', 'ConfirmedCases').dropna(axis=1).sum()[:29] / 1000
us_fatalities = test[test.Country_Region == 'US'].pivot('Province_State', 'Date', 'Fatalities').dropna(axis=1)

us_fatalities
train[train['Country_Region'] == 'US'].pivot('Province_State', 'Date', 'Fatalities').dropna(axis=1).sum()[-43:].plot()

us_fatalities.sum()[:29].plot(title='United State Cumulative Fatalities');
test[test.Country_Region == 'US'].pivot('Province_State', 'Date', 'Fatalities').dropna(axis=1).sum()[:29] / 1000
(train[train['Country_Region'] == 'US'].pivot('Province_State', 'Date', 'Fatalities').dropna(axis=1).sum()

 / train[train['Country_Region'] == 'US'].pivot('Province_State', 'Date', 'ConfirmedCases').dropna(axis=1).sum())[-43:].plot()



(us_fatalities.sum() / us_cases.sum())[:29].plot(title='US Fatalities as proportion of confirmed cases');
test
submission.ConfirmedCases = test.ConfirmedCases.fillna(0)

submission.Fatalities = test.Fatalities.fillna(0)

submission
submission.to_csv('submission.csv', index=False)
submission.tail(10)