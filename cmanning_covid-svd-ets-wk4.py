# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from statsmodels.tsa.holtwinters import ExponentialSmoothing

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



def fit_predict(data, forecast_period=43):

    model = auto_ets(data)

    fcast = model.forecast(forecast_period)

    return fcast, model



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
train
test
len(set(train.key)), len(set(test.key))
submission
cases = train.pivot('key', 'Date', 'ConfirmedCases')

fatalities = train.pivot('key', 'Date', 'Fatalities')

cases.index += ' cases'

fatalities.index += ' fatal'

combined = pd.concat([cases, fatalities])

combined
cases.sum().plot(label='Confirmed cases', legend=True)

fatalities.sum().plot(label='Fatalities', legend=True, title='COVID19 Global Confirmed Cases and Fatalities (log scale)', logy=True);
f, m = fit_predict(cases.sum())

f[:29].plot(title='Aggregate check for additive vs multiplicative trend on cumulative cases')

cases.sum()[-43:].plot()

m.summary()
f, m = fit_predict(fatalities.sum())

f[:29].plot(title='Aggregate check for additive vs multiplicative trend on cumulative deaths')

fatalities.sum()[-43:].plot()

m.summary()
f, m = fit_predict(cases.diff(axis=1).sum())

f[:29].plot(title='Aggregate check for additive vs multiplicative trend on new cases')

cases.diff(axis=1).sum()[-43:].plot()

m.summary()
f, m = fit_predict(fatalities.diff(axis=1).sum())

f[:29].plot(title='Aggregate check for additive vs multiplicative trend on new deaths')

fatalities.diff(axis=1).sum()[-43:].plot()

m.summary()
diff_combined = combined.diff(axis=1).dropna(axis=1)

svd = TruncatedSVD(100)

svd_factors = pd.DataFrame(svd.fit_transform(diff_combined.T).T, columns=diff_combined.columns)

svd_factors
svd.explained_variance_ratio_[:5].round(3)
svd_factors.T.iloc[:, :5].plot(title='Top five SVD components');
f, m = fit_predict(svd_factors.loc[0, :])

f[:29].plot(title='Projected component 0')

svd_factors.iloc[0, -43:].plot()

m.summary()
f, m = fit_predict(svd_factors.loc[1, :])

f[:29].plot(title='Projected component 1')

svd_factors.iloc[1, -43:].plot()

m.summary()
svd_forecast = predict_all(svd_factors)

forecast_combined_diff = pd.DataFrame(svd.inverse_transform(svd_forecast.T).T, index=combined.index, columns=svd_forecast.columns).clip(0)

forecast_combined = forecast_combined_diff.cumsum(axis=1) + combined.iloc[:, -1].values[:, None]

forecast_combined
forecast_new_cases = forecast_combined_diff.iloc[:len(cases), :]

forecast_new_fatalities = forecast_combined_diff.iloc[len(cases):, :]

forecast_cases = forecast_combined.iloc[:len(cases), :]

forecast_fatalities = forecast_combined.iloc[len(cases):, :]

forecast_new_fatalities
cases.sum()[-43:].plot()

forecast_cases.sum()[:29].plot(title='Global cumulative confirmed cases (millions)');
forecast_new_cases.sum()[:29]/1000
fatalities.sum()[-43:].plot()

forecast_fatalities.sum()[:29].plot(title='Global cumulative fatalities');
forecast_new_fatalities.sum()[:29] / 1000
(forecast_fatalities.sum() / forecast_cases.sum())[:29].plot(title='Global fatalities as proportion of confirmed cases')

(fatalities.sum() / cases.sum())[-43:].plot();
cases_melt = forecast_cases.reset_index().melt('key', var_name='Date', value_name='ConfirmedCases')

fatalities_melt = forecast_fatalities.reset_index().melt('key', var_name='Date', value_name='Fatalities')

cases_melt.key = [key[:-6] for key in cases_melt.key]

fatalities_melt.key = [key[:-6] for key in fatalities_melt.key]

cases_melt
test = test.merge(cases_melt, how='left', on=['key', 'Date'])

test = test.merge(fatalities_melt, how='left', on=['key', 'Date'])

test
us_cases = test[test.Country_Region == 'US'].pivot('Province_State', 'Date', 'ConfirmedCases').dropna(axis=1)

us_cases
train[train['Country_Region'] == 'US'].pivot('Province_State', 'Date', 'ConfirmedCases').dropna(axis=1).sum()[-43:].plot()

us_cases.sum()[:29].plot(title='United States Cumulative Confirmed Cases (millions)');
test[test.Country_Region == 'US'].pivot('Province_State', 'Date', 'ConfirmedCases').dropna(axis=1).sum()[:29] / 1000
us_fatalities = test[test.Country_Region == 'US'].pivot('Province_State', 'Date', 'Fatalities').dropna(axis=1)

us_fatalities
train[train['Country_Region'] == 'US'].pivot('Province_State', 'Date', 'Fatalities').dropna(axis=1).sum()[-43:].plot()

us_fatalities.sum()[:29].plot(title='United State Cumulative Fatalities');
test[test.Country_Region == 'US'].pivot('Province_State', 'Date', 'Fatalities').dropna(axis=1).sum()[:29] / 1000
(train[train['Country_Region'] == 'US'].pivot('Province_State', 'Date', 'Fatalities').dropna(axis=1).sum()

 / train[train['Country_Region'] == 'US'].pivot('Province_State', 'Date', 'ConfirmedCases').dropna(axis=1).sum())[-43:].plot()



(us_fatalities.sum() / us_cases.sum())[:29].plot(title='US fatalities as proportion of confirmed cases');
us_cases.loc['Virginia'].diff().dropna()[:29]
us_fatalities.loc['Virginia'].diff().dropna()[:29]
(us_fatalities.loc['Virginia'] / us_cases.loc['Virginia']).plot()
test.head(30)
test.tail(30)
submission.ConfirmedCases = test.ConfirmedCases.fillna(0)

submission.Fatalities = test.Fatalities.fillna(0)

submission
submission.to_csv('submission.csv', index=False)
submission.tail(10)