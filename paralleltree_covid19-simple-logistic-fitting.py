import sys

import numpy as np

import pandas as pd

from joblib import Parallel, delayed

import matplotlib.pyplot as plt



from sklearn.model_selection import GridSearchCV

from scipy.optimize import curve_fit
df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv', parse_dates=['Date'])

df_train = df_train.replace(np.nan, '', regex=True) # replace nan in Province_State with empty string

agg_unit = ['Country_Region', 'Province_State'] # an argument of the groupby method



df_pops = pd.read_csv('/kaggle/input/population-by-country-2020/population_by_country_2020.csv')

pops = dict(zip(df_pops['Country (or dependency)'], df_pops['Population (2020)']))

pops['US'] = pops['United States']

pops['Korea, South'] = pops['South Korea']

pops['Congo (Brazzaville)'] = pops['DR Congo']

pops['Congo (Kinshasa)'] = pops['DR Congo']

pops['Taiwan*'] = pops['Taiwan']

pops['Saint Vincent and the Grenadines'] = pops['St. Vincent & Grenadines']

pops['Saint Kitts and Nevis'] = pops['Saint Kitts & Nevis']

pops['Czechia'] = pops['Czech Republic (Czechia)']

pops['Cote d\'Ivoire'] = pops['Côte d\'Ivoire']

pops['Diamond Princess'] = 4000



unmapped_countries = list(df_train['Country_Region'].unique()) - pops.keys()

if len(unmapped_countries) > 0:

    print('warn: There are unmapped population data: ', end='', file=sys.stderr)

    print(*unmapped_countries, sep=', ', file=sys.stderr)
# fix non-cumulative records

df_train[['ConfirmedCases', 'Fatalities']] = df_train.groupby(agg_unit)[['ConfirmedCases', 'Fatalities']].transform('cummax')
# calculating moving average

df_train[['ConfirmedCasesMean', 'FatalitiesMean']] = df_train.groupby(agg_unit)[['ConfirmedCases', 'Fatalities']].transform(lambda x: x.rolling(5, center=True).mean())
# set the first confirmed date to each rows

df_train['FirstConfirmedDateCountry'] = df_train.query('ConfirmedCases>0').groupby(agg_unit)['Date'].transform('min')

df_train['FirstDeceasedDateCountry'] = df_train.query('Fatalities>0').groupby(agg_unit)['Date'].transform('min')
df_train['DaysSinceFirstConfirmed'] = (df_train['Date'] - df_train['FirstConfirmedDateCountry']).dt.days

df_train['DaysSinceFirstDeceased'] = (df_train['Date'] - df_train['FirstDeceasedDateCountry']).dt.days
# preparing an output dataframe

df_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv', parse_dates=['Date'])

df_test = df_test.replace(np.nan, '', regex=True)

df_test['ConfirmedCases'] = 0

df_test['Fatalities'] = 0
class LogisticCurve:

    def __init__(self, k_init=1, c_init=1,

                 k_lower=-np.inf, k_upper=np.inf,

                 c_lower=-np.inf, c_upper=np.inf):

        self.k_init = k_init

        self.c_init = c_init

        self.k_lower, self.k_upper = k_lower, k_upper

        self.c_lower, self.c_upper = c_lower, c_upper

        self.k = k_init

        self.b = c_init

        self.c = c_init





    def logistic_func(self, x, k, b, c):

        return c / (b * np.exp(-k * x) + 1)



    def fit(self, xs, ys):

        from scipy.optimize import curve_fit

        p0 = [self.k_init, self.c_init, self.c_init]

        bounds = [(self.k_lower, self.c_lower, self.c_lower), (self.k_upper, self.c_upper, self.c_upper)]

        try:

            params, pcov = curve_fit(self.logistic_func, xs, ys, p0=p0, bounds=bounds)

            self.k, self.b, self.c = params

        except RuntimeError:

            pass

        return self



    def predict(self, xs):

        return self.logistic_func(xs, self.k, self.b, self.c)



    def get_params(self, deep=False):

        return { 

            'k_init': self.k_init,

            'c_init': self.c_init,

            'k_lower': self.k_lower,

            'k_upper': self.k_upper,

            'c_lower': self.c_lower,

            'c_upper': self.c_upper

        }

    

    def set_params(self, **params):

        for param, value in params.items():

            setattr(self, param, value)

        return self



    def get_estimated_params(self):

        return [self.k, self.b, self.c]



    def set_curve_params(self, k, b, c):

        self.k = k

        self.b = b

        self.c = c

        return self
def estimate_column_params(df, xcol, ycol, maximum):

    try:

        init_count = df[df[xcol] == 0][ycol].iloc[0]

    except IndexError:

        return None

    

    rows = df[(df[xcol] >= 0) & (df[ycol] >= 0)]

    train_data = rows.query('Date <= "2020-03-26"')

    if len(train_data) == 0:

        return None

    

    cv = [(list(range(len(train_data))), list(range(len(train_data), len(rows))))]

    mcv = GridSearchCV(LogisticCurve(k_lower=1e-18, c_lower=0, c_upper=maximum*0.5), { 'k_init': [0.1, 1], 'k_upper': [1, 10], 'c_init': [0, init_count] }, cv=cv, scoring='neg_root_mean_squared_error')

    mcv.fit(rows[xcol], rows[ycol])

    return mcv.best_estimator_.get_estimated_params()

    

def estimate_params(df, maximum):

    return [estimate_column_params(df, 'DaysSinceFirstConfirmed', 'ConfirmedCasesMean', maximum), estimate_column_params(df, 'DaysSinceFirstDeceased', 'FatalitiesMean', maximum)]
# estimate params with parallelism, then map result to corresponding region.

estimated_params = dict(Parallel(n_jobs=-1, verbose=8)([delayed(lambda label, df, maximum: [label, estimate_params(df, maximum)])((country, state), df, pops[country]) for (country, state), df in df_train.groupby(agg_unit)]))
# make prediction with estimaed params

for (country, state), df in df_train.groupby(agg_unit):

    cparams, fparams = estimated_params[(country, state)]

    

    try:

        first_confirmed_on = df[df['DaysSinceFirstConfirmed'] == 0]['Date'].iloc[0]

        df_test.loc[(df_test['Country_Region'] == country) & (df_test['Province_State'] == state), 'FirstConfirmedDate'] = first_confirmed_on

        df_test['DaysSinceFirstConfirmed'] = (df_test['Date'] - df_test['FirstConfirmedDate']).dt.days

        if cparams is not None:

            mcv1 = LogisticCurve().set_curve_params(*cparams)

            df_train.loc[(df_train['Country_Region'] == country) & (df_train['Province_State'] == state), 'PredictedConfirmedCases'] = mcv1.predict(df_train.loc[(df_train['Country_Region'] == country) & (df_train['Province_State'] == state), 'DaysSinceFirstConfirmed'])

            df_test.loc[(df_test['Country_Region'] == country) & (df_test['Province_State'] == state), 'ConfirmedCases'] = mcv1.predict(df_test.loc[(df_test['Country_Region'] == country) & (df_test['Province_State'] == state), 'DaysSinceFirstConfirmed'])

    except IndexError:

        continue

    

    try:

        first_deceased_on = df[df['DaysSinceFirstDeceased'] == 0]['Date'].iloc[0]

        df_test.loc[(df_test['Country_Region'] == country) & (df_test['Province_State'] == state), 'FirstDeceasedDate'] = first_confirmed_on

        df_test['DaysSinceFirstDeceased'] = (df_test['Date'] - df_test['FirstDeceasedDate']).dt.days

        if fparams is not None:

            mcv2 = LogisticCurve().set_curve_params(*fparams)

            df_train.loc[(df_train['Country_Region'] == country) & (df_train['Province_State'] == state), 'PredictedFatalities'] = mcv2.predict(df_train.loc[(df_train['Country_Region'] == country) & (df_train['Province_State'] == state), 'DaysSinceFirstDeceased'])

            df_test.loc[(df_test['Country_Region'] == country) & (df_test['Province_State'] == state), 'Fatalities'] = mcv2.predict(df_test.loc[(df_test['Country_Region'] == country) & (df_test['Province_State'] == state), 'DaysSinceFirstDeceased'])

    except IndexError:

        continue
# picking up countries to visualize

for (country, state), df in df_train.groupby(agg_unit):

    if country not in ['Italy', 'Spain', 'Japan', 'Russia']:

        continue

    

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))

    fig.suptitle(f'{country} {state}')

    

    c = df.query('DaysSinceFirstConfirmed>=0').set_index('DaysSinceFirstConfirmed').sort_index()

    if len(c) == 0:

        continue

    c['ConfirmedCases'].plot(label='Actual', ax=ax1)

    c['PredictedConfirmedCases'].plot(label='Fitted', ax=ax1)

    ax1.legend()

    

    c = df.query('DaysSinceFirstDeceased>=0').set_index('DaysSinceFirstDeceased').sort_index()

    if len(c) == 0:

        continue

    c['Fatalities'].plot(label='Actual', ax=ax2)

    c['PredictedFatalities'].plot(label='Fitted', ax=ax2)

    ax1.legend()
df_test.to_csv('submission.csv', index=False, columns=['ForecastId', 'ConfirmedCases', 'Fatalities'])