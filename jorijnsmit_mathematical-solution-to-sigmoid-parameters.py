import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from tqdm.notebook import tqdm

from scipy.optimize import curve_fit

from sklearn.metrics import r2_score

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import PolynomialFeatures

from sklearn import linear_model
%matplotlib inline

dpi = 96

plt.rcParams['figure.figsize'] = (1600/dpi, 600/dpi)

plt.style.use('ggplot')
# grabbing prepared dataset from https://www.kaggle.com/jorijnsmit/population-and-sub-continent-for-every-entity

covid = pd.read_csv('/kaggle/input/population-and-sub-continent-for-every-entity/covid.csv', parse_dates=['date'])
# perform same manipulations from the prepared dataset to the test set

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/test.csv', parse_dates=['Date'])

test.columns = ['id', 'province_state', 'country_region', 'date']

test['country_region'].update(test['country_region'].str.replace('Georgia', 'Sakartvelo'))

test['entity'] = test['province_state'].where(~test['province_state'].isna(), test['country_region'])

test = test.set_index('id')[['date', 'entity']]
def logistic(t, k, r, a):

    """k > 0: final epidemic size

    r > 0: infection rate

    a = (k - c_0) / c_0

    """

    

    return k / (1 + a * np.exp(-r * t))
def solve(c):

    """port from https://mathworks.com/matlabcentral/fileexchange/74411-fitvirus"""

    

    n = len(c)

    nmax = max(1, n // 2)



    for i in np.arange(1, nmax+1):

        k1 = i

        k3 = n - 1

        if (n - i) % 2 == 0:

            k3 -= 1



        k2 = (k1 + k3) // 2

        m = k2 - k1 - 1



        if k1 < 1 or k2 < 1 or k3 < 1 or m < 1:

            return None



        k1 -= 1

        k2 -= 1

        k3 -= 1



        # calculate k

        v = c[k1] * c[k2] - 2 * c[k1] * c[k3] + c[k2] * c[k3]

        if v <= 0:

            continue

        w = c[k2]**2 - c[k3] * c[k1]

        if w <= 0:

            continue

        k = c[k2] * v / w

        if k <= 0:

            continue



        # calculate r

        x = c[k3] * (c[k2] - c[k1])

        if x <= 0:

            continue

        y = c[k1] * (c[k3] - c[k2])

        if y <= 0:

            continue

        r = (1 / m) * np.log(x / y)

        if r <= 0:

            continue



        # calculate a

        z = ((c[k3] - c[k2]) * (c[k2] - c[k1])) / w

        if z <= 0:

            continue

        a = z * (x / y) ** ((k3 + 1 - m) / m)

        if a <= 0:

            continue

        

        return k, r, a
def plot_fit(x_train, y_train, x_predict, y_predict, target, r2):

    fig, ax = plt.subplots()

    ax.set_title(f'{subject} ({target}) {r2}')

    color = 'green' if r2 > 0.99 else 'red'

    pd.Series(y_train, x_train).plot(subplots=True, style='.', color='black', legend=True, label='train')

    pd.Series(y_predict, x_predict).plot(subplots=True, style=':', color=color, legend=True, label='predict')

    plt.show()
herd_immunity = 0.7

test_ratio = 0.2



for target in ['confirmed', 'fatal']:

    for subject in tqdm(covid['entity'].unique()):

        population = covid[covid['entity'] == subject]['population'].max()



        x_train = covid[covid['entity'] == subject]['date'].dt.dayofyear.values

        y_train = covid[covid['entity'] == subject][target].values



        mask = y_train > 0

        x_train_m = x_train[mask]

        y_train_m = y_train[mask]

        

        # no point in modelling a single point or no ints at all

        if x_train_m.size < 2 or x_train_m.sum() == 0:

            continue



        x_predict = test[test['entity'] == subject]['date'].dt.dayofyear.values

        submission_size = x_predict.size

        # start calculating sigmoid at same point x_train_m starts

        x_predict = np.arange(start=x_train_m[0], stop=x_predict[-1]+1)



        params = solve(y_train_m)



        if params != None:

            params = (max(params[0], max(y_train_m)), params[1], params[2])

            lower_bounds = (max(y_train_m), 0, 0)

            upper_bounds = (max(population * herd_immunity * test_ratio, params[0]), np.inf, np.inf)



            try:

                params, _ = curve_fit(

                    logistic,

                    np.arange(x_train_m.size),

                    y_train_m,

                    p0=params,

                    bounds=(lower_bounds, upper_bounds)

#                    maxfev=100000

                )

            except:

                print(subject, params, lower_bounds, upper_bounds)



            y_eval = logistic(np.arange(x_train_m.size), params[0], params[1], params[2])

            y_predict = logistic(np.arange(x_predict.size), params[0], params[1], params[2])



            r2 = r2_score(y_train_m, y_eval)

            covid.loc[covid['entity'] == subject, f'log_{target}'] = r2

        else:

            # a couple of countries remain which have too low numbers to fit the sigmoid

            # simple regression with `np.maximum.accumulate` does the trick

            model = Pipeline([

                ("polynomial_features", PolynomialFeatures(degree=2)), 

                ("linear_regression", linear_model.Ridge())

            ])

            model.fit(np.arange(x_train_m.size).reshape(-1, 1), y_train_m)



            y_eval = model.predict(np.arange(x_train_m.size).reshape(-1, 1))

            y_predict = model.predict(np.arange(x_predict.size).reshape(-1 ,1))

            y_predict = np.maximum.accumulate(y_predict).astype('int')

            

            r2 = r2_score(y_train_m, y_eval)

            covid.loc[covid['entity'] == subject, f'poly_{target}'] = r2

            

        if subject in ['Hubei', 'Italy', 'New York', 'Nepal']:

            plot_fit(x_train, y_train, x_predict, y_predict, target, r2)

            

        # assign the prediction to the test dataframe

        delta = submission_size - y_predict.size

        if delta > 0:

            filler = [100] * delta if target == 'confirmed' else [1] * delta

            y_predict = filler + y_predict.tolist()

        test.loc[test['entity'] == subject, target] = y_predict[-submission_size:]
nonlogs = {}

# resulting R2 scores for logistic approach

for target in ['confirmed', 'fatal']:

    r2s = covid.groupby('entity')[f'log_{target}'].max()

    print(r2s.describe())

    nonlogs[target] = r2s[r2s.isna()].index.values

    print(nonlogs[target])
# TODO: some weird ones in here

test.groupby('entity')['fatal'].max().sort_values(ascending=False).head(10)
# sanity check before submitting

submission = test[['entity', 'date']].copy()

submission[['confirmed', 'fatal']] = test[['confirmed', 'fatal']].fillna(0).astype('int')

submission[submission['entity'] == 'Netherlands']
submission = submission[['confirmed', 'fatal']]

submission.index.name = 'ForecastId'

submission.columns = ['ConfirmedCases', 'Fatalities']

submission.to_csv('submission.csv')