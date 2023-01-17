import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn.linear_model import LinearRegression

from tqdm import tqdm_notebook as tqdm



from sklearn.metrics import mean_squared_log_error
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv')

ss = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/submission.csv')
train[['Country/Region', 'Province/State']] = train[['Country/Region', 'Province/State']].fillna('None')

test[['Country/Region', 'Province/State']] = test[['Country/Region', 'Province/State']].fillna('None')
train.head()
train.shape
test.head()
ss.head()
print(ss.shape)
train['Country/Region'].value_counts(dropna=False)
train['Province/State'].value_counts(dropna=False)
train['Date'].max(), test['Date'].min()
valid = train[train['Date'] >= test['Date'].min()]

train = train[train['Date'] < test['Date'].min()]
log_target = True

plot = False



test['ConfirmedCases'] = np.nan

test['Fatalities'] = np.nan



countries = train['Country/Region'].unique()

test_countries = test['Country/Region'].unique()



predictions = []

for c in tqdm(countries):

    train_df = train[train['Country/Region'] == c]

    provinces = train_df['Province/State'].unique()

    

    if c in test_countries:

        test_df = test[test['Country/Region'] == c]

        test_provinces = test_df['Province/State'].unique()

    

        for p in provinces:

            train_df_p = train_df[train_df['Province/State'] == p]

            test_df_p = test_df[test_df['Province/State'] == p]

            

            confirmed = train_df_p['ConfirmedCases'].values[-10:]

            fatalities = train_df_p['Fatalities'].values[-10:]



            if log_target:

                confirmed = np.log1p(confirmed)

                fatalities = np.log1p(fatalities)



            if np.sum(confirmed) > 0:            

                x = np.arange(len(confirmed)).reshape(-1, 1)

                x_test = len(confirmed) + np.arange(len(test_df_p)).reshape(-1, 1)

                

                model = LinearRegression()

                model.fit(x, confirmed)

                p_conf = model.predict(x_test)

                p_conf = np.clip(p_conf, 0, None)

                p_conf = p_conf - np.min(p_conf) + confirmed[-1]

                if log_target:

                    p_conf = np.expm1(p_conf)

                test.loc[(test['Country/Region'] == c) & (test['Province/State'] == p), 'ConfirmedCases'] = p_conf

                

                model = LinearRegression()

                model.fit(x, fatalities)

                p_fatal = model.predict(x_test)

                p_fatal = np.clip(p_fatal, 0, None)

                p_fatal = p_fatal - np.min(p_fatal) + fatalities[-1]

                if log_target:

                    p_fatal = np.expm1(p_fatal)

                test.loc[(test['Country/Region'] == c) & (test['Province/State'] == p), 'Fatalities'] = p_fatal

                

                if plot:

                    plt.figure();

                    plt.plot(x, confirmed);

                    plt.plot(x, fatalities);

                    plt.plot(x_test, p_conf);

                    plt.plot(x_test, p_fatal);

                    plt.title(c + ', ' + p);

            

test[['ConfirmedCases', 'Fatalities']] = test[['ConfirmedCases', 'Fatalities']].fillna(0)
valid.sort_values(['Country/Region', 'Province/State', 'Date'], inplace=True)

preds = test.sort_values(['Country/Region', 'Province/State', 'Date'])

preds = valid[['Country/Region', 'Province/State', 'Date']].merge(preds, on=['Country/Region', 'Province/State', 'Date'], how='left')



score_c = np.sqrt(mean_squared_log_error(valid['ConfirmedCases'].values, preds['ConfirmedCases']))

score_f = np.sqrt(mean_squared_log_error(valid['Fatalities'].values, preds['Fatalities']))



print(f'score_c: {score_c}, score_f: {score_f}, mean: {np.mean([score_c, score_f])}')
pd.concat([valid.reset_index().drop('index', axis=1), 

           preds.reset_index()[['ConfirmedCases', 'Fatalities']].rename({'ConfirmedCases': 'ConfirmedCases_p', 'Fatalities': 'Fatalities_p'}, axis=1)], axis=1)
valid.shape, preds.shape
plt.figure(figsize=(12, 8))

plt.plot([0, 70000], [0, 70000], 'black')

plt.plot(preds['ConfirmedCases'], valid['ConfirmedCases'], '.')

plt.xlabel('Predicted')

plt.ylabel('True')

plt.grid()



plt.figure(figsize=(12, 8))

plt.plot([0, 3500], [0, 3500], 'black')

plt.plot(preds['Fatalities'], valid['Fatalities'], 'r.')

plt.xlabel('Predicted')

plt.ylabel('True')

plt.grid()
submission = test[['ForecastId', 'ConfirmedCases', 'Fatalities']]

submission.to_csv('submission.csv', index=False)

print(submission.shape)
submission.head()