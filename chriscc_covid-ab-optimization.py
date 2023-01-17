#Libraried

import pandas as pd

pd.set_option('display.max_columns', None)

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")



import datetime

from time import time

from scipy import stats



from sklearn.model_selection import GroupKFold

import lightgbm as lgb

import xgboost as xgb

from catboost import CatBoostRegressor, CatBoostClassifier

from sklearn.metrics import mean_squared_error

import plotly.express as px

import plotly.graph_objects as go

import plotly.figure_factory as ff

from plotly.subplots import make_subplots

import os

import glob

import copy



import numpy as np

from scipy.integrate import odeint
# SI model

N = 2200000          # Total population

I = np.zeros(200)  # Infected

S = np.zeros(200)   # Susceptible



r = 10             # This value defines how quickly the disease spreads

B = 0.01            # Probability of being infected



I[0] = 1           # On day 0, there's only one infected person

S[0] = N-I[0]      # So the suspecptible people is equal = N - I[0]



for idx in range(199):

    S[idx+1] = S[idx] - r*B*I[idx]*S[idx]/N

    I[idx+1] = I[idx] + r*B*I[idx]*S[idx]/N

sns.lineplot(x=np.arange(200), y=S, label='Susceptible')

sns.lineplot(x=np.arange(200), y=I, label='Infected')
N = 2200000        # Total population

days = 200          # Period

E = np.zeros(days)  # Exposed          

E[0] = 0            # Day 0 exposed

I = np.zeros(days)  # Infected

I[0] = 1          # Day 0 infected                                                                

S = np.zeros(days)  # Susceptible

S[0] = N - I[0]     # Day 0 susceptible

R = np.zeros(days)  # Recovered

R[0] = 0



r = 20              # Number of susceptible could be contactes by an infected

B = 0.03            # Probability of spread for infected

a = 0.1             # Probability of converted from exposed to infected

r2 = r             # Number of susceptible could be contactes by an exposed

B2 = B          # Probability of spread for exposed

y = 0.1             # Probability of recovered





for idx in range(days-1):

    S[idx+1] = S[idx] - r*B*S[idx]*I[idx]/N - r2*B2*S[idx]*E[idx]/N

    E[idx+1] = E[idx] + r*B*S[idx]*I[idx]/N -a*E[idx] + r2*B2*S[idx]*E[idx]/N

    I[idx+1] = I[idx] + a*E[idx] - y*I[idx]

    R[idx+1] = R[idx] + y*I[idx]

    

plt.figure(figsize=(16,9))

sns.lineplot(x=np.arange(200), y=S, label='Susceptible')

sns.lineplot(x=np.arange(200), y=I, label='Infected')

sns.lineplot(x=np.arange(200), y=E, label='Exposed')

sns.lineplot(x=np.arange(200), y=R, label='Recovered')







I_origin = copy.copy(I)
N = 2200000        # Total population

days = 200          # Period

E = np.zeros(days)  # Exposed          

E[0] = 0            # Day 0 exposed

I = np.zeros(days)  # Infected

I[0] = 1            # Day 0 infected                                                                

S = np.zeros(days)  # Susceptible

S[0] = N - I[0]     # Day 0 susceptible

R = np.zeros(days)  # Recovered

R[0] = 0



r = 20              # Number of susceptible could be contactes by an infected

B = 0.03            # Probability of spread for infected

a = 0.1             # Probability of converted from exposed to infected

r2 = r             # Number of susceptible could be contactes by an exposed

B2 = B           # Probability of spread for exposed

y = 0.1             # Probability of recovered





for idx in range(days-1):

    if idx>10:

        r = 5

        r2 = r

    S[idx+1] = S[idx] - r*B*S[idx]*I[idx]/N - r2*B2*S[idx]*E[idx]/N

    E[idx+1] = E[idx] + r*B*S[idx]*I[idx]/N -a*E[idx] + r2*B2*S[idx]*E[idx]/N

    I[idx+1] = I[idx] + a*E[idx] - y*I[idx]

    R[idx+1] = R[idx] + y*I[idx]



plt.figure(figsize=(16,9))

sns.lineplot(x=np.arange(200), y=S, label='Secestible')

sns.lineplot(x=np.arange(200), y=I, label='Infected')

sns.lineplot(x=np.arange(200), y=E, label='Exposed')

sns.lineplot(x=np.arange(200), y=R, label='Recovered')



I_sd = copy.copy(I)
plt.figure(figsize=(16,9))

sns.lineplot(x=np.arange(200), y=I_origin, label='Infected w/o social distancing')

sns.lineplot(x=np.arange(200), y=I_sd, label='Infected w/ social distancing')
## Construct an dataframe for the data

init_date = '2020-03-06'  # Initial date

covid_df = pd.DataFrame({'Susceptible-pred':I, 

                         'Exposed-pred':E, 

                         'Infected-pred':I, 

                         'Recovered-pred':R,

                         'Date':pd.date_range(start=init_date, periods=len(I)) ## Setup report date: start date = init date and incresed by 1 for len(I) times

                        })

covid_df
plt.figure(figsize=(16,9))

sns.lineplot(x='Date', y='Susceptible-pred', data=covid_df, label='Susceptible-pred')

sns.lineplot(x='Date', y='Exposed-pred', data=covid_df, label='Exposed-pred')

sns.lineplot(x='Date', y='Infected-pred', data=covid_df, label='Infected-pred')

sns.lineplot(x='Date', y='Recovered-pred', data=covid_df, label='Recovered-pred')
actual_df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv',parse_dates=['Date'])

actual_df.head()
actual_df = actual_df[(actual_df['Province_State']=='Alberta') & (actual_df['Date']>=init_date)]

actual_df
covid_df = pd.merge(covid_df, actual_df, how='left', on='Date')
## train_df is the dataset that will be used to "train" the model.

train_df = covid_df[covid_df['ConfirmedCases'].notnull()]

train_df
plt.figure(figsize=(16,9))

sns.lineplot(x='Date', y='Infected-pred', data=train_df, label='Predictions')

sns.lineplot(x='Date', y='ConfirmedCases', data=train_df, label='Actuals')
def RMSLE(actuals, predictions):

    return np.sqrt(mean_squared_error(np.log1p(actuals), np.log1p(predictions)))



print("Baseline model's RMSLE is %f" % (RMSLE(train_df['ConfirmedCases'], train_df['Infected-pred'])))
def SEIR(N, days, init_date, init_exposed, init_infected, r, B, a, y, r_sd, sd_days):

    '''

    Define the SEIR model:

    Parameters:

    N: total population

    init_date: intital data of the model

    init_exposed: initial number of exposed

    init_infected: initial number of infected

    r: Number of susceptible could be contactes by an infected

    B: Probability of spread for infected

    a: Probability of converted from exposed to infected

    y: Probability of recovered

    r_sd: Number of susceptible could be contactes by an infected with social distancing

    sd_days: number of days that social distancing was implemented after the initial date



    

    '''

    E = np.zeros(days)  # Exposed          

    E[0] = init_exposed            # Day 0 exposed

    I = np.zeros(days)  # Infected

    I[0] = init_infected            # Day 0 infected                                                                

    S = np.zeros(days)  # Susceptible

    S[0] = N - I[0]     # Day 0 susceptible

    R = np.zeros(days)  # Recovered

    R[0] = 0



    r2 = r             # Number of susceptible could be contactes by an exposed

    B2 = B           # Probability of spread for exposed



    for idx in range(days-1):

        if idx>sd_days:

            r = r_sd

            r2 = r_sd

        S[idx+1] = S[idx] - r*B*S[idx]*I[idx]/N - r2*B2*S[idx]*E[idx]/N

        E[idx+1] = E[idx] + r*B*S[idx]*I[idx]/N -a*E[idx] + r2*B2*S[idx]*E[idx]/N

        I[idx+1] = I[idx] + a*E[idx] - y*I[idx]

        R[idx+1] = R[idx] + y*I[idx]



    df = pd.DataFrame({'Susceptible-pred':I, 

                         'Exposed-pred':E, 

                         'Infected-pred':I, 

                         'Recovered-pred':R,

                         'Date':pd.date_range(start=init_date, periods=len(I))

                        })

    return df





pred_df = SEIR(N=220000, days=200, init_date="2020-03-06", init_exposed=0, 

     init_infected=1, r=20, B=0.03, a=0.1, y=0.1, r_sd=5, sd_days=10)   



pred_df


def eval_model(N, days, init_date, init_exposed, init_infected, r, B, a, y, r_sd, sd_days, actual_df):

    pred_df = SEIR(N=N, days=days, init_date=init_date, init_exposed=init_exposed, 

     init_infected=init_infected, r=r, B=B, a=a, y=y, r_sd=r_sd, sd_days=sd_days) 

#     print(pred_df)

    pred_df = pd.merge(pred_df, actual_df, how='left', on='Date')

    pred_df = pred_df[pred_df['ConfirmedCases'].notnull()]

#     print(pred_df['ConfirmedCases'], pred_df['Infected-pred'])

    return RMSLE(pred_df['ConfirmedCases'], pred_df['Infected-pred'])

    

eval_model(N=220000, days=200, init_date="2020-03-06", init_exposed=0, 

     init_infected=1, r=20, B=0.03, a=0.1, y=0.1, r_sd=5, sd_days=10, actual_df=actual_df)
eval_model(N=220000, days=200, init_date="2020-03-06", init_exposed=10, 

     init_infected=1, r=20, B=0.03, a=0.1, y=0.1, r_sd=5, sd_days=10, actual_df=actual_df)
scores = []

i = 0 

for init_exposed in np.arange(0, 50, 5):

    for r in np.arange(15, 25, 2):

        for B in np.arange(0.01, 0.08, 0.01):

            for a in np.arange(0.05, 0.2, 0.05):

                for r_sd_ratio in np.arange(0.1, 0.5, 0.1):

                    r_sd = r*r_sd_ratio

                    for y in np.arange(0.05,0.2,0.05):

                        score = eval_model(N=220000, days=200, init_date="2020-03-06", 

                                          init_exposed=init_exposed, init_infected=1, r=r, B=B, a=a, y=y, r_sd=r_sd, sd_days=10, 

                                          actual_df=actual_df)

                        i = i + 1

                        if i%500: # Print out the process for every 500 runs

                            print("Score: %f, inti_exposed: %f, r:%f, B:%f, a:%f, y:%f, r_sd:%f" % (score, init_exposed, r, B, a, y, r_sd))

                        scores.append([score, init_exposed, r, B, a, y, r_sd]) # append validation score as well as parameters used to scores
scores_pd = pd.DataFrame(scores, columns=['score', 'init_exposed', 'r', 'B', 'a', 'y', 'r_sd']).sort_values('score')

scores_pd.head()
pred_df = SEIR(N=2200000, days=200, init_date="2020-03-06", 

     init_exposed=scores_pd['init_exposed'].values[0], 

     init_infected=1, 

     r=scores_pd['r'].values[0], 

     B=scores_pd['B'].values[0], 

     a=scores_pd['a'].values[0], 

     y=scores_pd['y'].values[0], 

     r_sd=scores_pd['r_sd'].values[0], 

     sd_days=10) 





pred_df = pd.merge(pred_df, actual_df, how='left', on='Date')

pred_df = pred_df[pred_df['ConfirmedCases'].notnull()]



print("Baseline model's RMSLE is %f" % (RMSLE(pred_df['ConfirmedCases'], pred_df['Infected-pred'])))



plt.figure(figsize=(16,9))

sns.lineplot(x='Date', y='Infected-pred', data=pred_df, label='Predictions')

sns.lineplot(x='Date', y='ConfirmedCases', data=pred_df, label='Actuals')





pred_df