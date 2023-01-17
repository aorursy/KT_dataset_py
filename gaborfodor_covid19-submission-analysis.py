%matplotlib inline

import pandas as pd

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = 'all'

pd.set_option('display.max_columns', 99)

pd.set_option('display.max_rows', 99)

import os

import numpy as np

from tqdm import tqdm

import datetime as dt
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [16, 10]

plt.rcParams['font.size'] = 14

import seaborn as sns

sns.set_palette(sns.color_palette('tab20', 20))



import plotly.express as px

import plotly.graph_objects as go
def to_log(x):

    return np.log(x + 1)





def to_exp(x):

    return np.exp(x) - 1



start = dt.datetime.now()



lb_periods = {

    1: ('2020-03-26', '2020-04-23'),

    2: ('2020-04-02', '2020-04-30'),

    3: ('2020-04-09', '2020-05-07'),

    4: ('2020-04-16', '2020-05-14')

}
def get_competition_data(week):

    train = pd.read_csv(f'../input/covid19-global-forecasting-week-{week}/train.csv')

    test = pd.read_csv(f'../input/covid19-global-forecasting-week-{week}/test.csv')

    

    if 'Province/State' in test.columns:

        test = test.rename(columns={'Province/State': 'Province_State', 'Country/Region': 'Country_Region'})

        train = train.rename(columns={'Province/State': 'Province_State', 'Country/Region': 'Country_Region'})

    train['Location'] = train['Country_Region'] + '-' + train['Province_State'].fillna('')

    test['Location'] = test['Country_Region'] + '-' + test['Province_State'].fillna('')

    train = train[['Date', 'Location', 'ConfirmedCases', 'Fatalities']]

    return train, test
def get_actual(skip_nova_scotia=False):

    #     actual, _ = get_competition_data(week=4)

    actual = pd.read_csv(f'../input/covid-19-forecasting-ongoing-data-updates/train.csv')

    actual.ConfirmedCases = actual.ConfirmedCases.clip(0, None)

    actual.Fatalities = actual.Fatalities.clip(0, None)

    actual['Location'] = actual['Country_Region'] + '-' + actual['Province_State'].fillna('')

    actual = actual[['Date', 'Location', 'ConfirmedCases', 'Fatalities']]

    if skip_nova_scotia:

        actual = actual[actual.Location != 'Canada-Nova Scotia']

    return actual
actual = get_actual(skip_nova_scotia=False)

actual[actual.Location=='Canada-Nova Scotia'].tail(10)

actual[actual.Location=='US-New York'].tail(40)
def get_submissions(week):

    submission_path = f'../input/covid19-global-forecasting-submissions/week_{week}'

    submission_files = os.listdir(submission_path)

    submissions_list = []



    for f in tqdm(submission_files):

        submission = pd.read_csv(os.path.join(submission_path, f))

        submission.insert(0, 'SubmissionId', int(f[:-4]))

        submissions_list.append(submission)



    submissions = pd.concat(submissions_list, ignore_index=True, sort=False)

    

    submissions = submissions[['SubmissionId', 'ForecastId', 'ConfirmedCases', 'Fatalities']]

    

    submissions.ConfirmedCases = submissions.ConfirmedCases.clip(0, None)

    submissions.Fatalities = submissions.Fatalities.clip(0, None)

    

    _, test = get_competition_data(week)

    submissions = submissions.merge(test, on='ForecastId', how='left')

    

    submissions = submissions.loc[submissions.Date >= lb_periods[week][0]]

    

    actual = get_actual()

    submissions = submissions.merge(actual, how='left', on=['Date', 'Location'], suffixes=['', 'Actual'])

    

    return submissions
# actual, _ = get_competition_data(week=4)

actual = get_actual()

print(f'Actual last day: {actual.Date.max()}')

actual.describe()

week = 1

submissions = get_submissions(week)





submissions.head()

submissions.shape
def add_errors(submissions):

    submissions.loc[:,'FatalitiesSLE'] = (to_log(submissions.Fatalities) - to_log(submissions.FatalitiesActual)) ** 2

    submissions.loc[:,'ConfirmedCasesSLE'] = (to_log(submissions.ConfirmedCases) - to_log(submissions.ConfirmedCasesActual)) ** 2

    return submissions



def calculate_lb(submissions):

    lb = submissions[['SubmissionId', 'FatalitiesSLE', 'ConfirmedCasesSLE']].groupby('SubmissionId').mean().reset_index()

    lb.loc[:, 'FatalatiesRMSLE'] = np.sqrt(lb['FatalitiesSLE'])

    lb.loc[:, 'ConfirmedCasesRMSLE'] = np.sqrt(lb['ConfirmedCasesSLE'])

    lb.loc[:, 'RMSLE'] = (lb['FatalatiesRMSLE'] + lb['ConfirmedCasesRMSLE']) / 2.0

    lb = lb.sort_values(by='RMSLE')

    lb['Rank'] = np.arange(len(lb))

    return lb
submissions = add_errors(submissions)



lb = calculate_lb(submissions)

submissions = submissions.merge(lb[['SubmissionId', 'RMSLE', 'Rank']], on='SubmissionId')

submissions.head()

lb.head()
def get_ensemble(submissions, k=10):

    submissions['LogCC'] = to_log(submissions.ConfirmedCases)

    submissions['LogF'] = to_log(submissions.Fatalities)



    ensemble = submissions[submissions.Rank < k].groupby(['Date', 'Location'])[['LogCC', 'LogF']].mean()

    ensemble['ConfirmedCases'] = to_exp(ensemble.LogCC)

    ensemble['Fatalities'] = to_exp(ensemble.LogF)

    ensemble = ensemble.reset_index()



    ensemble = ensemble.merge(actual, how='left', on=['Date', 'Location'], suffixes=['', 'Actual'])

    ensemble = add_errors(ensemble)

    return ensemble
def calculate_lb_and_ensemble(week, top_ranks=10):

    submissions = get_submissions(week)

    submissions = add_errors(submissions)



    lb = calculate_lb(submissions)

    submissions = submissions.merge(lb[['SubmissionId', 'RMSLE', 'Rank']], on='SubmissionId')



    ens = get_ensemble(submissions, k=10)

    np.sqrt((ens.FatalitiesSLE.mean() + ens.ConfirmedCasesSLE.mean() ) / 2.0)



    daily_error = submissions[submissions.Rank < top_ranks].groupby(['SubmissionId', 'Date']).mean().reset_index()

    daily_error['Daily RMSLE'] = np.sqrt(0.5 * daily_error.FatalitiesSLE + 0.5 * daily_error.ConfirmedCasesSLE)

    daily_error['LB Score'] = '#' + daily_error.Rank.astype(str) + ' - ' + daily_error.RMSLE.round(5).astype(str) + ' - ' + daily_error.SubmissionId.astype(str)

    daily_error = daily_error.sort_values(by=['Rank', 'Date'])

    fig = px.line(daily_error, x='Date', y='Daily RMSLE', color='LB Score')

    _ = fig.update_layout(

        title_text=f'COVID-19 Daily Prediction Error (Week {week})'

    )



    return submissions, lb, ens, daily_error, fig

    
week = 1

submissions, lb, ens1, daily_error, fig = calculate_lb_and_ensemble(week)

lb.head()

fig.show()
week = 2

submissions, lb, ens2, daily_error, fig = calculate_lb_and_ensemble(week)

lb.head()

fig.show()
week = 3

submissions, lb, ens3, daily_error, fig = calculate_lb_and_ensemble(week)

lb.head()

fig.show()
week = 4

submissions, lb, ens4, daily_error, fig = calculate_lb_and_ensemble(week, top_ranks=16)

lb.head()

fig.show()
ens1['Week'] = 1

ens2['Week'] = 2

ens3['Week'] = 3

ens4['Week'] = 4



ens1['Days'] = (pd.to_datetime(ens1.Date) - pd.to_datetime(ens1.Date).min()).dt.days

ens2['Days'] = (pd.to_datetime(ens2.Date) - pd.to_datetime(ens2.Date).min()).dt.days

ens3['Days'] = (pd.to_datetime(ens3.Date) - pd.to_datetime(ens3.Date).min()).dt.days

ens4['Days'] = (pd.to_datetime(ens4.Date) - pd.to_datetime(ens4.Date).min()).dt.days

ensembles = pd.concat([ens1, ens2, ens3, ens4])



daily_error = ensembles.groupby(['Week', 'Date']).mean().reset_index()

daily_error['Daily RMSLE'] = np.sqrt(0.5 * daily_error.FatalitiesSLE + 0.5 * daily_error.ConfirmedCasesSLE)



fig = px.line(daily_error, x='Date', y='Daily RMSLE', color='Week')

_ = fig.update_layout(

    title_text=f'COVID-19 Ensemble Daily Prediction Error'

)

fig.show()
fig = px.line(daily_error, x='Days', y='Daily RMSLE', color='Week')

_ = fig.update_layout(

    title_text=f'COVID-19 Ensemble Daily Prediction Error',

    width = 1600,

    height = 800,

)

fig.show()
'Difficult Locations'

ensembles.groupby(['Week', 'Location']).mean().reset_index().sort_values(by='ConfirmedCasesSLE', ascending=False).head(10)

'Easiest Locations'

ensembles.groupby(['Week', 'Location']).mean().reset_index().sort_values(by='ConfirmedCasesSLE', ascending=False).dropna().tail(10)
end = dt.datetime.now()

print('Finished', end, (end - start).seconds, 's')