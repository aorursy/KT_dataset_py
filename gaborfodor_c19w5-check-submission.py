%matplotlib inline

import pandas as pd

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

pd.set_option('display.max_columns', 99)

pd.set_option('display.max_rows', 99)

import os

import numpy as np

from matplotlib import pyplot as plt

from tqdm import tqdm

import plotly.express as px

from sklearn import metrics

from functools import partial

from geopy import distance

import datetime as dt
VERSION = 10

COMP_DIR = './data/w5/covid19-global-forecasting-week-5 (4)/'

LOG_FILE = f'./data/w5/paropt_lgb_v{VERSION}.csv'

FIMP_FILE = f'./data/w5/fimp_lgb_v{VERSION}.csv'

FEATURE_FILE_PATH = f'./data/w5/features_v{VERSION}.csv'

PREDS_PATH = f'./data/w5/predictions_v{VERSION}/'
EXTERNAL_DATA_PATH = '../input/covid19belugaw5/'



COMP_DIR = '../input/covid19-global-forecasting-week-5/'
subm = pd.read_csv(EXTERNAL_DATA_PATH + 'w5_private_submission_v7.csv')

subm['ForecastId'] = subm.ForecastId_Quantile.map(lambda s: s.split('_')[0]).astype(int)

subm['q'] = subm.ForecastId_Quantile.map(lambda s: s.split('_')[1])

subm.head(2)
train = pd.read_csv(COMP_DIR + 'train.csv')

train = train.fillna('')

train['Location'] = train.Country_Region + '-' + train.Province_State + '-' + train.County

train['DateTime'] = pd.to_datetime(train.Date)

train['TargetQ'] = train.Target + 'Actual'

train = train[['Location', 'DateTime', 'Target', 'TargetQ', 'TargetValue']]

train.head(2)

train.shape

test = pd.read_csv(COMP_DIR + 'test.csv')

test = test.fillna('')

test['Location'] = test.Country_Region + '-' + test.Province_State + '-' + test.County

test['DateTime'] = pd.to_datetime(test.Date)

test = test[['Location', 'DateTime', 'Target', 'ForecastId']]

test.head(2)

test.shape
test_predictions = test.merge(subm, how='inner', on=['ForecastId'])

test_predictions['ForecastId_Quantile'] = test_predictions.ForecastId.astype(str) +'_' + test_predictions.q.astype(str)

test_predictions['TargetQ'] = test_predictions.Target +' ' + test_predictions.q.astype(str)

test_predictions.head(2)

test_predictions.shape
overrides = pd.read_csv(EXTERNAL_DATA_PATH + 'w5_overrides.csv', sep=';')

overrides = overrides.fillna(0)

overrides.head()

confirmed = (test_predictions.Target == 'ConfirmedCases')

fatalities = (test_predictions.Target == 'Fatalities')

high = (test_predictions.q == '0.95')

med = (test_predictions.q == '0.5')

for i, row in tqdm(overrides.iterrows()):

    test_predictions.loc[(test_predictions.Location == row['Location']) & \

        confirmed & med, 'TargetValue'] += row['C50']

    test_predictions.loc[(test_predictions.Location == row['Location']) & \

        confirmed & high, 'TargetValue'] += row['C95']

    test_predictions.loc[(test_predictions.Location == row['Location']) & \

        fatalities & med, 'TargetValue'] += row['F50']

    test_predictions.loc[(test_predictions.Location == row['Location']) & \

        fatalities & high, 'TargetValue'] += row['F95']
top_locations = test_predictions.groupby(['Location', 'Target']).sum().reset_index()

top_locations = top_locations.pivot('Location', 'Target', 'TargetValue')

top_locations['Importance'] = top_locations.ConfirmedCases + 10 * top_locations.Fatalities

top_locations = top_locations.sort_values(by='Importance', ascending=False).head(60)

top_locations.to_csv('top_locations.csv')
lb_w_gt = pd.concat([train, test_predictions])
for loc in top_locations.index[:10]:

    df = lb_w_gt[lb_w_gt.Location == loc]



    fig = px.line(df[df.Target == 'ConfirmedCases'], x='DateTime', y='TargetValue', color='TargetQ')

    _ = fig.update_layout(title_text=f'Confirmed {loc}')

    fig.show()



    fig2 = px.line(df[df.Target == 'Fatalities'], x='DateTime', y='TargetValue', color='TargetQ')

    _ = fig2.update_layout(title_text=f'Fatalities {loc}')

    fig2.show()
df = lb_w_gt.groupby(['Target', 'TargetQ', 'DateTime']).sum().reset_index()



fig = px.line(df[df.Target == 'ConfirmedCases'], x='DateTime', y='TargetValue', color='TargetQ')

_ = fig.update_layout(title_text=f'Confirmed Total')

fig.show()



fig2 = px.line(df[df.Target == 'Fatalities'], x='DateTime', y='TargetValue', color='TargetQ')

_ = fig2.update_layout(title_text=f'Fatalities Total')

fig2.show()
submission = pd.read_csv(COMP_DIR + 'submission.csv')



submission.head()

submission.shape
lb_submit = submission.merge(test_predictions, how='left', on='ForecastId_Quantile')
lb_submit.count()
subm = lb_submit[['ForecastId_Quantile', 'TargetValue_y']].fillna(1)

subm.columns = ['ForecastId_Quantile', 'TargetValue']

subm.head()

subm.shape
subm.to_csv('submission.csv', index=False)