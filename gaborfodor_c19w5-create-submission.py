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
VERSION = 11

EXTERNAL_DATA_PATH = '../input/covid19belugaw5/'



COMP_DIR = '../input/covid19-global-forecasting-week-5/'





WORK_DIR = '/kaggle/working/'

# LOG_FILE = f'{WORK_DIR}paropt_lgb_v{VERSION}.csv'

# FIMP_FILE = f'{WORK_DIR}fimp_lgb_v{VERSION}.csv'

# FEATURE_FILE_PATH = f'{WORK_DIR}features_v{VERSION}.csv'

PREDS_PATH = f'{WORK_DIR}predictions_v{VERSION}/'

if not os.path.exists(PREDS_PATH):

    os.makedirs(PREDS_PATH)
train = pd.read_csv(COMP_DIR + 'train.csv')

train = train.fillna('')

train['Location'] = train.Country_Region + '-' + train.Province_State + '-' + train.County

train['DateTime'] = pd.to_datetime(train.Date)

train['TargetQ'] = train.Target + 'Actual'

train = train[['Location', 'DateTime', 'Target', 'TargetQ', 'TargetValue']]

train.head()

train.shape

test = pd.read_csv(COMP_DIR + 'test.csv')

test = test.fillna('')

test['Location'] = test.Country_Region + '-' + test.Province_State + '-' + test.County

test['DateTime'] = pd.to_datetime(test.Date)

test = test[['Location', 'DateTime', 'Target', 'ForecastId']]

test.head()

test.shape
# refresh if you have a bunch of fresh lgbs

# preds = []

# for f in tqdm(os.listdir(PREDS_PATH)):

#     if f.endswith('.csv'):

#         df = pd.read_csv(PREDS_PATH + f)

#         df['DateTime'] = pd.to_datetime(df.Date)

#         df['PredictionDay'] = df['DateTime'] + dt.timedelta(df.k.mean())

#         df['Target'] = df.target.str.replace('Confirmed', 'ConfirmedCases')

#         df['file'] = f

#         preds.append(df)



# preds = pd.concat(preds)

# preds = preds.drop(columns=['target'])

# preds = preds.rename(columns={'PREDICTION': 'TargetValue'})

# preds['DaysTillEnd'] = (preds.DateTime.max() - preds.DateTime).dt.days + 1

# preds['Decay'] = 0.5 ** preds['DaysTillEnd']

# preds.to_csv('lgb_oof_predictions.csv', index=False)
preds = pd.read_csv(EXTERNAL_DATA_PATH + 'lgb_oof_predictions.csv')

preds['DateTime'] = pd.to_datetime(preds.DateTime)

preds['PredictionDay'] = pd.to_datetime(preds.PredictionDay)



preds.shape

preds.head()
preds['PredDecay'] = preds.TargetValue * preds.Decay
lb_predictions = preds[preds.DaysTillEnd <= 2].groupby([

    'Location', 'PredictionDay', 'q', 'Target'])[['PredDecay', 'Decay']].sum().reset_index()

lb_predictions = lb_predictions.rename(columns={'PredictionDay': 'DateTime'})

lb_predictions['TargetValue'] = lb_predictions.PredDecay / lb_predictions.Decay

lb_predictions = lb_predictions.drop(columns=['PredDecay'])

lb_predictions.TargetValue = lb_predictions.TargetValue.clip(0, None)

lb_predictions.head(10)

us_states = lb_predictions[

    lb_predictions.Location.str.startswith('US') & lb_predictions.Location.str.endswith('-')]
us_total_med = us_states.groupby(['DateTime', 'q', 'Target']).sum().reset_index()

us_total_med = us_total_med[us_total_med.q == 0.5]

us_total_med['Location'] = 'US--'

us_total_med = us_total_med[lb_predictions.columns]

us_total_med.tail()
us_total_low = us_total_med.copy()

us_total_low.q = 0.05

us_total_low.loc[us_total_low.Target == 'ConfirmedCases', 'TargetValue'] -= 12000

us_total_low.loc[us_total_low.Target == 'Fatalities', 'TargetValue'] -= 1000



us_total_high = us_total_med.copy()

us_total_high.q = 0.95

us_total_high.loc[us_total_high.Target == 'ConfirmedCases', 'TargetValue'] += 8000

us_total_high.loc[us_total_high.Target == 'Fatalities', 'TargetValue'] += 1000
us_total = pd.concat([us_total_med, us_total_low, us_total_high])
lb_predictions = pd.concat([lb_predictions, us_total])

lb_predictions.min()

lb_predictions.TargetValue = lb_predictions.TargetValue.clip(0, None)

lb_predictions.shape
median = lb_predictions.loc[lb_predictions.q == 0.5, ['Location', 'DateTime', 'Target', 'TargetValue']]



lb_predictions = lb_predictions.merge(median, on=['Location', 'DateTime', 'Target'], suffixes=['', 'Median'])
lb_predictions.head()

lb_predictions.shape
low = lb_predictions.q == 0.05

high = lb_predictions.q == 0.95
np.mean(lb_predictions.loc[low, 'TargetValue'] > lb_predictions.loc[low, 'TargetValueMedian'])

np.mean(lb_predictions.loc[high, 'TargetValue'] > lb_predictions.loc[high, 'TargetValueMedian'])
change_low_idx = low & (lb_predictions.TargetValueMedian < lb_predictions.TargetValue)

np.mean(change_low_idx)

lb_predictions.loc[change_low_idx, 'TargetValue'] = lb_predictions.loc[change_low_idx, 'TargetValueMedian']



change_high_idx = high & (lb_predictions.TargetValueMedian > lb_predictions.TargetValue)

np.mean(change_high_idx)

lb_predictions.loc[change_high_idx, 'TargetValue'] = lb_predictions.loc[change_high_idx, 'TargetValueMedian']
lb_predictions.describe()
lb_predictions
DECAY = 0.99

second_week_mean = lb_predictions[lb_predictions.DateTime > '2020-05-17'].groupby(

    ['Location', 'Target', 'q'])[['TargetValue']].mean().reset_index()

all_predictions = [lb_predictions]

for k in range(1, 20):

    df = second_week_mean.copy()

    df['DateTime'] = lb_predictions.DateTime.max() + dt.timedelta(k)

    df.loc[df.q == 0.05, 'TargetValue'] *= DECAY ** k

    df.loc[df.q == 0.5, 'TargetValue'] *= DECAY ** k

    all_predictions.append(df)



all_predictions = pd.concat(all_predictions)

all_predictions.head()

all_predictions.shape
test_predictions = test.merge(all_predictions, how='left', on=['Location', 'DateTime', 'Target'])

test_predictions['ForecastId_Quantile'] = test_predictions.ForecastId.astype(str) +'_' + test_predictions.q.astype(str)

test_predictions['TargetQ'] = test_predictions.Target +' ' + test_predictions.q.astype(str)

test_predictions.head()
top_locations = all_predictions.groupby(['Location', 'Target']).sum().reset_index()

top_locations = top_locations.pivot('Location', 'Target', 'TargetValue')

top_locations['Importance'] = top_locations.ConfirmedCases + 10 * top_locations.Fatalities

top_locations.sort_values(by='Importance', ascending=False).head(60)

top_locations.to_csv('top_locations.csv')
lb_w_gt = pd.concat([train, test_predictions])
loc = 'US--'

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
lb_submit.groupby('DateTime').count()
subm = lb_submit[['ForecastId_Quantile', 'TargetValue_y']].fillna(1)

subm.columns = ['ForecastId_Quantile', 'TargetValue']

subm.head()

subm.shape
subm.to_csv('submission.csv', index=False)