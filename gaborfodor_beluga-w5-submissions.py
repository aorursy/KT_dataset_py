



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import plotly.express as px
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/train.csv')

train = train.fillna('')

train['Location'] = train.Country_Region + '-' + train.Province_State + '-' + train.County

train['DateTime'] = pd.to_datetime(train.Date)

train['TargetQ'] = train.Target + 'Actual'

train = train[['Location', 'DateTime', 'Target', 'TargetQ', 'TargetValue']]

train.head(2)

train.shape
test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/test.csv')

test = test.fillna('')

test['Location'] = test.Country_Region + '-' + test.Province_State + '-' + test.County

test['DateTime'] = pd.to_datetime(test.Date)

test = test[['Location', 'DateTime', 'Target', 'ForecastId']]

test.head(2)

test.shape
subm = pd.read_csv('/kaggle/input/covid19belugaw5/w5_private_submission_updated_fixed.csv')

subm['ForecastId'] = subm.ForecastId_Quantile.map(lambda s: s.split('_')[0]).astype(int)

subm['q'] = subm.ForecastId_Quantile.map(lambda s: s.split('_')[1])

subm.head(2)
test_predictions = test.merge(subm, how='inner', on=['ForecastId'])

test_predictions['ForecastId_Quantile'] = test_predictions.ForecastId.astype(str) +'_' + test_predictions.q.astype(str)

test_predictions['TargetQ'] = test_predictions.Target +' ' + test_predictions.q.astype(str)

test_predictions.head(2)

test_predictions.shape
subm.tail()
subm.shape
lb_w_gt = pd.concat([train, test_predictions])

df = lb_w_gt.groupby(['Target', 'TargetQ', 'DateTime']).sum().reset_index()



fig = px.line(df[df.Target == 'ConfirmedCases'], x='DateTime', y='TargetValue', color='TargetQ')

_ = fig.update_layout(title_text=f'Confirmed Total')

fig.show()



fig2 = px.line(df[df.Target == 'Fatalities'], x='DateTime', y='TargetValue', color='TargetQ')

_ = fig2.update_layout(title_text=f'Fatalities Total')

fig2.show()
subm[['ForecastId_Quantile', 'TargetValue']].to_csv('submission.csv', index=False)