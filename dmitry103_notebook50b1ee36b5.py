import pandas as pd

import numpy as np
data = pd.read_csv('../input/train.csv', sep=',', low_memory=False)

data['StateHoliday'] = data['StateHoliday'].apply(lambda x: str(x))
grouppedByStoreDayPromo = data[data['Sales'] > 0].groupby(by=['Store', 'DayOfWeek', 'Promo'])
test = pd.read_csv('../input/test.csv', sep=',')
mn = data[data['Sales'] > 0].groupby(['Store', 'DayOfWeek', 'Promo'])['Sales'].mean().reset_index()
res = pd.merge(left=test, right=mn, on=['Store', 'DayOfWeek', 'Promo'], how='left')

res.ix[res['Open'] == 0, 'Sales'] = 0
res[['Id', 'Sales']].to_csv('result_mean.csv', sep=',', index=None)