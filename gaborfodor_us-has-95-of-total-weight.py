import numpy as np

import pandas as pd

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/test.csv')

test = test.fillna('')

test['Location'] = test.Country_Region + '-' +test.Province_State + '-' + test.County

test.head()

test.shape

test.nunique()
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/train.csv')



train.nunique()

train.count()



train = train.fillna('')
confirmed_low = train[(train.Date >= '2020-04-05') & (train.Target == 'ConfirmedCases')].groupby(['County', 'Province_State', 'Country_Region'])[['TargetValue']].quantile(.05)

confirmed_high = train[(train.Date >= '2020-04-05') & (train.Target == 'Fatalities')].groupby(['County', 'Province_State', 'Country_Region'])[['TargetValue']].quantile(.95)

fatalities_low = train[(train.Date >= '2020-04-05') & (train.Target == 'Fatalities')].groupby(['County', 'Province_State', 'Country_Region'])[['TargetValue']].quantile(.05)

fatalities_high = train[(train.Date >= '2020-04-05') & (train.Target == 'Fatalities')].groupby(['County', 'Province_State', 'Country_Region'])[['TargetValue']].quantile(.95)
quantiles = pd.concat([confirmed_low, confirmed_high, fatalities_low, fatalities_high], axis=1)

quantiles.columns = ['confirmed_low', 'confirmed_high', 'fatalities_low', 'fatalities_high']
quantiles['ValueRange'] = np.abs(quantiles.confirmed_high - quantiles.confirmed_low) + 10 * np.abs((quantiles.fatalities_high - quantiles.fatalities_low))
quantiles.sort_values(by='ValueRange', ascending=False)




locs = train.groupby(['County', 'Province_State', 'Country_Region']).mean()[['Population']]
locs = train.groupby(['County', 'Province_State', 'Country_Region']).mean()[['Population']]

locs = locs.reset_index()



locs['Weight'] = 1. / np.log(locs.Population + 1)

locs['RelativeWeight'] = locs['Weight'] / locs['Weight'].sum()

locs[locs.Country_Region =='US'].head()
country_weights = locs.groupby('Country_Region').sum().sort_values(by='Weight', ascending = False)

country_weights.head(10)

country_weights.tail(10)
locs_weight = locs.merge(quantiles, on = ['County', 'Province_State', 'Country_Region'])
locs_weight['HeuristicWeight'] = locs_weight.Weight * locs_weight.ValueRange

locs_weight['RelativeHeuristicWeight'] = locs_weight['HeuristicWeight'] / locs_weight['HeuristicWeight'].sum()
country_weights = locs_weight.groupby('Country_Region').sum().sort_values(by='HeuristicWeight', ascending = False)

country_weights.head(30)

country_weights.tail(10)