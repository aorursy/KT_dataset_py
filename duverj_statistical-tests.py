import pandas as pd

pd.set_option('max_colwidth', 200)

pd.set_option('display.float_format', lambda x: '%.3f' % x)

from statsmodels.stats.weightstats import *

import scipy.stats



#this is the entire dataset, but we'll only be able to use to extract samples from it.

city_hall_dataset = pd.read_csv('../input/train.csv')
def results(p):

    if(p['p_value']<0.05):p['hypothesis_accepted'] = 'alternative'

    if(p['p_value']>=0.05):p['hypothesis_accepted'] = 'null'



    df = pd.DataFrame(p, index=[''])

    cols = ['value1', 'value2', 'score', 'p_value', 'hypothesis_accepted']

    return df[cols]
import numpy as np

city_hall_dataset['SalePrice'] = np.log1p(city_hall_dataset['SalePrice'])

logged_budget = np.log1p(120000) #logged $120 000 is 11.695

logged_budget
sample = city_hall_dataset.sample(n=25)

p = {} #dictionnary we'll use to stock information and results
p['value1'], p['value2'] = sample['SalePrice'].mean(), logged_budget

p['score'], p['p_value'] = stats.ttest_1samp(sample['SalePrice'], popmean=logged_budget)

results(p)
p['value1'], p['value2'] = sample['SalePrice'].mean(), logged_budget

p['score'], p['p_value'] = stats.ttest_1samp(sample['SalePrice'], popmean=logged_budget)

p['p_value'] = p['p_value']/2 #one-tailed test (with scipy function), we need to divide p-value by 2 ourselves

results(p)
smaller_houses = city_hall_dataset.sort_values('GrLivArea')[:730].sample(n=25)

larger_houses = city_hall_dataset.sort_values('GrLivArea')[730:].sample(n=25)
p['value1'], p['value2'] = smaller_houses['SalePrice'].mean(), larger_houses['SalePrice'].mean()

p['score'], p['p_value'], p['df'] = ttest_ind(smaller_houses['SalePrice'], larger_houses['SalePrice'])

results(p)
p['value1'], p['value2'] = smaller_houses['SalePrice'].mean(), larger_houses['SalePrice'].mean()

p['score'], p['p_value'], p['df'] = ttest_ind(smaller_houses['SalePrice'], larger_houses['SalePrice'], alternative='smaller')

results(p)
smaller_houses = city_hall_dataset.sort_values('GrLivArea')[:730].sample(n=100, random_state=1)

larger_houses = city_hall_dataset.sort_values('GrLivArea')[730:].sample(n=100, random_state=1)
p['value1'], p['value2'] = smaller_houses['SalePrice'].mean(), larger_houses['SalePrice'].mean()

p['score'], p['p_value'] = ztest(smaller_houses['SalePrice'], larger_houses['SalePrice'], alternative='smaller')

results(p)
from statsmodels.stats.proportion import *

A1 = len(smaller_houses[smaller_houses.SalePrice>logged_budget])

B1 = len(smaller_houses)

A2 = len(larger_houses[larger_houses.SalePrice>logged_budget])

B2 = len(larger_houses)

p['value1'], p['value2'] = A1/B1, A2/B2

p['score'], p['p_value'] = proportions_ztest([A1, A2], [B1, B2], alternative='smaller')

results(p)
p['value1'], p['value2'] = smaller_houses['SalePrice'].mean(), logged_budget

p['score'], p['p_value'] = ztest(smaller_houses['SalePrice'], value=logged_budget, alternative='larger')

results(p)
from statsmodels.stats.proportion import *

A = len(smaller_houses[smaller_houses.SalePrice<logged_budget])

B = len(smaller_houses)

p['value1'], p['value2'] = A/B, 0.25

p['score'], p['p_value'] = proportions_ztest(A, B, alternative='larger', value=0.25)

results(p)
replacement = {'FV': "Floating Village Residential", 'C (all)': "Commercial", 'RH': "Residential High Density",

              'RL': "Residential Low Density", 'RM': "Residential Medium Density"}

smaller_houses['MSZoning_FullName'] = smaller_houses['MSZoning'].replace(replacement)

mean_price_by_zone = smaller_houses.groupby('MSZoning_FullName')['SalePrice'].mean().to_frame()

mean_price_by_zone
sh = smaller_houses.copy()

p['score'], p['p_value'] = stats.f_oneway(sh.loc[sh.MSZoning=='FV', 'SalePrice'], 

               sh.loc[sh.MSZoning=='C (all)', 'SalePrice'],

               sh.loc[sh.MSZoning=='RH', 'SalePrice'],

               sh.loc[sh.MSZoning=='RL', 'SalePrice'],

               sh.loc[sh.MSZoning=='RM', 'SalePrice'],)

results(p)[['score', 'p_value', 'hypothesis_accepted']]
smaller_houses['GarageType'].fillna('No Garage', inplace=True)

smaller_houses['GarageType'].value_counts().to_frame()
city_hall_dataset['GarageType'].fillna('No Garage', inplace=True)

sample1 = city_hall_dataset.sort_values('GrLivArea')[:183].sample(n=100)

sample2 = city_hall_dataset.sort_values('GrLivArea')[183:366].sample(n=100)

sample3 = city_hall_dataset.sort_values('GrLivArea')[366:549].sample(n=100)

sample4 = city_hall_dataset.sort_values('GrLivArea')[549:730].sample(n=100)

dff = pd.concat([

    sample1['GarageType'].value_counts().to_frame(),

    sample2['GarageType'].value_counts().to_frame(), 

    sample3['GarageType'].value_counts().to_frame(), 

    sample4['GarageType'].value_counts().to_frame()], 

    axis=1, sort=False)

dff.columns = ['Sample1 (smallest houses)', 'Sample2', 'Sample3', 'Sample4 (largest houses)']

dff = dff[:3] #chi-square tests do not work when table contains some 0, we take only the most frequent attributes

dff 
p['score'], p['p_value'], p['ddf'], p['contigency'] = stats.chi2_contingency(dff)

p.pop('contigency')

results(p)[['score', 'p_value', 'hypothesis_accepted']]