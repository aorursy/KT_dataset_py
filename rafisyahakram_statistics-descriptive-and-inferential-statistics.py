#Basic setup

import numpy as np

import pandas as pd

import scipy as sp

import math

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
a = pd.DataFrame([5,5,5,6,7,8,8,9,9,10,13], columns=['value'])
def variance(df):

    var = 0

    mean = np.mean(df)

    

    for i in range(len(df)):

        var = var + (((df.loc[i,['value']]-mean)*(df.loc[i,['value']]-mean))/len(df))

    

    return sum(var)



def sd(df):

    var = 0

    mean = np.mean(df)

    

    for i in range(len(df)):

        var = var + (((df.loc[i,['value']]-mean)*(df.loc[i,['value']]-mean))/len(df))

        

    return math.sqrt(var)



def rg(df):

    ma = np.max(df)

    mi = np.min(df)

    

    c = ma - mi

    return sum(c)



def mode(df):

    df.sort_values(by=['value']).agg('count')

    h = df.sort_values(by=['value'], ascending=False).reset_index().groupby(['value']).agg(['count'])

    g = h['index']

    g.rename(columns={'count': 'J'}).reset_index().sort_values(by=['J'], ascending=False)

    res = g.iloc[0]

    mode = res.name

    

    return mode
print('Mean = ', sum(np.mean(a)), sep='')

print('Median = ', np.median(a), sep='')

print('Mode = ', mode(a), sep='')
describe = a.describe()

add = pd.DataFrame({'value': [variance(a), sd(a), rg(a), np.median(a), mode(a)]},

                    index=['var', 'std_dev', 'ran', 'median', 'mode'])

pd.concat([describe, add], axis=0).drop(['count', 'std']).rename(index={'mean': 'Mean', 'min': 'Minimum Value', '25%': 'Q1', '50%': 'Q2', '75%': 'Q3', 'max': 'Maximum Value', 'var': 'Variance', 'std_dev': 'Standard Deviation', 'ran': 'Range', 'median':'Median', 'mode': 'Mode'}, columns={'value': 'Value'})

values = pd.DataFrame(np.random.randint(165, 185, 50), columns=['value'])

sns.kdeplot(data=values['value'], shade=True)
from statsmodels.stats.weightstats import ztest



#zset, pval = ztest(values, x2=None, value=np.mean(values))

cf = (1.96*np.floor(sd(values)))/np.floor(math.sqrt(len(values)))



cf_min = sum(np.mean(values)-cf)

cf_max = sum(np.mean(values)+cf)

  

print('Mean = ', sum(np.mean(values)), '\nStandard Deviation = ', np.floor(sd(values)), '\nNumber of samples = ', len(values), sep='')

print('\nConfidence Interval: ', sum(np.mean(values)), ' ± ', cf, '\nBetween ',cf_min, ' and ', cf_max, sep='')