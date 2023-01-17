import pandas as pd

from pandas import Series,DataFrame

import numpy as np

from scipy.stats import linregress



# For Visualization

import matplotlib.pyplot as plt

import matplotlib

import seaborn as sns
#data

df=pd.read_csv('../input/indicators_by_company.csv')

df.head(5)
indicators=['Assets','LiabilitiesAndStockholdersEquity',

'StockholdersEquity',

'CashAndCashEquivalentsAtCarryingValue',

'NetCashProvidedByUsedInOperatingActivities',

'NetIncomeLoss',

'NetCashProvidedByUsedInFinancingActivities',

'CommonStockSharesAuthorized',

'CashAndCashEquivalentsPeriodIncreaseDecrease',

'CommonStockValue',

'CommonStockSharesIssued',

'RetainedEarningsAccumulatedDeficit',

'CommonStockParOrStatedValuePerShare',

'NetCashProvidedByUsedInInvestingActivities',

'PropertyPlantAndEquipmentNet',

'AssetsCurrent',

'LiabilitiesCurrent',

'CommonStockSharesOutstanding',

'Liabilities',

'OperatingIncomeLoss' ]
Values=df.loc[df['indicator_id'].isin(indicators),['company_id','indicator_id','2011']]

Values=pd.melt(Values, id_vars=['company_id', 'indicator_id'], var_name='year', value_name='value')

Values=Values.loc[Values['year']=='2011',['company_id','indicator_id','value']].pivot(index='company_id',columns='indicator_id', values='value').dropna()

Values.head(5)
print('There are '+str(len(Values))+' companies with not null values in the data set for 2011 and 20 indicators')
#heatmap visualization

def heatmap(data,title):

  fig, ax = plt.subplots(figsize=(15, 15))

  heatmap = sns.heatmap(data, cmap=plt.cm.Blues,annot=True, annot_kws={"size": 8})

  #ax.xaxis.tick_top()

  ax.set_title(title)

  # rotate

  plt.xticks(rotation=90)

  plt.yticks(rotation=0)

  plt.tight_layout()
#skipy linregress

#Pearson Correlation

rvalue = DataFrame(np.nan,index=indicators,columns=indicators)

#PValue

pvalue = DataFrame(np.nan,index=indicators,columns=indicators)

#StdErr

stderr = DataFrame(np.nan,index=indicators,columns=indicators)
#

for c_X in indicators:

  for c_Y in indicators:

    R=linregress(Values[[c_X,c_Y]])

    rvalue.set_value(c_Y,c_X, R.rvalue)

    pvalue.set_value(c_Y,c_X, R.pvalue)

    stderr.set_value(c_Y,c_X, R.stderr)
heatmap(rvalue,'R-value')
heatmap(pvalue,'P-value')
heatmap(stderr,'std-error')