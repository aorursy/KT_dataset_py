

import numpy as np

import pylab as pl

import pandas as pd

import matplotlib.pyplot as plt 

%matplotlib inline

import seaborn as sns

from scipy.stats import spearmanr

import os
train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/train.csv")

transport_rankings = pd.read_csv("/kaggle/input/transport-rankings-by-state/US_Transport_Rankings.csv")

train.head(2)
transport_rankings.head(5)
transport_rankings.columns=('OverallTransportRank','Province_State','CommuteTime','PublicTransitUsage','RoadQuality','BridgeQuality')

train = train[train['Country_Region'] == 'US']
unique = pd.DataFrame(train.groupby(['Country_Region', 'Province_State'],as_index=False)['ConfirmedCases'].sum())

unique['ConfirmedCases_rank'] = unique['ConfirmedCases'].rank(ascending=False)

unique.sort_values(by=['ConfirmedCases_rank'], inplace=True)

unique.head(5)
combined = pd.DataFrame(unique.merge(transport_rankings, on='Province_State'))

combined.head(5)
coef, p = spearmanr(combined['ConfirmedCases_rank'], combined['PublicTransitUsage'])

print('Spearmans rank correlation coefficient and p-value respectively: %.3f' % coef,p)

# interpret the significance

alpha = 0.05

if p > alpha:

    print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)

else:

    print('Samples are correlated (reject H0) p=%.3f' % p)
from scipy.stats import kendalltau

# calculate kendall's correlation

coef, p = kendalltau(combined['ConfirmedCases_rank'], combined['PublicTransitUsage'])

print('Kendall correlation coefficient and p-value respectively: %.3f' % coef, p)

# interpret the significance

alpha = 0.05

if p > alpha:

    print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)

else:

    print('Samples are correlated (reject H0) p=%.3f' % p)
plt.scatter(combined['ConfirmedCases_rank'], combined['PublicTransitUsage'])

# show line plot

plt.title('Confirmed Cases Rank vs Public Transport Usage Rank')

plt.show()
coef, p = spearmanr(combined['ConfirmedCases_rank'], combined['RoadQuality'])

print('Spearmans rank correlation coefficient and p-value respectively (RoadQuality): %.3f' % coef,p)

coef, p = spearmanr(combined['ConfirmedCases_rank'], combined['BridgeQuality'])

print('Spearmans rank correlation coefficient and p-value respectively (BridgeQuality): %.3f' % coef,p)

coef, p = spearmanr(combined['ConfirmedCases_rank'], combined['OverallTransportRank'])

print('Spearmans rank correlation coefficient and p-value respectively (OverallTrasportationRank): %.3f' % coef,p)
HDI_rankings = pd.read_csv("/kaggle/input/american-human-development-index/US_HDI_Rankings.csv")

HDI_rankings.head()
HDI_rankings.columns=('HDI_rank','Province_State','HDI')

combined = pd.DataFrame(unique.merge(HDI_rankings, on='Province_State'))

combined.head(5)
coef, p = spearmanr(combined['ConfirmedCases_rank'], combined['HDI_rank'])

print('Spearmans rank correlation coefficient and p-value respectively: %.3f' % coef,p)

# interpret the significance

alpha = 0.05

if p > alpha:

    print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)

else:

    print('Samples are correlated (reject H0) p=%.3f' % p)
# calculate kendall's correlation

coef, p = kendalltau(combined['ConfirmedCases_rank'], combined['HDI_rank'])

print('Kendall correlation coefficient: %.3f' % coef)

# interpret the significance

alpha = 0.05

if p > alpha:

    print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)

else:

    print('Samples are correlated (reject H0) p=%.3f' % p)
plt.scatter(combined['ConfirmedCases_rank'], combined['HDI_rank'])

# show line plot

plt.title('Confirmed Cases Rank vs Human Development Index')

plt.show()