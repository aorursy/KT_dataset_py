import numpy as np

import pylab as pl

import pandas as pd

import matplotlib.pyplot as plt 

%matplotlib inline

import seaborn as sns

from scipy.stats import spearmanr

import os
train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/train.csv")

BCG_policy = pd.read_csv("/kaggle/input/bcg-data-otazu/BCG_Data.csv")

train.head(2)
BCG_policy.head(2)
BCG_policy.columns = ('Country_Region','IncomeLevel','Population','Total COVID-19 Tests Administered (lower bound)','COVID-19 cases','COVID-19 Deaths','case mortality','BCG % coverage','BCG strain','BCG_policy')

BCG_policy.head(2)
country_date = pd.DataFrame(train.groupby(['Country_Region', 'Date'],as_index=False).agg({'ConfirmedCases':['sum'],'Fatalities':['sum']}))

country_date[country_date['Country_Region'] == 'China'].head(5)
country_date.columns = ('Country_Region','Date','ConfirmedCases','Fatalities')

country_aggs = pd.DataFrame(country_date.groupby(['Country_Region'],as_index=False).agg({'ConfirmedCases':['max'],'Fatalities':['max']}))

country_aggs['CaseFatalityRate'] = country_aggs['Fatalities']/country_aggs['ConfirmedCases']

country_aggs.columns = ('Country_Region','TotalConfirmedCases','TotalFatalities','CaseFatalityRate')

country_aggs.head(2)
combined = pd.merge(country_aggs,

                 BCG_policy,

                 on='Country_Region', 

                 how='left')

combined['BCG_policy'].fillna('0', inplace=True)

combined['BCG_policy'] = combined['BCG_policy'].apply({'--':0,'0':0,'1':1,'2':2,'3':3}.get)

#pd.set_option('display.max_rows', 200)

combined.head(2)
ax = sns.boxplot(x="BCG_policy", y="CaseFatalityRate", data=combined)

ax = sns.stripplot(x="BCG_policy", y="CaseFatalityRate", color='black',alpha=0.3,data=combined)

ax.set_ylim([0, 0.125]) 
combined[combined['BCG_policy'] == 3].head(10)
ax = sns.boxplot(x="IncomeLevel", y="CaseFatalityRate", data=combined)

ax = sns.stripplot(x="IncomeLevel", y="CaseFatalityRate", color='black',alpha=0.3,data=combined)

ax.set_ylim([0, 0.125]) 
#sns.boxplot(x="day", y="total_bill", hue="smoker", data=df, palette="Set1")

fig, ax = plt.subplots(1, 1, figsize=(12, 8))

ax = sns.boxplot(x="IncomeLevel", y="CaseFatalityRate",hue="BCG_policy", data=combined)

ax = sns.stripplot(x="IncomeLevel", y="CaseFatalityRate", hue="BCG_policy" , color='black',alpha=0.3,data=combined)

ax.set_ylim([0, 0.15]) 
combined['IncomeLevel2'] = combined['IncomeLevel'].apply({1:'LowerLowerMiddle',2:'LowerLowerMiddle',3: 'MiddleHighHigh', 4: 'MiddleHighHigh'}.get)

fig, ax = plt.subplots(1, 1, figsize=(12, 8))

ax = sns.boxplot(x="BCG_policy", y="CaseFatalityRate",hue="IncomeLevel2", data=combined[combined['TotalConfirmedCases'] >= 1000])

ax = sns.stripplot(x="BCG_policy", y="CaseFatalityRate", hue="IncomeLevel2" , color='black',alpha=0.3,data=combined[combined['TotalConfirmedCases'] >= 1000])

ax.set_ylim([0, 0.15]) 
from scipy.stats import mannwhitneyu

data1 = combined[(combined['IncomeLevel'].astype(float) >= 3) & (combined['BCG_policy'].astype(float) == 3)]["CaseFatalityRate"]

data2 = combined[(combined['IncomeLevel'].astype(float) >= 3) & (combined['BCG_policy'].astype(float) == 1)]["CaseFatalityRate"]

# compare samples

stat, p = mannwhitneyu(data1 , data2)

print('Statistics=%.3f, p=%.3f' % (stat, p))

# interpret

alpha = 0.05

if p > alpha:

    print('Same distribution (fail to reject H0)')

else:

    print('Different distribution (reject H0)')
from math import sqrt

r=0.03

p1=r*3

p2=r

p=(p1+p2)/2.0

ES = (p1-p2)/sqrt(p*(1-p))

ES
# estimate sample size via power analysis

from statsmodels.stats.power import TTestIndPower

# parameters for power analysis

effect = ES

alpha = 0.05

power = 0.8

# perform power analysis

analysis = TTestIndPower()

result = analysis.solve_power(effect, power=power, nobs1=None, ratio=1.0, alpha=alpha)

print('Sample Size: %.3f' % result)