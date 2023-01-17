import pandas as pd
import numpy as np
 
import matplotlib.pyplot as plt 
import seaborn as sns 
from plotnine import *
%matplotlib inline

from scipy import stats
from statsmodels import robust
state = pd.read_csv('/kaggle/input/state.csv')
state.shape
state.head()
# mean
state['Population'].mean()
# trimmed mean
stats.trim_mean(state['Population'], 0.1)
# median
state['Population'].median()
# weighted average
np.average(state["Murder.Rate"], weights=state["Population"])
# weighted median
state.sort_values('Murder.Rate', inplace=True)
cumsum = state.Population.cumsum()
cutoff = state.Population.sum() / 2.0
median_weight = state['Murder.Rate'][cumsum >= cutoff].iloc[0]
median_weight
# standard deviation
state['Population'].std()
# Interquatile range (IQR)
q1 = state['Population'].quantile(0.25)
q3 = state['Population'].quantile(0.75)
iqr = q3-q1
iqr
# MAD
robust.mad(state['Population'])
# homicide rate percentiles
q_5 = round(state['Murder.Rate'].quantile(0.05),2)
q_25 = round(state['Murder.Rate'].quantile(0.25),2)
q_50 = round(state['Murder.Rate'].quantile(0.50),2)
q_75 = round(state['Murder.Rate'].quantile(0.75),2)
q_95 = round(state['Murder.Rate'].quantile(0.95),2)
print(f'5% {q_5} - 25% {q_25} - 50% {q_50} - 75% {q_75} - 95% {q_95}')
# population data distribution by boxplot
plt.figure(figsize=(5,8))
sns.boxplot(y=state['Population']/1000000)
plt.title("Boxplot de Populações por estado")
plt.show()
# show outliers
state.sort_values(by='Population', ascending=False).head(4)
# histogram
plt.hist(state['Population'], bins='auto')
plt.xlabel('Population')
plt.ylabel('Frequency')
plt.show()
sns.distplot(state['Murder.Rate'])
plt.ylabel('Density')
plt.show()
# loading news datasets for this analysis
sp500_px = pd.read_csv('/kaggle/input/sp500_data.csv')
sp500_sym = pd.read_csv('/kaggle/input/sp500_sectors.csv')
sp500_px.head()
sp500_sym.head()
sp500_px = sp500_px.rename(columns={'Unnamed: 0': 'DATA'})

etfs = sp500_px.loc[sp500_px['DATA'] > "2012-07-01", 
                    sp500_sym[sp500_sym['sector'] == 'etf']['symbol']]
sns.heatmap(etfs.corr(), vmin=-1, vmax=1,
            cmap=sns.diverging_palette(20, 220, as_cmap=True))
plt.show()
#another way to plot using pandas
etfs_2 = etfs.corr()
etfs_2.style.background_gradient(cmap='coolwarm').set_precision(2)
# positively correlated
plt.xlabel('SPY')
plt.ylabel('DIA')
plt.scatter(etfs['SPY'], etfs['DIA'])
plt.show()
# negatively correlated
plt.xlabel('SPY')
plt.ylabel('VXX')
plt.scatter(etfs['SPY'], etfs['VXX'])
plt.show()
# no correlation
plt.xlabel('XLV')
plt.ylabel('GLD')
plt.scatter(etfs['XLV'], etfs['GLD'])
plt.show()
# loading new dataset for this analysis
kc_tax = pd.read_csv('/kaggle/input/kc_tax.csv')
kc_tax.head()
# Removing very expensive and very small or large residences
kc_tax0 = kc_tax.loc[(kc_tax.TaxAssessedValue < 750000) & 
                     (kc_tax.SqFtTotLiving > 100) & 
                     (kc_tax.SqFtTotLiving < 3500), :]
kc_tax0.shape
kc_tax0.plot(kind='hexbin', x='SqFtTotLiving', y='TaxAssessedValue', sharex=False, gridsize=25)
plt.xlabel('Finished Square Feet')
plt.ylabel('Tax Assessed Value')
plt.figure(figsize=(22,10))
plt.show()
kc_tax_zip1 = kc_tax0.loc[(kc_tax0.ZipCode == 98188), :]
kc_tax_zip2 = kc_tax0.loc[(kc_tax0.ZipCode == 98105), :]
kc_tax_zip3 = kc_tax0.loc[(kc_tax0.ZipCode == 98108), :]
kc_tax_zip4 = kc_tax0.loc[(kc_tax0.ZipCode == 98126), :]
sns.set_style('white')
sns.set_palette('dark')
sns.set_context('talk')

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12, 10))
ax1.hexbin(kc_tax_zip1.SqFtTotLiving, kc_tax_zip1.TaxAssessedValue, gridsize=20)
ax1.set_xlabel('zipcode = 98188')

ax2.hexbin(kc_tax_zip2.SqFtTotLiving, kc_tax_zip2.TaxAssessedValue, gridsize=20)
ax2.set_xlabel('zipcode = 98105')

ax3.hexbin(kc_tax_zip3.SqFtTotLiving, kc_tax_zip3.TaxAssessedValue, gridsize=20)
ax3.set_xlabel('zipcode = 98108')

ax4.hexbin(kc_tax_zip4.SqFtTotLiving, kc_tax_zip4.TaxAssessedValue, gridsize=20)
ax4.set_xlabel('zipcode = 98126')

plt.show()
lc_loans = pd.read_csv('/kaggle/input/lc_loans.csv')
lc_loans.head()
x_tab = pd.crosstab(lc_loans.grade, lc_loans.status)
x_tab['Total'] = pd.crosstab(lc_loans.grade, lc_loans.status).apply(lambda r: r.sum(), axis=1)
x_tab
airline_stats= pd.read_csv('/kaggle/input/airline_stats.csv')
airline_stats.sample(5)
plt.figure(figsize=(12,10))
sns.boxplot(x = airline_stats.airline, y = airline_stats.pct_carrier_delay)
plt.ylabel("% diário de voos atrasados")
plt.show()