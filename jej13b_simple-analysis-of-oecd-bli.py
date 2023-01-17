import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
bli = pd.read_csv('../input/OECDBLI2017cleanedcsv.csv', index_col='Country')
# check highest corr vals

bli.corr().abs().stack().drop_duplicates().sort_values(ascending=False).head(n=20)
# graph linear reg. between Dwellings withOUT basic facilites as percentage and Life expectancy in years

plt.figure(figsize=(20,10))

sns.regplot(x=bli['Dwellings without basic facilities as pct'], y=bli['Life expectancy in yrs'], scatter_kws={'s':50})
# graph linear reg. between personal earnings in USD and Feeling safe walking alone at night as percentage

plt.figure(figsize=(20,10))

sns.regplot(x=bli['Personal earnings in usd'], y=bli['Feeling safe walking alone at night as pct'], scatter_kws={'s':100})
# check life satisfaction corr vals

bli.corr()['Life satisfaction as avg score'].abs().drop_duplicates().sort_values(ascending=False)
# graph life satisfaction with OECD-total threshhold marker

fig = plt.figure(figsize=(20,10))

ax = fig.add_subplot(111)

ax.bar(bli.index.values, bli['Life satisfaction as avg score'], width=.3)

ax.set_ylabel('Life satisfaction as avg score', fontsize=16)

ax.set_xlabel('Country', fontsize=16)

ax.set_title('Life satisfaction as average score by country')

plt.xticks(rotation='vertical')

ax.axhline(bli.loc['OECD - Total', 'Life satisfaction as avg score'], color='r', label='OECD-Total')

plt.legend()