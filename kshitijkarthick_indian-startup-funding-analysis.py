# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



sns.set(style="whitegrid", color_codes=True)

pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/startup_funding.csv', parse_dates=['Date'], index_col='SNo')

df.AmountInUSD = df.AmountInUSD.str.replace(',','').astype('float32')

df.IndustryVertical = df.IndustryVertical.replace('eCommerce', 'ECommerce')

df.InvestmentType = df.InvestmentType.replace('Crowd funding', 'Crowd Funding')

df.InvestmentType = df.InvestmentType.replace('SeedFunding', 'Seed Funding')

df.InvestmentType = df.InvestmentType.replace('PrivateEquity', 'Private Equity')

df.Date = df.Date.str.replace('12/05.2015', '12/05/2015')

df.Date = df.Date.str.replace('13/04.2015', '13/04/2015')

df.Date = df.Date.str.replace('15/01.2015', '15/01/2015')

df.Date = df.Date.str.replace('22/01//2015', '22/01/2015')

df.Date = pd.to_datetime(df['Date'], format='%d/%m/%Y')

df.head()
df.describe(include='all').loc[['count', 'unique', 'top']]
df.AmountInUSD.describe()
df[df.AmountInUSD == 1400000000]
plt.figure(figsize=(15,10))

# Plotting hist without kde

ax = sns.distplot(df.AmountInUSD.dropna(), kde=False)



# Creating another Y axis

second_ax = ax.twinx()



#Plotting kde without hist on the second Y axis

fig = sns.distplot(df.AmountInUSD.dropna(), ax=second_ax, kde=True, hist=True, bins=20)

fig.get_xaxis().get_major_formatter().set_scientific(False)

fig.get_yaxis().get_major_formatter().set_scientific(False)

#Removing Y ticks from the second axis

second_ax.set_yticks([])

fig
plt.figure(figsize=(15,10))

total_investment_per_industry = df.pivot_table(index='IndustryVertical', values=['AmountInUSD'], aggfunc=np.sum, fill_value=0)

top_20_total_investment_per_industry = total_investment_per_industry.sort_values(by=['AmountInUSD'], ascending=False).head(20)

fig = sns.barplot(y=top_20_total_investment_per_industry.index, x=top_20_total_investment_per_industry['AmountInUSD'], orient='h')

fig.get_xaxis().get_major_formatter().set_scientific(False)

fig.set_title('Top 20 Verticals with their Total Investments')

for p in fig.patches:

    fig.annotate("{} million $".format(p.get_width() / 1000000.0), (p.get_width() * 1.05, p.get_y()))
plt.figure(figsize=(10,10))

bottom_20_total_investment_per_industry = total_investment_per_industry[total_investment_per_industry > 0].sort_values(by=['AmountInUSD'], ascending=True).head(20)

fig = sns.barplot(y=bottom_20_total_investment_per_industry.index, x=bottom_20_total_investment_per_industry['AmountInUSD'], orient='h')

fig.set_title('Bottom 20 Total Investments based on Verticals')
total_investment_per_industry_per_vert = df.pivot_table(index=['IndustryVertical', 'SubVertical', 'StartupName'], values=['AmountInUSD'], aggfunc=np.sum, fill_value=0)

total_investment_per_industry_per_vert.sort_values(by=['AmountInUSD'], ascending=False).head(20)
common_industry_verticals = df.dropna(subset=['AmountInUSD']).groupby('IndustryVertical').agg(['min', 'max', 'mean', 'count']).sort_values([('AmountInUSD', 'count')], ascending=False)['AmountInUSD']

top_10_common_industry_verticals = common_industry_verticals.head(10)

top_10_common_industry_verticals
# g = sns.FacetGrid(top_10_common_industry_verticals, col='industry')

# g.map(sns.barplot, "count", "mean")



# top_10_common_industry_verticals.loc[top_10_common_industry_verticals.index, 'Industry'] = top_10_common_industry_verticals.index.tolist()

# top_10_common_industry_verticals_plot = pd.melt(top_10_common_industry_verticals, id_vars=['Industry'], var_name="Metric Type", value_name='Value')

# g = sns.FacetGrid(top_10_common_industry_verticals_plot, col='Industry', col_wrap=4, sharex=False, row_order=['count', 'min', 'median', 'max'])

# g.map(sns.barplot, 'Metric Type', 'Value')
plt.figure(figsize=(10, 6))

investment_avg = df.pivot_table(index=['InvestmentType'], values=['AmountInUSD'], aggfunc=np.mean)

fig = sns.barplot(x=investment_avg.index, y=investment_avg.AmountInUSD)

fig.set_title('Avg Investment based on Type')

fig.get_yaxis().get_major_formatter().set_scientific(False)

fig.get_yaxis().get_major_formatter().set_scientific(False)

for p in fig.patches:

    fig.annotate("{} $".format(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))

plt.ticklabel_format(use_offset=False)
plt.figure(figsize=(10, 5))

df['Year'] = df['Date'].dt.year

df['Month'] = df['Date'].dt.month

yearly_investment = df[['Year', 'AmountInUSD']].groupby('Year').sum()

fig = sns.pointplot(x=yearly_investment.index, y=yearly_investment['AmountInUSD'])

fig.set_title('Yearly Total Investment')

fig.get_yaxis().get_major_formatter().set_scientific(False)
plt.figure(figsize=(10, 5))

fig = sns.violinplot(x=df.Month, y=df.AmountInUSD)

fig.set_title('Monthly Total Investment')

fig.get_yaxis().get_major_formatter().set_scientific(False)