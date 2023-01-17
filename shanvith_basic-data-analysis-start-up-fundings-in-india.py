# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # plotting!

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/startup_funding.csv', encoding='utf-8')

# removing null data
df = df[df['CityLocation'].notnull() & df['AmountInUSD'].notnull() & df['InvestorsName'].notnull()]

# removing bad city locaitons like ones with '/' and non Indian cities
df = df.loc[df['CityLocation'].str.contains('/')==False]
df = df.loc[df['CityLocation'].str.contains('Missourie')==False]
df = df.loc[df['CityLocation'].str.contains('US')==False]
df = df.loc[df['CityLocation'].str.contains('USA')==False]
df = df.loc[df['CityLocation'].str.contains('London')==False]
df = df.loc[df['CityLocation'].str.contains('Boston')==False]
df = df.loc[df['CityLocation'].str.contains('bangalore')==False]
df = df.loc[df['CityLocation'].str.contains('Singapore')==False]

# changing type of USD from string to float
df['AmountInUSD'] = df['AmountInUSD'].str.replace(',','')
df['AmountInUSD'] = df['AmountInUSD'].str.replace('$', '')
df['AmountInUSD'] = np.array(df['AmountInUSD']).astype(np.float)

# removing outliers
q = df['AmountInUSD'].quantile(.99)
df = df[df['AmountInUSD'] <= q]

df.head()
# Different type of investments
g = sns.factorplot('InvestmentType', data=df, kind='count')
g.set_xticklabels(rotation=30)
g.fig.set_size_inches(20,10)
# Number of start ups that get funding in particular city
g = sns.factorplot('CityLocation', data=df, kind='count')
g.set_xticklabels(rotation=90)
g.fig.set_size_inches(20,10)
# splits the investors by the commas, no commas means 1 investor, one comma means 2 investors, etc.
investors = df['InvestorsName'].str.split(',')
counts = []
for investor in investors:
    counts.append(len(investor))
df['InvestorsCount'] = counts

# Number of investors per start up "count"
g = sns.factorplot('InvestorsCount', data=df, kind='count')
g.set_xticklabels(rotation=90)
g.fig.set_size_inches(20,10)
# Investors grouped by type of funding
g = sns.factorplot('InvestorsCount', hue='InvestmentType',data=df, kind='count')
g.set_xticklabels(rotation=90)
g.fig.set_size_inches(20,10)
# Number of investors and their relation to how much funding a start up recieves

# getting the means for each type of InvestorsCount
nums = [1,2,3,4,5,6,7,8,10]
means = df.groupby(['InvestorsCount'])['AmountInUSD'].mean()
cumulative = pd.DataFrame({'nums':nums, 'means':means})

# plotting the graphs
g = sns.boxplot(x='InvestorsCount', y='AmountInUSD', data=df, showfliers=False)
g = sns.pointplot(x=nums, y=means, data=cumulative)

# configuring the displays
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.figure.set_size_inches(20,10)