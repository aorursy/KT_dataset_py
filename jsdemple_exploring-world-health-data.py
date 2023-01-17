# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
% matplotlib inline
df = pd.read_csv('../input/data.csv')
col_head = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code']

df.shape
df.tail()
df.columns
co = df['Country Name'].unique()

co
len(co)

ind = df['Indicator Name'].unique()

ind
len(ind)
lit = [i for i in ind if 'iteracy' in i]

#edu += [i for i in ind if 'iterature' in i]

sch = [i for i in ind if 'chool' in i]

edu = [i for i in ind if 'ucation' in i]

edu_tot = lit + sch + edu

edu_tot
lit2010 = df.loc[df['Indicator Name'].isin(lit) & df['2010']>0]

type(lit2010)

lit2010.shape

lit2010 = lit2010[col_head + ['2010']]

lit2010.sort_values('2010', inplace=True)
df.loc[df['Indicator Name'] ==\

'Literacy rate, adult female (% of females ages 15 and above)']
lit2010.plot(x='Country Name', y='2010', kind='bar', figsize=(120,10))
lit2010_adults = df.loc[df['Indicator Name'].isin([lit[2]]) & df['2010']>0]

lit2010_adults.shape

lit2010_adults = lit2010_adults[col_head + ['2010']]

lit2010_adults.sort_values('2010', inplace=True)
lit2010_adults.plot(x='Country Name', y='2010', kind='bar', figsize=(40,10))
df.loc[df['Country Name'].isin(['United States']) & df['Indicator Name'].isin([lit[2]])]
lit_adults = df.loc[df['Indicator Name'].isin([lit[2]])]

lit_adults
lit_adults.shape
lit_a = lit_adults.dropna(axis=1, how='all')
lit_a.shape
lit_a['mean_literacy'] = lit_a.mean(axis=1)

# drop rows with no mean_literacy

df.loc[df['Country Name'].isin(['United States']) & df['Indicator Name'].isin([lit[2]])]

lit_a = lit_a.loc[lit_a['mean_literacy'] > 0]
lit_a.sort_values('mean_literacy', inplace=True)

lit_a.plot(x='Country Name', y='mean_literacy', kind='bar', figsize=(60,10))
pd.DataFrame.hist(lit_a, column='mean_literacy', bins=10, figsize=(10,10))