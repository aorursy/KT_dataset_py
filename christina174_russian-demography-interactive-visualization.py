import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
# import chart_studio.plotly as py

import plotly.graph_objs as go

from plotly.offline import iplot

import cufflinks

cufflinks.go_offline()
data = pd.read_csv('../input/russian-demography/russian_demography.csv')
data.isnull().sum()
data = data.dropna()

data.isnull().any()
sns.heatmap(data.corr(), square=True, annot=True, cbar=False);
data.iloc[:, 1:].describe()
df_gb = data.groupby('year')[['birth_rate', 'death_rate']].mean()
df_gb
df_gb.iplot(mode='lines+markers', xTitle='Year', yTitle='Average',

    title='Yearly Average birth_rate and death_rate')
df_ = data.groupby('region')[['birth_rate', 'death_rate']].mean()

df_.iplot(kind='bar', xTitle='Region', yTitle='Average',

    title='Average birth_rate and death_rate in regions')
df_npg = data.groupby('region')[['npg']].mean()

df_npg.iplot(kind='bar', xTitle='Natural population growth by 1000 people', yTitle='Average',

    title='Average natural population growth by 1000 people in regions')
plt.scatter(data['year'],  data['gdw'], label=None,

            c=data['urbanization'], cmap='viridis',

            linewidth=0, alpha=0.5)

plt.xlabel('year')

plt.ylabel('general demographic weight')

plt.colorbar(label='% of urban population');
data_capital = pd.DataFrame()

data_capital = data[(data['region']=='Moscow') | (data['region']=='Saint Petersburg') | (data['region']=='Leningrad Oblast') | (data['region']=='Moscow Oblast')]

data_capital = data_capital.drop(columns=['npg', 'birth_rate', 'death_rate', 'gdw'])
fig, ax = plt.subplots(figsize=(10,6))

ax.plot(data_capital['year'][(data['region']=='Moscow')], 

        data_capital['urbanization'][(data['region']=='Moscow')], ':b', label='Moscow')

ax.plot(data_capital['year'][(data['region']=='Moscow Oblast')], 

        data_capital['urbanization'][(data['region']=='Moscow Oblast')], '-g', label='Moscow Oblast');

ax.plot(data_capital['year'][(data['region']=='Saint Petersburg')], 

        data_capital['urbanization'][(data['region']=='Saint Petersburg')], 'o', label='Saint Petersburg')

ax.plot(data_capital['year'][(data['region']=='Leningrad Oblast')], 

        data_capital['urbanization'][(data['region']=='Leningrad Oblast')], label='Leningrad Oblast')

plt.legend();