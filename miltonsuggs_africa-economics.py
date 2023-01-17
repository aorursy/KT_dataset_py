# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\import numpy as np 

import pandas as pd 

import os

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

import missingno as msno

import plotly.graph_objects as go

import plotly.figure_factory as ff

from plotly.subplots import make_subplots

import random



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/africa-economic-banking-and-systemic-crisis-data/african_crises.csv')

df.head()
df.info()
df.describe()
df.shape
df['banking_crisis'] = df['banking_crisis'].replace({'crisis': 1, 'no_crisis':0})

df.head()
unique_countries = df['country'].unique()

unique_countries
missing_percentage = df.isna().sum()

missing_percentage
exch_avg = df.groupby(['country'])[['exch_usd']].agg('mean').sort_values('exch_usd').reset_index()

exch_avg
fig = go.Figure(data=[go.Bar(x=exch_avg['country'], y=exch_avg['exch_usd'],

                            name='Average Exchange Rate', marker_color='blue')])



fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)

fig.update_layout(title_text='Average Exchange Rate by Country',xaxis_title='Countries',

                 yaxis_title='Average Exchange Rate', title_x=0.5,barmode='stack')

fig.show()
infla_avg = df.groupby(['country'])[['inflation_annual_cpi']].agg('mean').sort_values('inflation_annual_cpi').reset_index()

infla_avg
fig = go.Figure(data=[go.Bar(x=infla_avg['country'], y=infla_avg['inflation_annual_cpi'],

                            name='Average Exchange Rate', marker_color='blue')])



fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)

fig.update_layout(title_text='Average Inflation Rate by Country',xaxis_title='Countries',

                 yaxis_title='Average Inflation Rate', title_x=0.5,barmode='stack')

fig.show()
df.corr()
plt.figure(figsize = (8,6))

sns.heatmap(df.corr())
sns.set(style='whitegrid')

plt.figure(figsize=(20,35))

plt.title('Exchange Rates of Countries')

plot_number=1



for country in unique_countries:

    plt.subplot(7,2,plot_number)

    plot_number+=1

    color ="#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])

    

    plt.scatter(df[df.country==country]['year'],

                df[df.country==country]['exch_usd'],

                color=color,

                s=20)

    

    sns.lineplot(df[df.country==country]['year'],

                 df[df.country==country]['exch_usd'],

                 label=country,

                 color=color)

    

    plt.plot([np.min(df[np.logical_and(df.country==country,df.independence==1)]['year']),

              np.min(df[np.logical_and(df.country==country,df.independence==1)]['year'])],

             [0, np.max(df[df.country==country]['exch_usd'])],

             color='black',

             linestyle='dotted',

             alpha=0.8)

    

    plt.text(np.min(df[np.logical_and(df.country==country,df.independence==1)]['year']),

             np.max(df[df.country==country]['exch_usd'])/2,

             'Independence',

             rotation=-90)

    

    plt.scatter(x=np.min(df[np.logical_and(df.country==country,df.independence==1)]['year']),

                y=0,

                s=50)

    

    plt.title(country)

    

plt.tight_layout()

plt.show()

sns.set(style='whitegrid')

plt.figure(figsize=(20,35))

plot_number=1



for country in unique_countries:

    plt.subplot(7,2,plot_number)

    plot_number+=1

    color ="#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])

    

    plt.scatter(df[df.country==country]['year'],

                df[df.country==country]['inflation_annual_cpi'],

                color=color,

                s=20)

    

    sns.lineplot(df[df.country==country]['year'],

                 df[df.country==country]['inflation_annual_cpi'],

                 label=country,

                 color=color)

    

    plt.plot([np.min(df[np.logical_and(df.country==country,df.independence==1)]['year']),

              np.min(df[np.logical_and(df.country==country,df.independence==1)]['year'])],

             [0, np.max(df[df.country==country]['inflation_annual_cpi'])],

             color='black',

             linestyle='dotted',

             alpha=0.8)

    

    plt.text(np.min(df[np.logical_and(df.country==country,df.independence==1)]['year']),

             np.max(df[df.country==country]['inflation_annual_cpi'])/2,

             'Independence',

             rotation=-90)

    

    plt.scatter(x=np.min(df[np.logical_and(df.country==country,df.independence==1)]['year']),

                y=0,

                s=50)

    

    plt.title(country)

    

plt.tight_layout()

plt.show()
sns.set(style='darkgrid')

cols=['systemic_crisis','domestic_debt_in_default','sovereign_external_debt_default','currency_crises','inflation_crises','banking_crisis']

plt.figure(figsize=(20,20))

plot_num=1



for col in cols:

    plt.subplot(3,2,plot_num)

    plot_num+=1

    

    sns.countplot(y=df.country,hue=df[col],palette='rocket')

    

    plt.legend(loc=0)

    plt.title(col)

    

plt.tight_layout()

plt.show()
fig,ax = plt.subplots(figsize=(20,10))



sns.countplot(df['country'],hue=df['banking_crisis'],ax=ax)



plt.xlabel('Countries')

plt.ylabel('Counts')

plt.xticks(rotation=45)
banking = df[['year', 'country', 'systemic_crisis', 'banking_crisis']]

banking = banking[(banking['country'] == 'Central African Republic') | 

                    (banking['country'] == 'Nigeria') | (banking['country'] == 'Zimbabwe')]



plt.figure(figsize=(20,15))

count=1



for country in banking.country.unique():

    plt.subplot(len(banking.country.unique()), 1, count)

    subset = banking[(banking['country'] == country)]

    

    sns.lineplot(subset['year'], subset['banking_crisis'])

    plt.scatter(subset['year'], subset['systemic_crisis'], color='orange', label='Systemic Crisis')

    

    plt.subplots_adjust(hspace=0.6)

    plt.xlabel('Years')

    plt.ylabel('Banking Crisis/Systemic Crisis')

    plt.title(country)

    count+=1
fig,ax = plt.subplots(figsize=(20,10))



sns.countplot(df['country'],hue=df['domestic_debt_in_default'],ax=ax)



plt.xlabel('Countries')

plt.ylabel('Counts')

plt.xticks(rotation=45)
domestic_debt = df[['year', 'country', 'domestic_debt_in_default', 'systemic_crisis']]

domestic_debt = domestic_debt[(domestic_debt['country'] == 'Angola') | (domestic_debt['country'] == 'Zimbabwe')]



plt.figure(figsize=(20,15))

count=1



for country in domestic_debt.country.unique():

    plt.subplot(len(domestic_debt.country.unique()), 1, count)

    subset = domestic_debt[(domestic_debt['country'] == country)]

    

    sns.lineplot(subset['year'], subset['domestic_debt_in_default'])

    plt.scatter(subset['year'], subset['systemic_crisis'], color='orange', label='Systemic Crisis')

    

    plt.subplots_adjust(hspace=0.6)

    plt.xlabel('Years')

    plt.ylabel('Domestic Debt/Systemic Crisis')

    plt.title(country)

    count+=1
fig,ax = plt.subplots(figsize=(20,10))



sns.countplot(df['country'],hue=df['sovereign_external_debt_default'],ax=ax)



plt.xlabel('Countries')

plt.ylabel('Counts')

plt.xticks(rotation=45)
external_debt = df[['year', 'country', 'sovereign_external_debt_default', 'systemic_crisis']]

external_debt = external_debt[(external_debt['country'] == 'Central African Republic') | 

                              (external_debt['country'] == 'Ivory Coast') |

                              (external_debt['country'] == 'Zimbabwe')]



plt.figure(figsize=(20,15))

count=1



for country in external_debt.country.unique():

    plt.subplot(len(external_debt.country.unique()), 1, count)

    subset = external_debt[(external_debt['country'] == country)]

    

    sns.lineplot(subset['year'], subset['sovereign_external_debt_default'])

    plt.scatter(subset['year'], subset['systemic_crisis'], color='orange', label='Systemic Crisis')

    

    plt.subplots_adjust(hspace=0.6)

    plt.xlabel('Years')

    plt.ylabel('External Debt/Systemic Crisis')

    plt.title(country)

    count+=1
fig,ax = plt.subplots(figsize=(20,10))



sns.countplot(df['country'],hue=df['currency_crises'],ax=ax)



plt.xlabel('Countries')

plt.ylabel('Counts')

plt.xticks(rotation=45)
currency = df[['year', 'country', 'currency_crises', 'systemic_crisis']]

currency = currency[(currency['country'] == 'Angola') | 

                              (currency['country'] == 'Zambia') |

                              (currency['country'] == 'Zimbabwe')]

currency = currency.replace(to_replace=2, value=1, regex=False)





plt.figure(figsize=(20,15))

count=1



for country in currency.country.unique():

    plt.subplot(len(currency.country.unique()), 1, count)

    subset = currency[(currency['country'] == country)]

    

    sns.lineplot(subset['year'], subset['currency_crises'])

    plt.scatter(subset['year'], subset['systemic_crisis'], color='orange', label='Systemic Crisis')

    

    plt.subplots_adjust(hspace=0.6)

    plt.xlabel('Years')

    plt.ylabel('Currency Crisis/Systemic Crisis')

    plt.title(country)

    count+=1
fig,ax = plt.subplots(figsize=(20,10))



sns.countplot(df['country'],hue=df['inflation_crises'],ax=ax)



plt.xlabel('Countries')

plt.ylabel('Counts')

plt.xticks(rotation=45)
inflation = df[['year', 'country', 'inflation_crises', 'systemic_crisis']]

inflation = inflation[(inflation['country'] == 'Angola') | 

                              (inflation['country'] == 'Zambia') |

                              (inflation['country'] == 'Zimbabwe')]



plt.figure(figsize=(20,15))

count=1



for country in inflation.country.unique():

    plt.subplot(len(inflation.country.unique()), 1, count)

    subset = inflation[(inflation['country'] == country)]

    

    sns.lineplot(subset['year'], subset['inflation_crises'])

    plt.scatter(subset['year'], subset['systemic_crisis'], color='orange', label='Systemic Crisis')

    

    plt.subplots_adjust(hspace=0.6)

    plt.xlabel('Years')

    plt.ylabel('Inflation Crisis/Systemic Crisis')

    plt.title(country)

    count+=1