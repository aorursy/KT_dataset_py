# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/suicide-rates-overview-1985-to-2016/master.csv", thousands=',')
#df.head(10)
df.isnull().sum()
print(df.columns.values)
country_df = df[['suicides_no', 'population', 'country', 'year']].groupby(['country', 'year'], as_index=False).sum()
country_df.reindex(columns=['suicides_no', 'population', 'country', 'year'])
country_df['suicides/100k pop'] = 100000*country_df['suicides_no'] / country_df['population']
country_df.head()
econ_df = df[['country', 'year', 'gdp_per_capita ($)', ' gdp_for_year ($) ']].groupby(['country', 'year'], as_index=False).mean()
econ_df.head()

country_df = pd.merge(country_df, econ_df,  how='left', left_on=['year','country'], right_on = ['year','country'])
country_df.head()
gen_df = df[['suicides_no', 'population', 'generation', 'year']].groupby(['generation', 'year'], as_index=False).sum()
gen_df['suicides/100k pop'] = 100000*gen_df['suicides_no'] / gen_df['population']
gen_df.head()

econ_gen_df = df[['generation', 'year', 'gdp_per_capita ($)', ' gdp_for_year ($) ']].groupby(['generation', 'year'], as_index=False).mean()
#econ_gen_df.head()

gen_df = pd.merge(gen_df, econ_gen_df,  how='left', left_on=['year','generation'], right_on = ['year','generation'])
gen_df.head()
age_df = df[['suicides_no', 'population', 'age', 'year']].groupby(['age', 'year'], as_index=False).sum()
age_df['suicides/100k pop'] = 100000*gen_df['suicides_no'] / gen_df['population']
age_df.head()
econ_age_df = df[['age', 'year', 'gdp_per_capita ($)', ' gdp_for_year ($) ']].groupby(['age', 'year'], as_index=False).mean()
#econ_gen_df.head()

age_df = pd.merge(age_df, econ_age_df,  how='left', left_on=['year','age'], right_on = ['year','age'])
age_df.head()
age_df.corr()
def plot_deaths_per_gdppp(df, str):
    x = df['gdp_per_capita ($)']
    y = df['suicides/100k pop']
    plt.scatter(x,y, label=str)
    plt.xlabel('gdp_per_capita ($)')
    plt.ylabel('suicides/100k pop')

countries =df['country'].unique()    
for country in countries:
    plot_deaths_per_gdppp(country_df[country_df['country']== country],country)
#plt.legend()
#G8 = [ 'United States', 'Canada', 'France', 'Germany', 'Japan', 'Italy', 'Russian Federation', 'United Kingdom' ]
G8 = [ 'United States', 'Canada', 'France', 'Germany', 'Japan', 'Italy', 'United Kingdom' ]
g8_df = country_df[country_df['country'].isin(G8)]
g8_df[['country', 'suicides/100k pop']].boxplot(figsize=(40,10), by='country')

import matplotlib
import matplotlib.pyplot as plt

countries =df['country'].unique()

def plot_deaths_per_year(df, str):
    x = df['year']
    y = df['suicides/100k pop']
    plt.plot(x,y, label=str)
    plt.xlabel('Year')
    plt.ylabel('suicides/100k pop')

for country in countries:
    plot_deaths_per_year(country_df[country_df['country']== country], country)
#plt.legend()

gens =df['generation'].unique()

for gen in gens:
    plot_deaths_per_year(gen_df[gen_df['generation']== gen],gen)
plt.legend()
gen_df[['generation', 'suicides/100k pop']].boxplot(by='generation')
ages =df['age'].unique()
#print(ages)
for age in ages:
    plot_deaths_per_year(age_df[age_df['age']== age], age)
plt.legend()
def divide_two_cols(df_sub):
    return df_sub['suicides_no'].sum() / float(df_sub['population'].sum())

df.groupby('sex').apply(divide_two_cols)

import seaborn as sns
sns.distplot(df['gdp_per_capita ($)']);
plt.figure(figsize=(18,10))
plt.subplot(211)
sns.boxplot(x = country_df["year"],y = country_df["suicides/100k pop"],palette="rainbow")
plt.title("Suicide number ")
plt.show()
sns.distplot(df['suicides/100k pop']);
df['country'].unique()
North_America = {'Antigua and Barbuda', 'Bahamas', 'Barbados', 'Belize','Canada', 'Cuba', 'Dominica', 'El Salvador', 'Guatemala', 'Jamaica', 'Mexico', 'Nicaragua', 'Panama',
        'Puerto Rico','Saint Kitts and Nevis', 'Saint Lucia',
       'Saint Vincent and Grenadines','Trinidad and Tobago', 'United States'}
Europe  = ['Albania', 'Armenia','Austria',  'Azerbaijan', 'Belarus', 'Belgium','Bosnia and Herzegovina', 'Bulgaria',
        'Costa Rica', 'Croatia', 'Cyprus', 'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France', 'Georgia',
       'Germany', 'Greece', 'Hungary', 'Iceland', 'Ireland', 'Italy', 'Latvia',
       'Lithuania', 'Luxembourg',
       'Malta', 'Montenegro', 'Netherlands', 'Norway', 'Poland', 'Portugal', 'Romania', 'Russian Federation',
        'Serbia', 'San Marino', 'Slovakia', 'Slovenia',
       'Spain', 'Sweden', 'Switzerland',
         'Turkey',  'Ukraine', 'United Kingdom']
Asia ={'Bahrain', 'Israel', 'Japan', 'Kazakhstan', 'Kuwait', 'Kyrgyzstan', 'Macau', 'Maldives', 'Mongolia', 'Oman','Philippines', 'Qatar', 'Republic of Korea','Seychelles',
       'Singapore', 'Sri Lanka','Thailand','Turkmenistan', 'United Arab Emirates','Uzbekistan'}
Africa ={ 'Cabo Verde', 'Mauritius', 'South Africa',}
South_America = {'Argentina', 'Aruba', 'Brazil', 'Chile', 'Colombia', 'Ecuador','Grenada', 'Guyana', 'Paraguay', 'Suriname', 'Uruguay'}
Oceania = {'Australia', 'Fiji', 'Kiribati', 'New Zealand'}

G8 = [ 'United States', 'Canada', 'France', 'Germany', 'Japan', 'Italy', 'Russian Federation', 'United Kingdom' ]

Europe_df = country_df[country_df['country'].isin(Europe)]
Europe_df[['country', 'suicides/100k pop']].boxplot(figsize=(40,10), by='country')
Asia_df = country_df[country_df['country'].isin(Asia)]
Asia_df[['country', 'suicides/100k pop']].boxplot(figsize=(40,10), by='country')
South_America_df = country_df[country_df['country'].isin(South_America)]
South_America_df[['country', 'suicides/100k pop']].boxplot(figsize=(40,10), by='country')
North_America_df = country_df[country_df['country'].isin(North_America)]
North_America_df[['country', 'suicides/100k pop']].boxplot(figsize=(40,10), by='country')
for dataset in df:
    dataset['sex'] = dataset['ex'].map( {'female': 1, 'male': 0} ).astype(int)

df.head()
df.plot.scatter(x='gdp_per_capita ($)', y='suicides/100k pop')
J_df=df[df['country']=='Japan']
Japan_gen_df = J_df[['suicides_no', 'population', 'generation', 'year']].groupby(['generation', 'year'], as_index=False).sum()
Japan_gen_df['suicides/100k pop'] = 100000*gen_df['suicides_no'] / gen_df['population']
Japan_gen_df.head()

gens =Japan_gen_df['generation'].unique()

for gen in gens:
    plot_deaths_per_year(Japan_gen_df[Japan_gen_df['generation']== gen],gen)
plt.legend()
J_df.corr()
