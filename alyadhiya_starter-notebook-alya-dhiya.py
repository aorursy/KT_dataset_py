import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')



%matplotlib inline
df_cov = pd.read_csv('../input/dsc-summer-school-data-visualization-challenge/2019_nCoV_20200121_20200206.csv', infer_datetime_format=True, parse_dates=['Last Update'])

df_cov.head()
df_cov.describe()
sns.set_style('white')
df_cov
df_cov.info()
df_cov.groupby('Country/Region').size()
country=pd.DataFrame(df_cov.groupby('Country/Region')['Confirmed'].sum().to_frame(name='Confirmed')).reset_index()

country
top_country=country.sort_values(by=['Confirmed'],ascending=False,inplace=False).reset_index().head(10).reset_index()

top_country
sns.barplot(x='Confirmed',y='Country/Region', data=top_country, ci=None)

plt.xlabel('Number of Cases')

plt.ylabel('Country')

plt.title('CONFIRMED COVID-19 CASES')
recovered=pd.DataFrame(df_cov.groupby('Country/Region')['Recovered'].sum().to_frame(name='Recovered')).reset_index()

recovered
death=pd.DataFrame(df_cov.groupby('Country/Region')['Death'].sum().to_frame(name='Death')).reset_index()

death
df1=country[['Country/Region','Confirmed']].copy()

df2=recovered[['Recovered']].copy()

df3=death[['Death']].copy()



print(df1.shape,df2.shape,df3.shape)
final=pd.concat([df1,df2,df3],axis=1)

final
top_final=final.sort_values(by=['Confirmed'],ascending=False,inplace=False)[:10]

top_final
sns.barplot(x='Recovered', y='Country/Region',data=top_final, ci=None)

plt.xlabel('Number of Cases')

plt.ylabel('Country')

plt.title('Recovered Patient in COVID-19 CASES')
sns.barplot(x='Death', y='Country/Region',data=top_final, ci=None)

plt.xlabel('Number of Cases')

plt.ylabel('Country')

plt.title('NUMBER OF DEATH IN COVID-19 CASES')