import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import plotly.express as px

from pandas_profiling import ProfileReport

from plotly.offline import iplot

!pip install joypy

import joypy

from sklearn.cluster import KMeans



plt.rcParams['figure.figsize'] = 8, 5

plt.style.use("fivethirtyeight")



data = pd.read_csv('../input/world-bank-youth-unemployment/API_ILO_country_YU.csv')
# describing the data



data.describe(include='all')
# Covariance



data.cov()
# correlation



data.corr()
sns.heatmap(data.corr())

plt.show()
report = ProfileReport(data)
report
#checking for null values



data.isnull().sum()
#dropping duplicates



data = data.drop_duplicates()
px.box(data.drop(['Country Name','Country Code'], axis=1))
asian_countries = ['India', 'China', 'Sri Lanka','Japan','Bangladesh']



df = data[data['Country Name'].isin(asian_countries)].reset_index(drop=True)



plt.figure(figsize=(10,7))

for i in range(df.shape[0]):

    lst = df.iloc[i].tolist()[2:]

    plt.plot([0,1,2,3,4], lst, label=df['Country Name'][i])

    

plt.legend()

plt.show()
african_countries = ['Nigeria', 'Kenya', 'Ghana','Ethiopia','Tanzania']



df = data[data['Country Name'].isin(african_countries)].reset_index(drop=True)



plt.figure(figsize=(10,7))

for i in range(df.shape[0]):

    lst = df.iloc[i].tolist()[2:]

    plt.plot([0,1,2,3,4], lst, label=df['Country Name'][i])

    

plt.legend()

plt.show()
north_american_countries = ['United States', 'Canada', 'Panama','Mexico','Cuba']



df = data[data['Country Name'].isin(north_american_countries)].reset_index(drop=True)



plt.figure(figsize=(10,7))

for i in range(df.shape[0]):

    lst = df.iloc[i].tolist()[2:]

    plt.plot([0,1,2,3,4], lst, label=df['Country Name'][i])

    

plt.legend()

plt.show()
data = pd.read_csv('../input/unemployment-by-county-us/output.csv')

pd.options.plotting.backend = 'plotly'
df = data.loc[:,['County', 'Rate']]

df['maxrating'] = df.groupby('County')['Rate'].transform('max')

df = df.drop('Rate', axis=1).drop_duplicates().sort_values('maxrating', ascending=False).head(6)



df.plot(x='County', y='maxrating', kind='bar', color='maxrating')
df = data.loc[:,['Year', 'County', 'Rate']]

df['meanrating'] = df.groupby([df.Year, df.County])['Rate'].transform('mean')

df = df.drop('Rate', axis=1).drop_duplicates().sort_values('meanrating', ascending=False)

df = df[df['County'].isin(['San Juan County','Starr County','Sioux County','Presidio County','Maverick County'])]

df = df.sort_values('Year')



fig=px.bar(df,x='County', y="meanrating", animation_frame="Year", 

           animation_group="County", color="County", hover_name="County", range_y=[0,45])

fig.show()
df = data.loc[:,['State', 'Rate']]

df['maxrating'] = df.groupby('State')['Rate'].transform('max')

df = df.drop('Rate', axis=1).drop_duplicates().sort_values('maxrating', ascending=False).head(6)



df.plot(x='State', y='maxrating', kind='bar', color='maxrating')
df = data.loc[:,['Year', 'State', 'Rate']]

df['meanrating'] = df.groupby([df.Year, df.State])['Rate'].transform('mean')

df = df.drop('Rate', axis=1).drop_duplicates().sort_values('meanrating', ascending=False)

df = df[df['State'].isin(['Colorado','Texas','North Dakota','Arizona','Michigan'])]

df = df.sort_values('Year')



fig=px.bar(df,x='State', y="meanrating", animation_frame="Year", 

           animation_group="State", color="State", hover_name="State", range_y = [0,15])

fig.show()
df = data.loc[:,['Year', 'Rate']]

df['maxrating'] = df.groupby('Year')['Rate'].transform('max')

df = df.drop('Rate', axis=1).drop_duplicates().sort_values('maxrating', ascending=False).head(6)



df.plot(x='Year', y='maxrating', kind='bar', color='maxrating')