import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()

import warnings

import plotly.express as px 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/housing-in-london/housing_in_london_monthly_variables.csv')

data.head()
data.info()
data = data.set_index(pd.to_datetime(data['date']))

data.head()
data['houses_sold'].fillna(data['houses_sold'].mean(), inplace = True)

data['no_of_crimes'].fillna(data['no_of_crimes'].mean(), inplace = True)

data['borough_flag'].fillna(0, inplace =True)
data['houses_sold']=data['houses_sold'].astype('int64')

data['no_of_crimes']=data['no_of_crimes'].astype('int64')
data.drop(['code'], axis=1, inplace=True)
data_borough_flag_1 = data[data['borough_flag']==1]

data_borough_flag_1_mean = data_borough_flag_1.groupby('area').mean().reset_index()

data_borough_flag_1_mean.head()
data_borough_flag_0 = data[data['borough_flag']==0]

data_borough_flag_0_mean = data_borough_flag_0.groupby('area').mean().reset_index()

data_borough_flag_0_mean.head()
data_borough_flag_1.describe()
data_borough_flag_0.describe()
sns.distplot(data_borough_flag_1['average_price']);
sns.distplot(data_borough_flag_0['average_price']);
#skewness and kurtosis for average price to the area of borough flag 1 

print("Skewness for average price to the area of borough flag 1: %f" % data_borough_flag_1['average_price'].skew())

print("Kurtosis for average price to the area of borough flag 1: %f" % data_borough_flag_1['average_price'].kurt())
#skewness and kurtosis for average price to the area of borough flag 0

print("Skewness for average price to the area of borough flag 0: %f" % data_borough_flag_0['average_price'].skew())

print("Kurtosis for average price to the area of borough flag 0: %f" % data_borough_flag_0['average_price'].kurt())
sns.set(style="darkgrid")

g = sns.jointplot('average_price','houses_sold', data=data_borough_flag_1, kind="reg", truncate=False, color="r", height=7)

plt.ylabel('Number of houses sold to the area of borough flag 1', fontsize=13)

plt.xlabel('Average Price', fontsize=13)
sns.set(style="darkgrid")

g = sns.jointplot('average_price','houses_sold', data=data_borough_flag_0, kind='reg', truncate=False, color='b', height=7)

plt.ylabel('Number of houses sold to the area of borough flag 0', fontsize=13)

plt.xlabel('Average Price', fontsize=13)
sns.set(style="darkgrid")

d = sns.jointplot('average_price','no_of_crimes', data=data_borough_flag_1, kind='reg', truncate=False, color='g', height=7)

plt.ylabel('Number of Crimes', fontsize=13)

plt.xlabel('Average Price', fontsize=13)
sns.set(style="darkgrid")

d = sns.jointplot('average_price','no_of_crimes', data=data_borough_flag_0, kind='reg', truncate=False, color='gold', 

                  height=7)

plt.ylabel('Number of Crimes', fontsize=13)

plt.xlabel('Average Price', fontsize=13)
fig = px.box(data_borough_flag_1, x='area', y='average_price')

fig.update_layout(

    template='gridon',

    title='Average Monthly London House Price to the area of borough flag 1',

    xaxis_title='Area',

    yaxis_title='Average Price (£)',

    xaxis_showgrid=False,

    yaxis_showgrid=False

)

fig.show()
fig = px.box(data_borough_flag_0, x='area', y='average_price')

fig.update_layout(

    template='gridon',

    title='Average Monthly London House Price to the area of borough flag 0',

    xaxis_title='Area',

    yaxis_title='Average Price (£)',

    xaxis_showgrid=False,

    yaxis_showgrid=False

)

fig.show()
fig = px.line(data_borough_flag_1, x='date', y='average_price', color='area', line_shape='hv', title='Average Price area over Years')

fig.update_layout(

    template='gridon',

    title='Average Monthly London House Price over Years to the area of borough flag 1',

    xaxis_title='Year',

    yaxis_title='Average Price (£)',

    xaxis_showgrid=False,

    yaxis_showgrid=False

)

# Show plot 

fig.show()
fig = px.line(data_borough_flag_0, x='date', y='average_price', color='area', line_shape='hv')

fig.update_layout(

    template='gridon',

    title='Average Monthly London House Price over Years to the area of borough flag 0',

    xaxis_title='Year',

    yaxis_title='Average Price (£)',

    xaxis_showgrid=False,

    yaxis_showgrid=False

)

# Show plot 

fig.show()
fig = px.scatter(data_borough_flag_1, x="date", y="average_price",size="no_of_crimes", color="area")

fig.update_layout(

    template='plotly_dark',

    title='Number of Crimes & Average price over Years to the area of borough flag 1',

    xaxis_title='Year',

    yaxis_title='Average Price (£)',

    xaxis_showgrid=False,

    yaxis_showgrid=False

)

fig.show()
fig = px.scatter(data_borough_flag_0, x="date", y="average_price",size="no_of_crimes", color="area")

fig.update_layout(

    template='plotly_dark',

    title='Number of Crimes & Average price over Years to the area of borough flag 0',

    xaxis_title='Year',

    yaxis_title='Average Price (£)',

    xaxis_showgrid=False,

    yaxis_showgrid=False

)

fig.show()
fig = px.line(data_borough_flag_1, x="date", y="houses_sold", color="area")

fig.update_layout(

    template='plotly_dark',

    title='Houses sold over Years to the area of borough flag 1',

    xaxis_title='Year',

    yaxis_title='houses sold',

    xaxis_showgrid=False,

    yaxis_showgrid=False

)

fig.show()
fig = px.line(data_borough_flag_0, x="date", y="houses_sold", color="area")

fig.update_layout(

    template='plotly_dark',

    title='Houses sold over Years to the area of borough flag 0',

    xaxis_title='Year',

    yaxis_title='houses sold',

    xaxis_showgrid=False,

    yaxis_showgrid=False

)

fig.show()
data2=pd.read_csv('/kaggle/input/housing-in-london/housing_in_london_yearly_variables.csv')

data2.head()
data2.info()
data2['median_salary'].fillna(data2['median_salary'].mean(), inplace = True)

data2['population_size'].fillna(data2['population_size'].mean(), inplace = True)

data2['number_of_jobs'].fillna(data2['number_of_jobs'].mean(), inplace = True)
data2['median_salary']=data2['median_salary'].astype('int64')

data2['population_size']=data2['population_size'].astype('int64')

data2['number_of_jobs']=data2['number_of_jobs'].astype('int64')
data2 = data2.set_index(pd.to_datetime(data2['date']))

data2.info()
data2.drop(['code','mean_salary','life_satisfaction', 'recycling_pct', 'area_size', 'no_of_houses'], axis=1,inplace=True)
data2.info()
data2_borough_flag_1 = data2[data2['borough_flag']==1]

data2_borough_flag_1_mean = data2_borough_flag_1.groupby('area').mean().reset_index()

data2_borough_flag_1_mean.head()
data2_borough_flag_0 = data2[data2['borough_flag']==0]

data2_borough_flag_0_mean = data2_borough_flag_0.groupby('area').mean().reset_index()

data2_borough_flag_0_mean.head()
sns.distplot(data2_borough_flag_1['median_salary'])
sns.distplot(data2_borough_flag_0['median_salary'])
fig = px.scatter(data2_borough_flag_1, x='date', y='median_salary', color='area', size='median_salary')

fig.update_layout(

    template='plotly_dark',

    title='Average salary over Years to the area of borough flag 1',

    xaxis_title='Year',

    yaxis_title='Average Salary (£)',

    xaxis_showgrid=False,

    yaxis_showgrid=False

)

# Show plot 

fig.show()
fig = px.scatter(data2_borough_flag_0, x='date', y='median_salary', color='area', size='median_salary')

fig.update_layout(

    template='plotly_dark',

    title='Average salary over Years to the area of borough flag 0',

    xaxis_title='Year',

    yaxis_title='Average Salary (£)',

    xaxis_showgrid=False,

    yaxis_showgrid=False

)

# Show plot 

fig.show()
fig = px.pie(data2_borough_flag_1, names='area', values='population_size', color='area')

fig.update_layout(

    template='plotly_white',

    title='Population distribution to area of borough flag 1'

)

fig.show()
fig = px.pie(data2_borough_flag_0, names='area', values='population_size', color='area')

fig.update_layout(

    template='plotly_white',

    title='Population distribution to area of borough flag 0'

)

fig.show()
fig = px.pie(data2_borough_flag_1, names='area', values='number_of_jobs', color='area')

fig.update_layout(

    template='plotly_white',

    title='Number of Jobs distribution to area of borough flag 1'

)

fig.show()
fig = px.pie(data2_borough_flag_0, names='area', values='number_of_jobs', color='area')

fig.update_layout(

    template='plotly_white',

    title='Number of Jobs distribution to area of borough flag 0'

)

fig.show()