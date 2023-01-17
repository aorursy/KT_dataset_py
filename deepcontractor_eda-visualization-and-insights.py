import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
import plotly.graph_objs as go

import plotly.figure_factory as ff

import plotly.express as px
data = pd.read_csv('../input/us-police-shootings/shootings.csv')
data.head()
data.shape
data.columns
data.isnull().sum()
data['race'].value_counts()
data["date"] = pd.to_datetime(data["date"])

data["weekday"] = data["date"].dt.weekday

data['month'] = data['date'].dt.month

data['month_day'] = data['date'].dt.day

data['year'] = data['date'].dt.year

data['month_year'] = pd.to_datetime(data['date']).dt.to_period('M')

plt.title('Which race was shot the most')

sns.countplot(data = data, x = 'race')
topcity = data.city.value_counts().to_frame().reset_index()

topcity.columns = ['City','Count']

topcity = topcity[0:5]
plt.title('Top city with shooting counts')

sns.barplot(data=topcity,x='City',y='Count')
plt.title('Was the shot person mentally ill?')

sns.countplot(data = data, x = 'signs_of_mental_illness')
plt.title('Threat Level Of the Suspect')

sns.countplot(data = data, x = 'threat_level')
plt.title('Suspect Flee vs Not Flee')

sns.countplot(data = data, x = 'flee')
plt.figure(figsize=(10,6))

plt.title('Suspect Flee vs Not Flee')

sns.countplot(data = data, y = 'arms_category')
plt.title('Manner of Death')

sns.countplot(data = data, x = 'manner_of_death')
plt.figure(figsize=(7,5))

plt.title('Age Distribution')

sns.distplot(data['age'],kde = False)
monthlydeaths = data.groupby(['month_year'])['name'].count().reset_index()

monthlydeaths.columns = ['month_year', 'count']

monthlydeaths['month_year'] = monthlydeaths['month_year'].astype(str)
sns.lineplot(data=monthlydeaths['count'])