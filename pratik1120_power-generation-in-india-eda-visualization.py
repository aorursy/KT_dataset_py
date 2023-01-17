import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly.express as px



plt.rcParams['figure.figsize'] = 8, 5

plt.style.use("fivethirtyeight")

pd.options.plotting.backend = "plotly"



states = pd.read_csv('../input/daily-power-generation-in-india-20172020/State_Region_corrected.csv')

data = pd.read_csv("../input/daily-power-generation-in-india-20172020/file.csv")

data['Date'] = pd.to_datetime(data['Date'])
#changing thermal generation values to numerical values



data['Thermal Generation Actual (in MU)'] = data['Thermal Generation Actual (in MU)'].str.replace(',','').astype('float')

data['Thermal Generation Estimated (in MU)'] = data['Thermal Generation Estimated (in MU)'].str.replace(',','').astype('float')
# description of data



data.describe()
# correlation in the data



sns.heatmap(data.corr())
# checking for null values



plt.title('Missing Values Plot')

sns.barplot(data=data.isnull().sum().reset_index(), y='index', x=0)

plt.ylabel('Variables')

plt.title('Missing Values Plot')

plt.xlabel('Missing value Count')

plt.show()
#filling the missing values



data = data.fillna(0.0)
# dropping duplicate values(if any)



data = data.drop_duplicates()
features = data.columns[2:].tolist()



sns.boxplot(data=data[features[:2]])

plt.title('Thermal Generation Distribution')

plt.show()
sns.boxplot(data=data[features[2:4]])

plt.title('Nuclear Generation Distribution')

plt.show()
sns.boxplot(data=data[features[4:]])

plt.title('Hydro Generation Distribution')

plt.show()
df = data[data['Region']=='Northern']

fig = df.plot(x='Date',y=['Thermal Generation Actual (in MU)', 'Thermal Generation Estimated (in MU)'])

fig.update_layout(title="Thermal Generation in Northern Region",legend_orientation="h")
df = data[data['Region']=='Northern']

fig = df.plot(x='Date',y=['Nuclear Generation Actual (in MU)', 'Nuclear Generation Estimated (in MU)'])

fig.update_layout(title="Nuclear Generation in Northern Region",legend_orientation="h")
df = data[data['Region']=='Northern']

fig = df.plot(x='Date',y=['Hydro Generation Actual (in MU)', 'Hydro Generation Estimated (in MU)'])

fig.update_layout(title="Hydro Generation in Northern Region",legend_orientation="h")
df = data[data['Region']=='Southern']

fig = df.plot(x='Date',y=['Thermal Generation Actual (in MU)', 'Thermal Generation Estimated (in MU)'])

fig.update_layout(title="Thermal Generation in Southern Region",legend_orientation="h")
df = data[data['Region']=='Southern']

fig = df.plot(x='Date',y=['Nuclear Generation Actual (in MU)', 'Nuclear Generation Estimated (in MU)'])

fig.update_layout(title="Nuclear Generation in Southern Region",legend_orientation="h")
df = data[data['Region']=='Southern']

fig = df.plot(x='Date',y=['Hydro Generation Actual (in MU)', 'Hydro Generation Estimated (in MU)'])

fig.update_layout(title="Hydro Generation in Southern Region",legend_orientation="h")
df = states[['Area (km2)', 'Region']].copy()

df['Total_Area'] = df.groupby('Region')['Area (km2)'].transform('sum')

df.drop('Area (km2)', axis=1, inplace=True)

region_areas = df.drop_duplicates()

region_areas = region_areas[region_areas['Region']!='Central'].reset_index(drop=True)
df = data.loc[data['Date'].dt.year==2017, ['Region','Thermal Generation Actual (in MU)','Nuclear Generation Actual (in MU)','Hydro Generation Actual (in MU)']].copy()



df[['Thermal','Nuclear','Hydro']] = df.groupby('Region')[['Thermal Generation Actual (in MU)','Nuclear Generation Actual (in MU)','Hydro Generation Actual (in MU)']].transform('sum')



df.drop(['Thermal Generation Actual (in MU)','Nuclear Generation Actual (in MU)','Hydro Generation Actual (in MU)'], axis=1,inplace=True)



df = df.drop_duplicates().reset_index(drop=True)

df['Area'] = region_areas['Total_Area']

df['Thermal per area'] = df['Thermal']/df['Area']

df['Nuclear per area'] = df['Nuclear']/df['Area']

df['Hydro per area'] = df['Hydro']/df['Area']



fig = df.plot(kind='bar', x='Region', y=['Thermal per area','Nuclear per area','Hydro per area'], barmode='group')

fig.update_layout(title="Power Generation per unit Area in 2017",legend_orientation="h")
df = data.loc[data['Date'].dt.year==2018, ['Region','Thermal Generation Actual (in MU)','Nuclear Generation Actual (in MU)','Hydro Generation Actual (in MU)']].copy()



df[['Thermal','Nuclear','Hydro']] = df.groupby('Region')[['Thermal Generation Actual (in MU)','Nuclear Generation Actual (in MU)','Hydro Generation Actual (in MU)']].transform('sum')



df.drop(['Thermal Generation Actual (in MU)','Nuclear Generation Actual (in MU)','Hydro Generation Actual (in MU)'], axis=1,inplace=True)



df = df.drop_duplicates().reset_index(drop=True)

df['Area'] = region_areas['Total_Area']

df['Thermal per area'] = df['Thermal']/df['Area']

df['Nuclear per area'] = df['Nuclear']/df['Area']

df['Hydro per area'] = df['Hydro']/df['Area']



fig = df.plot(kind='bar', x='Region', y=['Thermal per area','Nuclear per area','Hydro per area'], barmode='group')

fig.update_layout(title="Power Generation per unit Area in 2018",legend_orientation="h")
df = data.loc[data['Date'].dt.year==2019, ['Region','Thermal Generation Actual (in MU)','Nuclear Generation Actual (in MU)','Hydro Generation Actual (in MU)']].copy()



df[['Thermal','Nuclear','Hydro']] = df.groupby('Region')[['Thermal Generation Actual (in MU)','Nuclear Generation Actual (in MU)','Hydro Generation Actual (in MU)']].transform('sum')



df.drop(['Thermal Generation Actual (in MU)','Nuclear Generation Actual (in MU)','Hydro Generation Actual (in MU)'], axis=1,inplace=True)



df = df.drop_duplicates().reset_index(drop=True)

df['Area'] = region_areas['Total_Area']

df['Thermal per area'] = df['Thermal']/df['Area']

df['Nuclear per area'] = df['Nuclear']/df['Area']

df['Hydro per area'] = df['Hydro']/df['Area']



fig = df.plot(kind='bar', x='Region', y=['Thermal per area','Nuclear per area','Hydro per area'], barmode='group')

fig.update_layout(title="Power Generation per unit Area in 2019",legend_orientation="h")
df = data.loc[data['Date'].dt.year==2020, ['Region','Thermal Generation Actual (in MU)','Nuclear Generation Actual (in MU)','Hydro Generation Actual (in MU)']].copy()



df[['Thermal','Nuclear','Hydro']] = df.groupby('Region')[['Thermal Generation Actual (in MU)','Nuclear Generation Actual (in MU)','Hydro Generation Actual (in MU)']].transform('sum')



df.drop(['Thermal Generation Actual (in MU)','Nuclear Generation Actual (in MU)','Hydro Generation Actual (in MU)'], axis=1,inplace=True)



df = df.drop_duplicates().reset_index(drop=True)

df['Area'] = region_areas['Total_Area']

df['Thermal per area'] = df['Thermal']/df['Area']

df['Nuclear per area'] = df['Nuclear']/df['Area']

df['Hydro per area'] = df['Hydro']/df['Area']



fig = df.plot(kind='bar', x='Region', y=['Thermal per area','Nuclear per area','Hydro per area'], barmode='group')

fig.update_layout(title="Power Generation per unit Area in 2020",legend_orientation="h")