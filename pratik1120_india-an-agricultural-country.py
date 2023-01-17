import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go



plt.rcParams['figure.figsize'] = 8, 5

plt.style.use("fivethirtyeight")

pd.options.plotting.backend = "plotly"



data1 = pd.read_csv("../input/agricuture-crops-production-in-india/datafile (1).csv")

data2 = pd.read_csv("../input/agricuture-crops-production-in-india/datafile (2).csv")

data3 = pd.read_csv("../input/agricuture-crops-production-in-india/datafile (3).csv")

data0 = pd.read_csv("../input/agricuture-crops-production-in-india/datafile.csv")

production = pd.read_csv("../input/agricuture-crops-production-in-india/produce.csv")
df = data0.copy()

crops = []

production = []

for i in range(df.shape[0]):

    for _ in range(8):

        crops.append(df['Crop'][i])

    production = production + df.loc[i][1:].tolist()

    

years = df.drop('Crop',axis=1).columns.tolist() * 13



df = pd.DataFrame({'Crop':crops, 'Year': years, 'Production': production})

df = df.dropna().reset_index(drop=True)



fig=px.bar(df,x='Crop', y="Production", animation_frame="Year", 

           animation_group="Crop", color="Crop", hover_name="Crop", range_y=[80,150])

fig.update_layout(title="Production rate of Crops per year")

fig.show()
df = data1[["Crop","Cost of Cultivation (`/Hectare) A2+FL","Cost of Cultivation (`/Hectare) C2"]].copy()

df = df.groupby('Crop')[["Cost of Cultivation (`/Hectare) A2+FL","Cost of Cultivation (`/Hectare) C2"]].mean().reset_index()

fig = df.plot(kind='bar',x='Crop',y=["Cost of Cultivation (`/Hectare) A2+FL","Cost of Cultivation (`/Hectare) C2"],barmode='group')

fig.update_layout(showlegend=False, title='Cost of Cultivation A2+FL vs C2')

fig.show()
df = data1[["State","Cost of Production (`/Quintal) C2"]].copy()

df = df.groupby('State')[["Cost of Production (`/Quintal) C2"]].mean().reset_index()

fig = df.plot(kind='bar',x='State',y="Cost of Production (`/Quintal) C2", color="Cost of Production (`/Quintal) C2")

fig.update_layout(title="State-wise Cost of Production (`/Quintal) C2")

fig.show()
df = data1[["State","Yield (Quintal/ Hectare) "]].copy()

df = df.groupby('State')[["Yield (Quintal/ Hectare) "]].mean().reset_index()

fig = df.plot(kind='bar',x='State',y="Yield (Quintal/ Hectare) ", color="Yield (Quintal/ Hectare) ")

fig.update_layout(title="State-wise Yield (Quintal/ Hectare)")

fig.show()
print('Cost of Cultivation (`/Hectare) Distributions')

sns.violinplot(data=data1[["Cost of Cultivation (`/Hectare) A2+FL", "Cost of Cultivation (`/Hectare) C2"]] , size=8)

plt.show()
data2 = data2.sort_values('Yield 2010-11', ascending=False).reset_index(drop=True)
df = data2[["Crop             ","Production 2006-07","Production 2007-08","Production 2008-09","Production 2009-10","Production 2010-11"]].copy().head(8)



fig = df.drop(0).plot(kind='bar',x='Crop             ', y=["Production 2006-07","Production 2007-08","Production 2008-09","Production 2009-10","Production 2010-11"])

fig.update_layout(title='Production of Crops from 2006 to 2011')
mean_production = []

for i in range(data2.shape[0]):

    mean_production.append((data2.iloc[i,1]+data2.iloc[i,2]+data2.iloc[i,3]+data2.iloc[i,4]+data2.iloc[i,5])/5)

    

data2['mean_production'] = pd.Series(mean_production)

sns.boxplot(data=data2['mean_production'])

plt.title('Distribution of mean production for Crops')

plt.show()
df = data2[["Crop             ","Area 2006-07","Area 2007-08","Area 2008-09","Area 2009-10","Area 2010-11"]].copy().head(8)



fig = df.drop(0).plot(kind='bar',x='Crop             ', y=["Area 2006-07","Area 2007-08","Area 2008-09","Area 2009-10","Area 2010-11"])

fig.update_layout(title='Area of Crops from 2006 to 2011')
mean_area = []

for i in range(data2.shape[0]):

    mean_area.append((data2.iloc[i,6]+data2.iloc[i,7]+data2.iloc[i,8]+data2.iloc[i,9]+data2.iloc[i,10])/5)

    

data2['mean_area'] = pd.Series(mean_area)

sns.boxplot(data=data2['mean_area'])

plt.title('Distribution of mean area for Crops')

plt.show()
fig = data3.Crop.value_counts().reset_index().head(7).plot(kind='bar', x='index', y='Crop', color='Crop')

fig.update_layout(title='Crops with most varieties')
new = data3['Season/ duration in days'].value_counts().reset_index().head(10)

sns.barplot(data=new, x='index', y='Season/ duration in days')

plt.xlabel('Duration')

plt.ylabel('Frequency')

plt.title('Most common durations')

plt.show()
fig = data3['Recommended Zone'].str.split(',').str.len().reset_index()['Recommended Zone'].value_counts().reset_index().plot(kind='bar',x='index',y='Recommended Zone')

fig.update_layout(title='Number of States in which most crops are suitable to plant')

fig.show()