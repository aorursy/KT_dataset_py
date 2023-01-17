import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd
df=pd.read_csv('../input/demographic.csv')
df.head()
import seaborn as sns
corr = df.corr()
fig, ax = plt.subplots(figsize=(15, 15))
sns.heatmap(corr, annot=True,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
df.isna().sum()/df.shape[0]
df1=df.drop(['Industry','Inflation','Political_stability','Migration'], axis=1)
df2=df1.dropna()
import matplotlib.pyplot as plt
sns.pairplot(df2)
df['date']=pd.to_numeric(df.date, errors='coerce')
df_2015=df[df['date']==2015]
df_2015.head()
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)
data=dict(type='choropleth',
         locations=df_2015['c_codes'],
         z=df_2015['GNP per Capita'],
         text=df_2015['country'],
         colorbar={'title':'Countries GNP perr capita'})
layout=dict(title='Countries GNP per capita',
           geo=dict(showframe=False,
                   projection={'type':'mercator'}))
choromap=go.Figure(data=[data],layout=layout)
iplot(choromap)
data=dict(type='choropleth',
         locations=df_2015['c_codes'],
         z=df_2015['life_expectancy'],
         text=df_2015['country'],
         colorbar={'title':'Life expectancy'})

layout=dict(title='Life expectancy',
           geo=dict(showframe=False,
                   projection={'type':'mercator'}))
choromap=go.Figure(data=[data],layout=layout)
iplot(choromap)
data=dict(type='choropleth',
         locations=df_2015['c_codes'],
         z=df_2015['Population'],
         text=df_2015['country'],
         colorbar={'title':'Population'})

layout=dict(title='Population',
           geo=dict(showframe=False,
                   projection={'type':'mercator'}))
choromap=go.Figure(data=[data],layout=layout)
iplot(choromap)
data=dict(type='choropleth',
         locations=df_2015['c_codes'],
         z=df_2015['Total_Fertility_Rate'],
         text=df_2015['country'],
         colorbar={'title':'Total_Fertility_Rate'})

layout=dict(title='Total_Fertility_Rate',
           geo=dict(showframe=False,
                   projection={'type':'mercator'}))
choromap=go.Figure(data=[data],layout=layout)
iplot(choromap)
df2=df2.reset_index()
df2=df2.drop(['index'],axis=1)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
df3=df2.drop(['country','c_codes','date'],axis=1)
scaler.fit(df3)
scaled_df=scaler.transform(df3)
df3.head()
X=scaled_df
from sklearn.cluster import KMeans
w=[]
for i in range(1,13):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10, random_state=1)
    kmeans.fit(X)
    w.append(kmeans.inertia_)
plt.plot(range(1,13),w)
kmeans=KMeans(n_clusters=4,init='k-means++',max_iter=300,n_init=10, random_state=1)
y=kmeans.fit_predict(X)
df2['y']=y.tolist()
df2.head()
df2_2015=df2[df2['date']==2015]
df2_2015['c_codes']=df2_2015.country.map(Dict)
df2_2015.head()
data=dict(type='choropleth',
         locations=df2_2015['c_codes'],
         z=df2_2015['y'],
         text=df2_2015['country'],
         colorbar={'title':'Clusters'})

layout=dict(title='Clusters',
           geo=dict(showframe=False,
                   projection={'type':'mercator'}))
choromap=go.Figure(data=[data],layout=layout)
iplot(choromap)
fig, ax = plt.subplots(figsize=(15, 20))
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
sns.scatterplot(x="GNP per Capita", y="Total_Fertility_Rate", data=df2_2015,hue='y',size="y",palette="Set2",legend='full',ax=ax1)
sns.scatterplot(x="GNP per Capita", y="Population", data=df2_2015,hue='y',size="y",palette="Set2",legend='full',ax=ax2)
sns.scatterplot(x="GNP per Capita", y="life_expectancy", data=df2_2015,hue='y',size="y",palette="Set2",legend='full',ax=ax3)
fig, ax = plt.subplots(figsize=(15, 15))
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)

sns.scatterplot(x="GNP per Capita", y="y", data=df2.groupby(['country']).mean(),hue='y',size="y",palette="Set2",ax=ax1)
sns.scatterplot(x="Total_Fertility_Rate", y="y", data=df2.groupby(['country']).mean(),hue='y',size="y",palette="Set2",ax=ax2)
sns.scatterplot(x="life_expectancy", y="y", data=df2.groupby(['country']).mean(),hue='y',size="y",palette="Set2",ax=ax3)
fig, ax = plt.subplots(figsize=(15, 15))
sns.lineplot(x="date", y="Total_Fertility_Rate", data=df2.dropna(),hue='y',palette="Set2")