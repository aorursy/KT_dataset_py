import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as ex

import plotly.graph_objs as go

import plotly.figure_factory as ff

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('darkgrid')

e_data = pd.read_csv('/kaggle/input/earthquake-database/database.csv')

e_data.head()
missing = e_data.isna().sum()

missing = missing[missing>0]

missing = missing.reset_index()

tr = go.Bar(x=missing['index'],y=missing[0],name='Missing')

tr2 = go.Bar(x=missing['index'],y=[e_data.shape[0]]*len(missing['index']),name='Total')



data = [tr2,tr]

fig = go.Figure(data=data,layout={'title':'Proportion Of Missing Values In Our Dataset','barmode':'overlay'})

fig.show()
#Tackle Missing Values

e_data['Magnitude Type'] = e_data['Magnitude Type'].fillna(e_data['Magnitude Type'].mode()[0])



missing = e_data.isna().sum()

missing = missing[missing>0]

missing = missing.reset_index()

not_missing = [col for col in e_data.columns if col not in missing['index'].values]
def get_day_of_week(sir):

    return sir.weekday()

def get_month(sir):

    return sir.month

def get_year(sir):

    return sir.year





e_data =e_data[not_missing]

e_data.Date = pd.to_datetime(e_data.Date)



e_data['Day_of_Week'] = e_data.Date.apply(get_day_of_week)

e_data['Month'] = e_data.Date.apply(get_month)

e_data['Year'] = e_data.Date.apply(get_year)

Info = e_data.describe()

Info.loc['kurt'] = e_data.kurt()

Info.loc['skew'] = e_data.skew()

Info
tmp = e_data.groupby(by='Year').count()

tmp = tmp.reset_index()[['Year','Date']]

tmp

fig = ex.line(tmp,x='Year',y='Date')

fig.update_layout(

    title= 'Number Of Earthquakes Over The Years 1965-1966',

    xaxis = dict(

        tickmode = 'linear',

        tick0 = 0.0,

        dtick = 1

    )

)

fig.show()
tmp = e_data.groupby(by='Year').mean()

tmp = tmp.reset_index()[['Year','Magnitude']]

tmp

fig = ex.line(tmp,x='Year',y='Magnitude')

fig.update_layout(

    title= 'Mean Earthquakes Magnitude Over The Years 1965-1966',

    xaxis = dict(

        tickmode = 'linear',

        tick0 = 0.0,

        dtick = 1

    )

)

fig.show()
tmp = e_data.groupby(by='Year').std()

tmp = tmp.reset_index()[['Year','Magnitude']]

tmp

fig = ex.line(tmp,x='Year',y='Magnitude')

fig.update_layout(

    title= 'Earthquake Standard Deviation From The Mean Over The Years 1965-1966, Mean SD Shown With Red Line',

    xaxis = dict(

        tickmode = 'linear',

        tick0 = 0.0,

        dtick = 1

    )

)

fig.add_shape(

        # Line Horizontal

            type="line",

            x0=1965,

            y0=tmp['Magnitude'].mean(),

            x1=2016,

            y1=tmp['Magnitude'].mean(),

            line=dict(

                color="red",

                width=2.5,

                dash="dashdot",

            ),

    )

fig.show()
tmp = e_data[['Year','Day_of_Week']]

tmp=tmp.groupby(by='Year').agg(lambda x:x.value_counts().index[0])

tmp = tmp.reset_index()

days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday',

            'Sunday']

days = {k:days[k] for k in range(0,7)}

tmp['Day_of_Week'] = tmp['Day_of_Week'].replace(days)

fig = ex.pie(tmp,names='Day_of_Week',title='Propotion Of Earthquakes On A Certian Day Of Week Over The Years ')

fig.show()
ex.pie(e_data,names='Status',title='Proportion Of Different Earthquake Statuses',hole=.3)
ex.pie(e_data,names='Source',title='Proportion Of Different Earthquake Sources',hole=.3)
pivot_table = e_data.pivot_table(index='Year',columns='Month',values='Magnitude')

sns.clustermap(pivot_table,annot=True,cmap='coolwarm',col_cluster=False,figsize=(20,13))
plt.figure(figsize=(20,11))

ax = sns.distplot(e_data['Latitude'],label='Latitude')

ax.set_title('Distribution Of Earthquake Latitudes',fontsize=19)

ax.set_ylabel('Density',fontsize=16)

ax.set_xlabel('Latitude',fontsize=16)



plt.show()
plt.figure(figsize=(20,11))

ax = sns.distplot(e_data['Longitude'],label='Longitude',color='teal')

ax.set_title('Distribution Of Earthquake Longitudes',fontsize=19)

ax.set_ylabel('Density',fontsize=16)

ax.set_xlabel('Longitude',fontsize=16)



plt.show()
ex.pie(e_data,names='Type',title='Proportion Of Different Eqrthquake Types In Our Dataset')
plt.figure(figsize=(20,11))

ax = sns.distplot(e_data['Depth'],label='Depth',color='red')

ax.set_title('Distribution Of Earthquake Depths',fontsize=19)

ax.set_ylabel('Density',fontsize=16)

ax.set_xlabel('Depth',fontsize=16)



plt.show()
plt.figure(figsize=(20,11))

ax = sns.distplot(e_data['Magnitude'],label='Magnitude',color='teal')

ax.set_title('Distribution Of Earthquake Magnitudes',fontsize=19)

ax.set_ylabel('Density',fontsize=16)

ax.set_xlabel('Magnitude',fontsize=16)



plt.show()
#Outlier Removal

e_data = e_data[e_data['Depth'] <300]
tmp = e_data.copy()

tmp = tmp[tmp['Magnitude']<=6.1]

tmp = tmp[tmp['Depth']<60]



sns.jointplot(data=tmp,x='Depth',y='Magnitude',kind='kde',cmap='coolwarm',height=12,levels=30)
#Clustring

from sklearn.cluster import KMeans,DBSCAN

KMeans = KMeans(n_clusters=3)



c_data = e_data.copy()

KMeans.fit(e_data[['Magnitude','Depth']])

c_data['Cluster'] = KMeans.labels_



ex.scatter_3d(c_data,x='Longitude',y='Depth',z='Magnitude',color='Cluster',height=900)