# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd



data= pd.read_csv('../input/crimes-in-boston/crime.csv',encoding="iso-8859-1")



data.info()
data.head()
data[['Lat-1','Long-1']]=data['Location'].str.split(",", expand = True)

data['Lat'] = data['Lat-1'].str[1:]

data['Long'] = data['Long-1'].str[:-1]
data['Lat']=data['Lat'].astype(float)

data['Long']=data['Long'].astype(float)
import datetime

data['OCCURRED_ON_DATE']=pd.to_datetime(data['OCCURRED_ON_DATE'])

data['Date']=data['OCCURRED_ON_DATE'].dt.strftime('%Y-%m-%d')

data['Date']=pd.to_datetime(data['Date'])

data['MONTH'] = data.Date.dt.strftime("%m")
data.isna().any()[lambda x: x] # I wanted to determine which columns have NaN values, then I will examine these columns one by one.
data['DISTRICT'].value_counts() #I will write 'unclear' in nan places.
data['DISTRICT'].fillna('unclear',inplace=True)
data['SHOOTING'].value_counts() # I understand that they wanted to label it as Yes and No, so I'll write N to Nan values.

data['SHOOTING'].fillna('N',inplace=True)
data['UCR_PART'].value_counts() # I will call those who are NaN "uncertain".
data['UCR_PART'].fillna('uncertain',inplace=True)
data['STREET'].value_counts() # I will call those who are NaN "uncertain".
data['STREET'].fillna('uncertain',inplace=True)
data.describe() #there is something wrong with lat and long. 
(data.groupby(["Lat","Long"]).count()[['INCIDENT_NUMBER']]).reset_index()
import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

x=pd.pivot_table(data, index='OFFENSE_CODE_GROUP',values="INCIDENT_NUMBER", aggfunc="count", fill_value=0)

x=x.reset_index()

x = x.reindex(x.sort_values(by=['INCIDENT_NUMBER'], ascending=False).index)

x=x[:10]

fig_dims = (8, 7)

fig, ax = plt.subplots(figsize=fig_dims)

ax = sns.barplot(x="INCIDENT_NUMBER",y="OFFENSE_CODE_GROUP", data=x)



#We can see that the most of the crimes occured named Motor Vehicle Accident Response. Larceny and medical assistance follow it.
x=pd.pivot_table(data, index='YEAR',values="INCIDENT_NUMBER", aggfunc="count", fill_value=0)

x=x.reset_index()

ax = sns.barplot(x="YEAR", y="INCIDENT_NUMBER", data=x)

import altair as alt

x=pd.pivot_table(data, index='MONTH',values="INCIDENT_NUMBER", aggfunc="count", fill_value=0)

x=x.reset_index()

ax = sns.barplot(x="MONTH", y="INCIDENT_NUMBER", data=x)





import altair as alt

x=pd.pivot_table(data, index='DAY_OF_WEEK',values="INCIDENT_NUMBER", aggfunc="count", fill_value=0)

x=x.reset_index()

ax = sns.barplot(x="DAY_OF_WEEK", y="INCIDENT_NUMBER", data=x)



x=pd.pivot_table(data, index='HOUR',values="INCIDENT_NUMBER", aggfunc="count", fill_value=0)

x=x.reset_index()



ax = sns.barplot(x="HOUR", y="INCIDENT_NUMBER", data=x)



data.head()

data['Day']=data.OCCURRED_ON_DATE.dt.day

x=pd.pivot_table(data, index=['YEAR','MONTH','Day'],values="INCIDENT_NUMBER", aggfunc='count', fill_value=0)

x=x.reset_index()

Y=pd.pivot_table(x, index=['YEAR'],values="INCIDENT_NUMBER", aggfunc=np.mean, fill_value=0)

Y=Y.reset_index()

Y.INCIDENT_NUMBER.round(0)





a=pd.pivot_table(data, index=['YEAR','MONTH','Day'],values="INCIDENT_NUMBER", aggfunc='count', fill_value=0)

a=a.reset_index()

b=pd.pivot_table(a, index=['MONTH'],values="INCIDENT_NUMBER", aggfunc=np.mean, fill_value=0)

b=b.reset_index()

b.INCIDENT_NUMBER.round(0)



c=pd.pivot_table(data, index=['YEAR','MONTH','Day','DAY_OF_WEEK'],values="INCIDENT_NUMBER", aggfunc='count', fill_value=0)

c=c.reset_index()

d=pd.pivot_table(c, index=['DAY_OF_WEEK'],values="INCIDENT_NUMBER", aggfunc=np.mean, fill_value=0)

d=d.reset_index()

d.INCIDENT_NUMBER.round(0)



a4_dims = (20, 5)

fig, (ax1, ax2,ax3) = plt.subplots(ncols=3, sharey=True,figsize=a4_dims)



ax = sns.barplot(x="YEAR", y="INCIDENT_NUMBER", data=Y,ax=ax1,color='lightblue')



ax = sns.barplot(x="MONTH", y="INCIDENT_NUMBER", data=b,ax=ax2,color='lightgreen')



ax = sns.barplot(x="DAY_OF_WEEK", y="INCIDENT_NUMBER", data=d,ax=ax3,color='lightseagreen')

x=pd.pivot_table(data, index=['Date','YEAR','MONTH','Day'],values="INCIDENT_NUMBER", aggfunc="count", fill_value=0)

x=x.reset_index()



x['MONTH'] = x['MONTH'].astype(str)

x['YEAR']=x['YEAR'].astype(str)

x['YEARMONTH'] = x['YEAR'].str.cat(x['MONTH'],sep="")



y=pd.pivot_table(x, index=['YEARMONTH'],values="INCIDENT_NUMBER", aggfunc="mean", fill_value=0)

y=y.reset_index()

y['YEARMONTH']=y['YEARMONTH'].astype(str)

y = y.reindex(y.sort_values(by=['YEARMONTH'], ascending=[True]).index)



fig, ax = plt.subplots()

fig.set_size_inches(15, 5)

sns.set(style="darkgrid")

g=sns.lineplot(x="YEARMONTH", y="INCIDENT_NUMBER",data=y)

plt.xticks(rotation=70)

plt.tight_layout()
abc=pd.pivot_table(data, index=['OFFENSE_CODE_GROUP'],values="INCIDENT_NUMBER",  aggfunc='count', fill_value=0)

abc=abc.reset_index()

abc = abc.reindex(abc.sort_values(by='INCIDENT_NUMBER', ascending=False).index)

abc['cumsum'] = abc['INCIDENT_NUMBER'].cumsum()

abc['sum'] = abc['INCIDENT_NUMBER'].sum()

abc['percentage'] =  (abc['cumsum']/abc['sum'])*100



def ABC_segmentation(perc):

   

     if perc > 0 and perc < 80:

        return 'A'

     elif perc >= 80 and perc < 95:

        return 'B'

     elif perc >= 95:

        return 'C'







abc['clustering'] = abc['percentage'].apply(ABC_segmentation)

abc[abc['clustering']=='A']
data.groupby("MONTH")["YEAR"].unique()
x=pd.pivot_table(data, index=['OFFENSE_CODE_GROUP','YEAR','MONTH','Day'],values="INCIDENT_NUMBER", aggfunc="count", fill_value=0)

x=x.reset_index()

Y=pd.pivot_table(x, index='OFFENSE_CODE_GROUP',columns='YEAR',values="INCIDENT_NUMBER", aggfunc=np.mean, fill_value=0)

Y=Y.reset_index()

Y.columns=['OFFENSE_CODE_GROUP','YIL2015','YIL2016','YIL2017','YIL2018']

Y['DEGISIM']=((Y['YIL2017']-Y['YIL2016'])/Y['YIL2016'])*100

Y=Y.replace([np.inf,-np.inf], np.nan)

Y=Y.dropna()

Y = Y.reindex(Y.sort_values(by="DEGISIM", ascending=False).index)

fig_dims = (10, 13)

fig, ax = plt.subplots(figsize=fig_dims)

ax = sns.barplot(x="DEGISIM", y="OFFENSE_CODE_GROUP", data=Y)
hypothesis=data[(data['YEAR']==2016) | (data['YEAR']==2017)]
boxplot=pd.pivot_table(hypothesis, index=['YEAR','MONTH','Day'],values="INCIDENT_NUMBER", aggfunc="count", fill_value=0)

boxplot=boxplot.reset_index()
ax = sns.boxplot(x="YEAR", y="INCIDENT_NUMBER", data=boxplot)
from scipy.stats import mannwhitneyu

test=pd.pivot_table(hypothesis, index=['Day'],columns='YEAR',values="INCIDENT_NUMBER", aggfunc="count", fill_value=0)

test.columns=['YIL2016','YIL2017']

stat, p = mannwhitneyu(test['YIL2016'], test['YIL2017'])

print('Statistics=%.3f, p=%.3f' % (stat, p))

# interpret

alpha = 0.05

if p > alpha:

	print('Same distribution (fail to reject H0)')

else:

	print('Different distribution (reject H0)')
x=pd.pivot_table(data, index='DISTRICT',values="INCIDENT_NUMBER", aggfunc="count", fill_value=0)

x=x.reset_index()

x = x.reindex(x.sort_values(by=['INCIDENT_NUMBER'], ascending=False).index)

ax = sns.barplot(x="DISTRICT", y="INCIDENT_NUMBER", data=x)

#We can see that the crimes are concentrated in the B2 district.
x=pd.pivot_table(data, index='STREET',values="INCIDENT_NUMBER", aggfunc="count", fill_value=0)

x=x.reset_index()

x = x.reindex(x.sort_values(by=['INCIDENT_NUMBER'], ascending=False).index)

x.head()

#We can see that the crimes are concentrated in WASHINGTON ST. Crimes with an uncertain values follow it.
x=pd.pivot_table(data, index='REPORTING_AREA',values="INCIDENT_NUMBER", aggfunc="count", fill_value=0)

x=x.reset_index()

x = x.reindex(x.sort_values(by=['INCIDENT_NUMBER'], ascending=False).index)

x.head()

#There are records with an empty reporting area section.
x=pd.pivot_table(data, index=['DISTRICT','OFFENSE_CODE_GROUP'],values="INCIDENT_NUMBER", aggfunc="count", fill_value=0)

x=x.reset_index()

x['D-O'] = x['DISTRICT'].str.cat(x['OFFENSE_CODE_GROUP'],sep="-")

x = x.reindex(x.sort_values(by=['INCIDENT_NUMBER','DISTRICT'], ascending=False).index)

xvalue=x[:5]

fig_dims = (5, 5)

fig, ax = plt.subplots(figsize=fig_dims)

ax = sns.barplot(x="INCIDENT_NUMBER", y="D-O", data=xvalue)
pivfordis=pd.pivot_table(data, index=['DISTRICT','OFFENSE_CODE_GROUP'],values="INCIDENT_NUMBER", aggfunc="count", fill_value=0)

pivfordis=pivfordis.reset_index()
pivfordisB2=pivfordis[(pivfordis['DISTRICT']== 'B2') ]

pivfordisB2['D-O'] = pivfordisB2['DISTRICT'].str.cat(pivfordisB2['OFFENSE_CODE_GROUP'],sep="-")

pivfordisB2 = pivfordisB2.reindex(pivfordisB2.sort_values(by=['DISTRICT','INCIDENT_NUMBER'], ascending=False).index)

xvalue=pivfordisB2[:5]

fig_dims = (5, 5)

fig, ax = plt.subplots(figsize=fig_dims)

ax = sns.barplot(x="INCIDENT_NUMBER", y="D-O", data=xvalue)
pivfordisC11=pivfordis[(pivfordis['DISTRICT']== 'C11') ]

pivfordisC11['D-O'] = pivfordisC11['DISTRICT'].str.cat(pivfordisC11['OFFENSE_CODE_GROUP'],sep="-")

pivfordisC11 = pivfordisC11.reindex(pivfordisC11.sort_values(by=['DISTRICT','INCIDENT_NUMBER'], ascending=False).index)

xvalue=pivfordisC11[:5]



fig_dims = (5, 5)

fig, ax = plt.subplots(figsize=fig_dims)

ax = sns.barplot(x="INCIDENT_NUMBER", y="D-O", data=xvalue)
pivfordisD4=pivfordis[(pivfordis['DISTRICT']== 'D4') ]

pivfordisD4['D-O'] = pivfordisD4['DISTRICT'].str.cat(pivfordisD4['OFFENSE_CODE_GROUP'],sep="-")

pivfordisD4 = pivfordisD4.reindex(pivfordisD4.sort_values(by=['DISTRICT','INCIDENT_NUMBER'], ascending=False).index)

xvalue=pivfordisD4[:5]

fig_dims = (5, 5)

fig, ax = plt.subplots(figsize=fig_dims)

ax = sns.barplot(x="INCIDENT_NUMBER", y="D-O", data=xvalue)
pivfordisB3=pivfordis[(pivfordis['DISTRICT']== 'B3') ]

pivfordisB3['D-O'] = pivfordisB3['DISTRICT'].str.cat(pivfordisB3['OFFENSE_CODE_GROUP'],sep="-")

pivfordisB3 = pivfordisB3.reindex(pivfordisB3.sort_values(by=['DISTRICT','INCIDENT_NUMBER'], ascending=False).index)

xvalue=pivfordisB3[:5]

fig_dims = (5, 5)

fig, ax = plt.subplots(figsize=fig_dims)

ax = sns.barplot(x="INCIDENT_NUMBER", y="D-O", data=xvalue)
(data.groupby(["Lat","Long"]).count()[['INCIDENT_NUMBER']]).reset_index()
data_folium=data.drop(data[(data['Lat']==-1) & (data['Long']==-1)].index)

data_folium1=data_folium.drop(data_folium[(data_folium['Lat']==0) & (data_folium['Long']==0)].index)
import folium

from folium import plugins

from folium.plugins import HeatMap

crime_map = folium.Map(location=[42.3125,-71.0875], 

                       tiles = "Stamen Terrain",

                      zoom_start = 11)

df_drop=data_folium1.dropna(subset=['Lat', 'Long', 'DISTRICT'])

# Add data for heatmp 

data_heatmap = df_drop[df_drop["YEAR"]==2016]

data_heatmap = df_drop[['Lat','Long']]

data_heatmap = df_drop.dropna(axis=0, subset=['Lat','Long'])

data_heatmap = [[row['Lat'],row['Long']] for index, row in data_heatmap.iterrows()]

HeatMap(data_heatmap, radius=10).add_to(crime_map)



# Plot!

crime_map