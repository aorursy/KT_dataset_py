# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import geopandas as gpd

from shapely import wkt

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.offline as offline

offline.init_notebook_mode()

from plotly import tools





import matplotlib.cm



from mpl_toolkits.basemap import Basemap

from matplotlib.collections import PatchCollection

from matplotlib.colors import Normalize

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Read and load the file to df and set the date column to date type

df=pd.read_csv('/kaggle/input/gun-violence-data/gun-violence-data_01-2013_03-2018.csv',parse_dates =['date'],index_col=['date']

               )
df.info()
df.head()
#double check there are no duplicate indexes

sum(df.index.duplicated())
# Check number of rows

len(df)
# Check for null values

print('Null values in gun dataset: \n')

print(df.isnull().sum())
fig, ax = plt.subplots(figsize=(10,10))

m = Basemap(width=6000000,height=4500000,resolution='c',projection='aea',lat_1=35.,lat_2=45,lon_0=-100,lat_0=40)

m.drawcoastlines()

m.fillcontinents(color='tan',lake_color='lightblue', zorder=0)

plt.title("Geography of guns")

m.drawstates(linewidth=0.5, linestyle='solid', color='k')

m.drawcountries(linewidth=2, linestyle='solid', color='k' )

x, y = m(df['longitude'].tolist(), df['latitude'].tolist()) 

g=sns.scatterplot(x, y, hue='state',s=4, data=df, palette="bright", zorder=10,alpha=0.3)

g.legend_.remove()

#plt.scatter(x,y,marker='o', color='Red',alpha =0.1,s=2)

plt.show()
male_count=df['participant_gender'].str.count('Male')

female_count=df['participant_gender'].str.count('Female')

#Make the data to dataframe for ease of use

female_count = female_count.rename(0).to_frame()

male_count = male_count.rename(0).to_frame()

#Check that the subsets are pandas dataframes
#remove decimal place and print as a string

male_sum=int(male_count[0].sum())

female_sum=int(female_count[0].sum())

print('number of males: ' + str(male_sum))

print('number of females: ' + str(female_sum))
#Make a simple pie chart of the proportion of crimes

x=[male_sum,female_sum]

plt.pie(x)

plt.legend(['Total number of males involved', 'Total number of females involved'],loc ='best')

plt.show()

df['gender_parsed']=df["participant_gender"].fillna('0:Unkown')

df['age_parsed']=df["participant_age"].fillna('0:Unkown')

gen=df['gender_parsed'].iloc[0:5]

age=df['age_parsed'].iloc[0:5]

print(gen)

print(type(gen))
month=df.groupby(by=[df.index.year,df.index.month]).agg('count')['incident_id']

month.plot(x=month.index,y=month)

plt.ylabel('number of incidents')

plt.xlabel('year/month')
month=df['2013']

month2013=month.groupby(by=month.index.month).agg('count')['incident_id']

month2013.plot(x=month2013.index,y=month2013)

plt.ylabel('number of incidents')

plt.xlabel('year/month')
#drop 2013

df=df[df.index.year!=2013]

df.index.year.unique()

no2013=df.groupby(by=[df.index.year,df.index.month]).agg('count')['incident_id']

no2013.plot(x=no2013.index,y=no2013)

plt.ylim(bottom=0) 

plt.ylabel('Incidents')

plt.xlabel('Year,Month')

plt.show()

sumkilled=df.groupby(by=[df.index.year,df.index.month,'state']).agg(['sum'])['n_killed']

print(sumkilled)
m=df.groupby(df.index.year).agg('count')

x_val=m.index

y_val=m.incident_id

plt.bar(x=m.index,height=m.incident_id)

plt.title('Incidents by year')

#only part of 2018 is shown
def separate(df):

    df=df.split("||")

    df=[(x.split("::")) for x in df]

    y = []

    for  i in range (0, len(df)):

        y.append(df[i][-1])

    return(y)



df['participant_gender'] = df['participant_gender'].fillna("0::Zero")

df['gender'] = df['participant_gender'].apply(lambda x: separate(x))

df['Males'] = df['gender'].apply(lambda x: x.count('Male'))

df['Females'] = df['gender'].apply(lambda x: x.count('Female'))





dx=df[['state', 'Males', 'Females']].groupby('state').sum()



dx


trace1 = go.Bar(

    x=dx.index,

    y=dx['Males'],

    name='Males'

)

trace2 = go.Bar(

    x=dx.index,

    y=dx['Females'],

    name='Females'

)



data = [trace1, trace2]

layout = go.Layout(

    barmode='group',

    title="Gender Ratio of Shooters"

)



fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='grouped-bar')
df['gun_type_parsed'] = df['gun_type'].fillna('0:Unknown')

gt = df.groupby(by=['gun_type_parsed']).agg({'n_killed': 'sum', 'n_injured' : 'sum', 'state' : 'count'}).reset_index().rename(columns={'state':'count'})

gt.head()



results = {}

for i, each in gt.iterrows():

    wrds = each['gun_type_parsed'].split("||")

    for wrd in wrds:

        if "Unknown" in wrd:

            continue

        wrd = wrd.replace("::",":").replace("|1","")

        gtype = wrd.split(":")[1]

        if gtype not in results: 

            results[gtype] = {'killed' : 0, 'injured' : 0, 'used' : 0}

        results[gtype]['killed'] += each['n_killed']

        results[gtype]['injured'] +=  each['n_injured']

        results[gtype]['used'] +=  each['count']



gun_names = list(results.keys())

used = [each['used'] for each in list(results.values())]

killed = [each['killed'] for each in list(results.values())]

injured = [each['injured'] for each in list(results.values())]

danger = []

for i, x in enumerate(used):

    danger.append((killed[i] + injured[i]) / x)



trace1 = go.Bar(x=gun_names, y=used, name='SF Zoo', orientation = 'v',

    marker = dict(color = '#EEE8AA', 

        line = dict(color = '#EEE8AA', width = 1) ))

data = [trace1]

layout = dict(height=400, title='Which guns have been used?', legend=dict(orientation="h"));

fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='marker-h-bar')