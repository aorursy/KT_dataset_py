# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

import plotly



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

df=pd.read_csv("../input/crisis-data.csv",sep=",",encoding="utf-8")

df.head(5)
import warnings

warnings.filterwarnings('ignore')
## summary of data set-there are 25 columns and 49567 rows

df.info()
#Check on the % of null values

df.isnull().sum()*100/df.shape[0]
df = df.drop(['CIT Certified Indicator'], axis=1)
###we can drop remaining NaN values

df = df.dropna()
df.isnull().sum()*100/df.shape[0]
df.shape
## understanding type of calls and there proportion

##There are 195 type in initial calls

df['Initial Call Type'].astype('category').value_counts()
#top 10 initial call types

df["Initial Call Type"].value_counts().head(10).plot(kind='pie', title='Initial call type distribution')

plt.grid(which='both')
## There are 207 types of final call type

df['Final Call Type'].astype('category').value_counts()
df['Final Call Type'].value_counts().head(15).plot(kind='pie', title='Final call type distribution')

plt.grid(which='both')
df['Officer Gender'].astype('category').value_counts()
f = plt.figure(figsize=(15,8))

#figure 1 shows that count of male officers are higher than female and none

ax = f.add_subplot(231)

df['Officer Gender'].value_counts().plot(kind='pie', title='Gender Distribution')

plt.grid(which='both')



#figure 2 shows that 911 call type has highest distribution

ax=f.add_subplot(233)

df['Call Type'].value_counts().plot(kind='bar', title='Call type Distribution')

plt.grid(which='both')





#Emergent detention has highest count in the categories of disposion

ax=f.add_subplot(234)

df['Disposition'].value_counts().plot(kind='bar', title='Disposition Distribution')

plt.grid(which='both')





#White officer race has highest count in department

ax=f.add_subplot(235)

df['Officer Race'].value_counts().plot(kind='bar', title='Officer race Distribution')

plt.grid(which='both')

f = plt.figure(figsize=(15,8))

ax = f.add_subplot(131)



df['Precinct'].value_counts().plot(kind='bar', title='Precinct distribution', ax=ax)

plt.grid(which='both')



ax = f.add_subplot(132)



df['Sector'].value_counts().plot(kind='bar', title='Sector distribution', ax=ax)

plt.grid(which='both')



ax=f.add_subplot(133)

df['Beat'].value_counts().plot(kind='bar', title='Beat distribution', ax=ax)

plt.grid(which='both')

print('Data is current as of ', max(df['Reported Date']))
# what range of dates do we have?

print("Min date: {0} | Max date: {1}".format(df['Reported Date'].min(), df['Reported Date'].max()))
dfg = df[['Reported Date','Template ID']].groupby('Reported Date').count()

dfg.rename({'Template ID':'Incidents'},axis=1,inplace=True)
dfg.plot.line(

    figsize=(12,5), 

    colormap='tab20',

    title="Incidents per day",

    legend=False

)
dfp_precinct = pd.pivot_table(

    data=df[['Reported Date', 'Precinct']],

    index='Reported Date',

    columns=['Precinct'],

    aggfunc=len,

)

dfp_precinct.iloc[-30::,:].plot.bar(

    stacked=True, 

    figsize=(12,5), 

    colormap='tab20',

    title="Incidents per day by precint (last 30 days)"

).legend(loc='center left', bbox_to_anchor=(1, 0.5))
import chart_studio.plotly as py

import plotly.graph_objects as go

datar = [

    go.Bar(

        x=dfp_precinct.index.tolist(), 

        y=dfp_precinct[col],

        name=col

    ) for col in dfp_precinct.columns

]



# specify the layout of our figure

layout = dict(

    title = "Number of Incidents Per Day",

    xaxis= dict(

        title= 'Date',

        ticklen= 5,

        zeroline= False,

        rangeselector=dict(

            buttons=list([

                dict(count=1,

                     label='1m',

                     step='month',

                     stepmode='backward'),

                dict(count=6,

                     label='6m',

                     step='month',

                     stepmode='backward'),

                dict(count=1,

                    label='YTD',

                    step='year',

                    stepmode='todate'),

                dict(count=1,

                    label='1y',

                    step='year',

                    stepmode='backward'),

                dict(step='all')

            ])

        ),

        rangeslider=dict(

            visible = True

        ),

        type='date',

    ),

    barmode='stack',

)



# create and show our figure

fig = dict(data = datar, layout = layout)

iplot(fig)
fig, ax = plt.subplots(figsize=(14, 10))

(df[['Template ID', 'Disposition']].groupby(['Disposition'])['Template ID'].nunique().sort_values()).plot.barh(ax=ax)

plt.title('Events by Disposition')

plt.show()
precinct = df["Officer Precinct Desc"].value_counts(sort=True, ascending=False)

top_5_precinct = precinct.head(5)

top_5_precinct.plot(kind = 'bar')