import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot



init_notebook_mode(connected=True) 



import pandas_profiling as pp

import plotly.express as px
df_train = pd.read_csv('/kaggle/input/bigquery-geotab-intersection-congestion/train.csv')

df_test =  pd.read_csv('/kaggle/input/bigquery-geotab-intersection-congestion/test.csv')
df_train.head()
df_train.describe().T
print("Train dataset shape: "+ str(df_train.shape))

print("Test dataset shape:  "+ str(df_test.shape))
f,ax = plt.subplots(figsize=(10,8))

sns.heatmap(df_train.iloc[:,12:].corr(),annot=True, linewidths=.5, fmt='.1f', ax=ax)



plt.show()
train_cities = df_train.iloc[:,-1].value_counts()

test_cities = df_test.iloc[:,-1].value_counts()



f,ax=plt.subplots(1,2,figsize=(13,6))

train_cities.plot(ax=ax[0],color='crimson', kind='bar')

ax[0].set_title('Train Dataset City Counts')



test_cities.plot(ax=ax[1],color='darkmagenta', kind='bar')

ax[1].set_title('Test Dataset City Counts')



plt.show()
Philadelphia = df_train[df_train.loc[:,"City"]=='Philadelphia']

Hours = [x for x in range(0,24)]



Weekend = []

Weekday = []



for i in Hours:

    Weekend.append(

        sum(Philadelphia[(Philadelphia["Hour"]==i) & (Philadelphia["Weekend"]== 1)]["TotalTimeStopped_p80"])/

        sum(Philadelphia[(Philadelphia["Hour"]==i)]["TotalTimeStopped_p80"])*100)

    Weekday.append(

        sum(Philadelphia[(Philadelphia["Hour"]==i) & (Philadelphia["Weekend"]== 0)]["TotalTimeStopped_p80"])/

        sum(Philadelphia[(Philadelphia["Hour"]==i)]["TotalTimeStopped_p80"])*100)



f,ax = plt.subplots(figsize=(12,5))

sns.barplot(x=Hours, y=Weekend, label='Weekend', color='r', alpha = 0.7)

sns.barplot(x=Hours, y=Weekday, label='Weekday', color='b', alpha = 0.4)



ax.set(xlabel='Hour', ylabel='Percentage', title='In Terms Of "Total Time Stopped" Weekend Weekday Percentage In Philadelphia')

ax.legend(loc='upper right',frameon= True)



plt.show()
train_hours = df_train["Hour"].value_counts()

train_hours = train_hours.iloc[np.lexsort([train_hours.index])]

test_hours = df_test["Hour"].value_counts()

test_hours = test_hours.iloc[np.lexsort([test_hours.index])]



f,ax=plt.subplots(1,2,figsize=(13,5))



train_hours.plot(ax=ax[0],color='sandybrown', kind='bar')

ax[0].set_title('Train Dataset Hour Distribution')



test_hours.plot(ax=ax[1],color='sienna', kind='bar')

ax[1].set_title('Test Dataset Hour Distribution')



plt.show()
df_train["Month"].value_counts()

Months = [1, 5, 6, 7, 8, 9, 10, 11, 12]



Weekend = []

Weekday = []



for i in Months:

    Weekend.append(

        sum(Philadelphia[(Philadelphia["Month"]==i) & (Philadelphia["Weekend"]== 1)]["DistanceToFirstStop_p80"])/

        sum(Philadelphia[(Philadelphia["Month"]==i)]["DistanceToFirstStop_p80"])*100)

    Weekday.append(

        sum(Philadelphia[(Philadelphia["Month"]==i) & (Philadelphia["Weekend"]== 0)]["DistanceToFirstStop_p80"])/

        sum(Philadelphia[(Philadelphia["Month"]==i)]["DistanceToFirstStop_p80"])*100)



f,ax = plt.subplots(figsize=(7,4))

sns.barplot(x=Months, y=Weekend, label='Weekend', color='crimson', alpha = 0.7)

sns.barplot(x=Months, y=Weekday, label='Weekday', color='darkmagenta', alpha = 0.4)



ax.set(xlabel='Month', ylabel='Percentage', title='Distance To First Stop Distribution In Philadelphia (Wend/Wday)')

ax.legend(loc='upper right',frameon= True)



plt.show()
train_months = df_train["Month"].value_counts()

train_months = train_months.iloc[np.lexsort([train_months.index])]

test_months = df_test["Month"].value_counts()

test_months = test_months.iloc[np.lexsort([test_months.index])]



f,ax=plt.subplots(1,2,figsize=(12,5))



train_months.plot(ax=ax[0],color='slateblue', kind='bar')

ax[0].set_title('Train Dataset Month Distribution')



test_months.plot(ax=ax[1],color='steelblue', kind='bar')

ax[1].set_title('Test Dataset Month Distribution')



plt.show()
train_months_1_5 = df_train[df_train["Month"].isin([1,5])]["Weekend"].value_counts()

test_months_1_5 = df_test[df_test["Month"].isin([1,5])]["Weekend"].value_counts()





f,ax=plt.subplots(1,2,figsize=(7,4))



train_months_1_5.plot(ax=ax[0],color='rosybrown', kind='bar')

test_months_1_5.plot(ax=ax[1],color='chocolate', kind='bar')

plt.suptitle('1. and 5. Months Weekend Distribution (Train and Test Datasets)')



plt.show()
trace1 = go.Box(

    y=Philadelphia[Philadelphia["Weekend"]== 0]["TimeFromFirstStop_p80"],

    name = 'Weekday',

    marker = dict(color = 'rgb(0,145,119)')

)

trace2 = go.Box(

    y=Philadelphia[Philadelphia["Weekend"]== 1]["TimeFromFirstStop_p80"],

    name = 'Weekend',

    marker = dict(color = 'rgb(255,111,145)')

)



data = [trace1, trace2]

layout = dict(autosize=False, width=500,height=400, 

              title='Time From First Stop at Weekday and Weekend', 

              paper_bgcolor='rgb(243, 243, 243)', 

              plot_bgcolor='rgb(243, 243, 243)', 

              margin=dict(l=40,r=30,b=80,t=100,)

             )



fig = dict(data=data, layout=layout)



iplot(fig)
trace1 = go.Box(

    y=df_train[df_train["City"]== "Atlanta"]["TotalTimeStopped_p80"],

    name = 'Atlanta',

    marker = dict(color = 'rgb(255,111,145)')

)

trace2 = go.Box(

    y=df_train[df_train["City"]== "Boston"]["TotalTimeStopped_p80"],

    name = 'Boston',

    marker = dict(color = 'rgb(214,93,177)')

)

trace3 = go.Box(

    y=df_train[df_train["City"]== "Chicago"]["TotalTimeStopped_p80"],

    name = 'Chicago',

    marker = dict(color = 'rgb(132,94,194)')

)

trace4 = go.Box(

    y=df_train[df_train["City"]== "Philadelphia"]["TotalTimeStopped_p80"],

    name = 'Philadelphia',

    marker = dict(color = 'rgb(0,138,219)')

)



data = [trace1, trace2, trace3, trace4]

layout = dict(autosize=False, width=800,height=600, 

              title='Comparison of City Congestion (Total Time Stopped)', 

              paper_bgcolor='rgb(243, 243, 243)', 

              plot_bgcolor='rgb(243, 243, 243)', 

              margin=dict(l=40,r=30,b=80,t=100,))

fig = dict(data=data, layout=layout)



iplot(fig)


TotalTimeStopped=df_train.groupby(['City','Latitude','Longitude'])['TotalTimeStopped_p80'].mean().reset_index()



fig = px.scatter_mapbox(TotalTimeStopped[TotalTimeStopped["City"]=='Boston'], 

                        lat="Latitude", lon="Longitude",

                        size="TotalTimeStopped_p80",

                        size_max=12,

                        color="TotalTimeStopped_p80", 

                        color_continuous_scale=px.colors.sequential.

                        Inferno, zoom=11

                       )



fig.update_layout(mapbox_style="stamen-terrain")

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})



fig.show()
fig = px.scatter_mapbox(TotalTimeStopped[TotalTimeStopped["City"]=='Philadelphia'], 

                        lat="Latitude", lon="Longitude",

                        size="TotalTimeStopped_p80",

                        size_max=12,

                        color="TotalTimeStopped_p80", 

                        color_continuous_scale=px.colors.sequential.

                        Inferno, zoom=11

                       )



fig.update_layout(mapbox_style="stamen-terrain")

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})



fig.show()