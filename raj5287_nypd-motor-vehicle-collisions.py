import pandas as pd

import plotly.plotly as py

import plotly.graph_objs as go

import numpy as np
data = pd.read_csv("../input/nypd-motor-vehicle-collisions.csv")
data.head()
data.describe()
data.info()
#Number of accidents per Borough

data.groupby('BOROUGH').size()
#NUMBER OF PERSONS INJURED per BOROUGH

print(data[["BOROUGH","NUMBER OF PERSONS INJURED"]].groupby("BOROUGH").count())
#NUMBER OF PERSONS Killed per BOROUGH

print(data[["BOROUGH","NUMBER OF PERSONS KILLED"]].groupby("BOROUGH").count())
#NUMBER OF PERSONS Killed per BOROUGH

print(data[["BOROUGH","NUMBER OF PERSONS KILLED"]].groupby("BOROUGH").count())

print(data[["BOROUGH","NUMBER OF PEDESTRIANS KILLED"]].groupby("BOROUGH").count())
#getting the counts of people killed then rearranging the columns

persk_data=data[["BOROUGH","NUMBER OF PERSONS KILLED"]].groupby("BOROUGH").count()

persk_data['BOROUGH'] = persk_data.index

persk_data=persk_data.reset_index(drop=True)

pedk_data=data[["BOROUGH","NUMBER OF PEDESTRIANS KILLED"]].groupby("BOROUGH").count()

pedk_data['BOROUGH'] = pedk_data.index

pedk_data=pedk_data.reset_index(drop=True)

motk_data=data[["BOROUGH","NUMBER OF MOTORIST KILLED"]].groupby("BOROUGH").count()

motk_data['BOROUGH'] = motk_data.index

motk_data=motk_data.reset_index(drop=True)
s1 = persk_data.set_index('BOROUGH')['NUMBER OF PERSONS KILLED']

s2 = pedk_data.set_index('BOROUGH')['NUMBER OF PEDESTRIANS KILLED']

s3 = motk_data.set_index('BOROUGH')['NUMBER OF MOTORIST KILLED']

df1 = (s1+s2+s3).reset_index(name='Total People Killed')

print (df1)
import matplotlib.pyplot as plt



plt.rcdefaults()

fig, ax = plt.subplots()

y_pos= np.arange(df1.shape[0])

y=df1["Total People Killed"].values

names=df1["BOROUGH"].values

ax.barh(y_pos, y, align='center',color='green', ecolor='black')

ax.set_yticks(y_pos)

ax.set_yticklabels(names)

ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('People Killed in Accident')

ax.set_title('Number of People Killed Per Borough')



for i, v in enumerate(y):

    ax.text(v + 3, i + .25, str(v), color='Green', fontweight='bold')

plt.show()
# these two lines are what allow code to show up in a notebook!

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode()



dataPlot = [go.Histogram(histfunc = "sum",x=names,y=y)]



# the layout of figure

layout = dict(title = "Number of Deaths Per Borough",

              xaxis= dict(title= 'Borough',ticklen= 2,zeroline= False))



# create and show figure

fig = dict(data = dataPlot, layout = layout)

iplot(fig)
#delete empty rows

data=data.dropna(how='all')



#reset index to remove missing indexes

data=data.reset_index(drop=True)





#split date from "DATE" column

data['Date']=data['DATE'].apply(lambda x: x[:10])
# parse dates

for i in range(len(data['Date'])):

    if(data['Date'][i].startswith('00')):

        p=data['Date'][i-1].split('-')

        current=data['Date'][i].split('-')

        current[0]=p[0]

        data['Date'][i]="-".join(current)

data['Date'] = pd.to_datetime(data['Date'], format = "%Y-%m-%d")

data['Date'][:5]
# count of accidents per month

acc_per_month = data['Date'].groupby([data.Date.dt.year, data.Date.dt.month]).agg('count') 



# convert to dataframe

acc_per_month = acc_per_month.to_frame()



# move date month from index to column

acc_per_month['date'] = acc_per_month.index



# rename column

acc_per_month = acc_per_month.rename(columns={acc_per_month.columns[0]:"acc"})



# re-parse dates

acc_per_month['date'] = pd.to_datetime(acc_per_month['date'], format="(%Y, %m)")



# remove index

acc_per_month = acc_per_month.reset_index(drop=True)



# get month of meet

acc_per_month['month'] = acc_per_month.date.dt.month
# these two lines are what allow code to show up in a notebook!

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode()



# date on the x axis and accidents on the y axis

dataPlot2 = [go.Scatter(x=acc_per_month.date, y=acc_per_month.acc)]



# the layout of figure

layout = dict(title = "Number of Accidents per Month",

              xaxis= dict(title= 'Accidents',ticklen= 2,zeroline= False))



# create and show figure

fig = dict(data = dataPlot2, layout = layout)

iplot(fig)
def getAccidentsAndPlot(borough):

    data_ = pd.DataFrame()

    acc_per_day = pd.DataFrame()

    #Number of accidents per day per in Brooklyn

    data_ = data[data["BOROUGH"]==borough]

    # count of accidents per day in Brooklyn

    acc_per_day = data_['Date'].groupby([data.Date.dt.year, data.Date.dt.month,data.Date.dt.day]).agg('count') 



    # convert to dataframe

    acc_per_day = acc_per_day.to_frame()



    # move date day from index to column

    acc_per_day['date'] = acc_per_day.index



    # rename column

    acc_per_day = acc_per_day.rename(columns={acc_per_day.columns[0]:"acc"})



    # re-parse dates

    acc_per_day['date'] = pd.to_datetime(acc_per_day['date'], format="(%Y, %m, %d)")



    # remove index

    acc_per_day = acc_per_day.reset_index(drop=True)



    # get day of meet

    acc_per_day['day'] = acc_per_day.date.dt.day



    # date on the x axis and accidents on the y axis

    dataPlot3 = [go.Scatter(x=acc_per_month.date, y=acc_per_month.acc)]



    # the layout of figure

    layout = dict(title = "Number of Accidents per Day",

                  xaxis= dict(title= borough,ticklen= 2,zeroline= False))



    # create and show figure

    fig = dict(data = dataPlot3, layout = layout)

    iplot(fig)
#Number of Accidents Pe Day in Brooklyn

getAccidentsAndPlot("BROOKLYN")
#Number of Accidents Pe Day in MANHATTAN

getAccidentsAndPlot("MANHATTAN")
#Number of Accidents Per Day in Bronx

getAccidentsAndPlot("BRONX")
#Number of Accidents Pe Day in Queens

getAccidentsAndPlot("QUEENS")
#Number of Accidents Pe Day in STATEN ISLAND

getAccidentsAndPlot("STATEN ISLAND")