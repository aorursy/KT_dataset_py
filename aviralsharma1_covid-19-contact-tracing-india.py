import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import datetime

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import plot_confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn import metrics

import datetime

column = [ "id" , "government_id" , "diagnosed_date", "age" , "gender" , "city" , "district" , "state" , "nationality" ,"status" , "status_change_date" , 'notes']

covid_df = pd.read_csv("../input/covid19-in-india/IndividualDetails.csv", names = column)

# print(covid_df.head().transpose())

covid_df=covid_df[1:]

# print(covid_df.head().transpose())

no_det = covid_df.query('notes == "Details awaited" ')

det = covid_df.query('notes != "Details awaited" ')

x = len(no_det.index)

y = len(det.index)

print(" Cases with details available" , y)

print(" Cases with no details available" , x)



# Figure 1

ax1 = plt.subplot2grid((2,2),(0,0))

sizes = [ x , y]

labels = ["No contact tracing" , "Some contact/travel info known" ]

explode = [ 0 , 0.05]

plt.pie(sizes, labels =labels , autopct= '%1.1f%%',explode=explode ,shadow= False )

centre_circle = plt.Circle((0,0),0.7,fc= 'white')

fig = plt.gcf()

fig.gca().add_artist(centre_circle)

plt.title("Tracing of Overall Cases" , fontsize = 15)

# plt.show()



# MonthWise Analysis

format_str = '%d/%m/%y'

#dates = datetime.datetime.strptime(covid_df['diagnosed_date'], format_str )

covid_df['diagnosed_date']=pd.to_datetime(covid_df['diagnosed_date'], format = '%d/%m/%Y')



# Defining Months

feb_df = covid_df.query('diagnosed_date.dt.month == 2')

march_df = covid_df.query('diagnosed_date.dt.month == 3')

april_df = covid_df.query('diagnosed_date.dt.month == 4')



# Feb

no_det = feb_df.query('notes == "Details awaited" ')

det = feb_df.query('notes != "Details awaited" ')

x = len(no_det.index)

y = len(det.index)

sizes = [ x , y]

labels = ["No contact tracing in Feb" , "Some contact/travel info known" ]

explode = [ 0 , 0.05]

# plt.subplot(412)

ax1 = plt.subplot2grid((2,2),(0,1))

plt.pie(sizes, autopct= '%1.1f%%',shadow= False )

centre_circle = plt.Circle((0,0),0.7,fc= 'white')

fig = plt.gcf()

fig.gca().add_artist(centre_circle)

plt.title("Tracing of Feb Cases", fontsize = 15)

# plt.show()



# March

no_det = march_df.query('notes == "Details awaited" ')

det = march_df.query('notes != "Details awaited" ')

x = len(no_det.index)

y = len(det.index)

sizes = [ x , y]

labels = ["No contact tracing in March" , "Some contact/travel info known" ]

explode = [ 0 , 0.05]

# plt.subplot(421)

ax1 = plt.subplot2grid((2,2),(1,0))

plt.pie(sizes, autopct= '%1.1f%%',explode=explode ,shadow= False )

centre_circle = plt.Circle((0,0),0.7,fc= 'white')

fig = plt.gcf()

fig.gca().add_artist(centre_circle)

plt.title("Tracing of March Cases", fontsize = 15)

# plt.show()



# April

no_det = april_df.query('notes == "Details awaited" ')

det = april_df.query('notes != "Details awaited" ')

x = len(no_det.index)

y = len(det.index)

sizes = [ x , y]

labels = ["No contact tracing in April" , "Some contact/travel info known" ]

explode = [ 0 , 0.05]

# plt.figure(2)

# plt.subplot(422)

ax1 = plt.subplot2grid((2,2),(1,1))

plt.pie(sizes, labels =labels , autopct= '%1.1f%%',explode=explode ,shadow= False ,textprops={'fontsize': 10})

centre_circle = plt.Circle((0,0),0.7,fc= 'white')

fig = plt.gcf()

fig.gca().add_artist(centre_circle)

plt.title("Tracing of April Cases", fontsize = 15)

plt.show()
