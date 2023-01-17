import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import bokeh 

import altair as alt

import ipywidgets as widgets

from IPython.display import clear_output

import folium

import networkx

import datetime

import sklearn

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



#Creating the first dataframe

df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

#This will convert the Column Date into a Datetime

df['ObservationDate'] = pd.to_datetime(df['ObservationDate'])



#Current Totals

confirmedCases = df.groupby('ObservationDate')['Confirmed'].sum().tail(1)[0]

totalDeaths = df.groupby("ObservationDate")['Deaths'].sum().tail(1)[0]

totalRecovered = df.groupby("ObservationDate")['Recovered'].sum().tail(1)[0]

unresolvedCases = confirmedCases - (totalDeaths + totalRecovered)



#Printing Summary of Info

print("Summary Info:")

print("=" * 60)

print("* Confirmed Cases:\t", int(confirmedCases), "Cases")

print("")

print("* Percentage Unresolved:", "%.0f%%" % (unresolvedCases / confirmedCases * 100))

print("* Percentage Recovered:\t", "%.0f%%" % (totalRecovered / confirmedCases * 100))

print("* Percentage Deaths:\t", "%.0f%%" % (totalDeaths / confirmedCases * 100))
#This will create the Drop Down List of all Unique Countries

countryDropDown = widgets.Dropdown(

    options = df['Country/Region'].unique(),

    description = "Options:"

)



#Creates the Button

filterButton = widgets.Button(

    description = "Click Me",

    disabled = False,

    tooltip = "Click Me"

)



#This will display the output of the widget

filterButtonOutput = widgets.Output()



#Function to handle the interaction of Button

def filterButtonClicked(b):

    with filterButtonOutput:

        #Clears Output

        clear_output()

        #Will filter the DF for Country based on value of Drop Down Widget

        selectDF = df[df['Country/Region']==countryDropDown.value]

        del selectDF['SNo']

        

        #Will groupby Date and Gather the last result of the sum for most recent data

        # !! There is probably a better way to do this...

        selectDF.groupby('ObservationDate').sum().tail(1).plot(kind='barh',

                                                               title="Current Status of {}".format(countryDropDown.value))

        #Will display the Bar Graph

        plt.show()



#Activates the function 

filterButton.on_click(filterButtonClicked)



#Will display the Results.

display(widgets.HBox([widgets.VBox([countryDropDown,filterButton]), filterButtonOutput]))
#This will capture the trending CFR measurement of COVID19

totalsByDate = df.groupby("ObservationDate").sum()

#Will perform a Naive calculation of the 

cfrByDateMeasurement = totalsByDate['Deaths']/totalsByDate['Confirmed'] * 100



#Resets the Index 

CFRdf = cfrByDateMeasurement.reset_index()

#Renames the column into CFR

CFRdf.rename(columns = {0: 'CFR'}, inplace=True)



#Plotting the CFR

alt.Chart(CFRdf).mark_line().encode(

    x='ObservationDate:T',

    y='CFR',

    tooltip=['ObservationDate','CFR']

).properties(

    title="Naive CFR: (TotalDeaths / Total Confirmed) * 100"

).interactive()
df