import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns 

import matplotlib.style as style 

import os 
# victims of rape DataFrame

data = pd.read_csv('../input/20_Victims_of_rape.csv')
data.shape
data.columns
data.head(5)
data.loc[(data['Rape_Cases_Reported'] != data['Victims_of_Rape_Total']),:].shape[0]
data.loc[(data['Rape_Cases_Reported'] != data['Victims_of_Rape_Total']),:]
data.loc[:,'Area_Name'].unique()
data.loc[:,'Subgroup'].unique()
## Q1. How many rape cases have been registered per state overall (for all years put together) ... Good way to visualize this would be horizontal bar graphs since showing this on 

##    conventional bar graphs may make the chart dirty !!! 



state_wise = data.pivot_table(values='Victims_of_Rape_Total',index = 'Area_Name',aggfunc='sum').reset_index()



style.use('ggplot')

plt.figure(figsize=(20, 15))





_ = sns.barplot(x = 'Victims_of_Rape_Total', y = 'Area_Name', data = state_wise,edgecolor = None)



_ = plt.title("Rapes in each state",fontdict={'fontsize':30},pad = 30, color = 'red')
## Q2 : How have the rape cases happenned year on year ? Can we plot a Y-o-Y trend ?



y_o_y = data.pivot_table(values='Victims_of_Rape_Total',index='Year',aggfunc='sum').reset_index()



style.use('ggplot')

plt.figure(figsize=(15, 8))



_ = sns.lineplot(x = 'Year', y = 'Victims_of_Rape_Total' , data = y_o_y, color = 'blue')

_ = plt.title("Rape Victims Year on Year",fontdict={'fontsize':30},pad = 30,color = 'blue')
## Q3 : Which type of rapes are reported more ? Incest rape or other ?



sub = data.pivot_table(index='Subgroup',values='Victims_of_Rape_Total',aggfunc='sum').reset_index()

sub.drop(index = 0, inplace=True)



style.use('ggplot')

plt.figure(figsize=(12, 4))



_ = sns.barplot(x='Subgroup', y = 'Victims_of_Rape_Total', data=sub)

_ = plt.title("Incest vs Other rapes",fontdict={'fontsize':30},pad = 30,color = 'indigo')
## Q4 : Which states have highest / lowest difference between reported and actual cases ?



## We need to create one more column which subtracts Victims_of_Rape_Total from Rape_Cases_Reported 



data['Difference'] = data['Victims_of_Rape_Total'] - data['Rape_Cases_Reported']

diff = data.pivot_table(index="Area_Name", values="Difference", aggfunc='sum').reset_index()



style.use('ggplot')

plt.figure(figsize=(18, 10))



_ = sns.barplot(x='Difference', y = 'Area_Name', data=diff.loc[(diff['Difference'] != 0),:])

_ = plt.title("State wise difference between actual and reported cases",fontdict={'fontsize':15},pad = 8,color = 'orange')
## Q5 : Has any state shown a declining trend of rapes over years ?



## We will create a helper function which will take state as an input and publish a graph of rapes over the years as output !



def state_wise(state,information):

    

    """

    This function publishes year on year graph about rapes for a particular state.

    

    Input :: State name (Area_Name) and information (Segment of reported rapes)

    """

    

    ## Slice information as per inputs provided

    

    _ = sns.lineplot(x = 'Year', y = information, data = data.loc[(data['Area_Name'] == state),:],color = 'indigo')



style.use('ggplot')

plt.figure(figsize=(18, 10))

    

state_wise('Madhya Pradesh','Victims_of_Rape_Total')