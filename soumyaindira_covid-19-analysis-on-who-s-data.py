# 1.1 Call libraries

%reset -f

# 1.2 For data manipulations

import numpy as np

import pandas as pd

import seaborn as sns

# 1.3 For plotting

import matplotlib.pyplot as plt

import matplotlib

import matplotlib.pyplot

# Install as: conda install -c plotly plotly 

import plotly.express as px

# 1.4 For data processing

from sklearn.preprocessing import StandardScaler

# 1.5 OS related

import os

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
#Load the data frame

covid_df1=pd.read_csv("../input/covid19-who-report/who-situation-reports-covid-19.csv",

                      parse_dates = ['reported_date'])  # especial for date parsing

covid_df1
type(covid_df1['reported_date'].dt)    # Accessor like get()

                            # pandas.core.indexes.accessors.DatetimeProperties



# Extract month from the given date

covid_df1['month'] = covid_df1['reported_date'].dt.month 
covid_df1['month']

#Define the months

def month(x):

    if x==1 :

        return "JANUARY"            

    if x==2:

        return "FEBRUARY"            

    if x==3:

        return "MARCH"           

    if x==4:

        return "APRIL"            

    if x==5:

        return "MAY"

    if x==6:

        return "JUNE"



covid_df1['month'] = covid_df1['month'].map(lambda x : month(x))



covid_df1['month'].head()
#Remove the NaN with zero

covid_df2=covid_df1.fillna(0)
covid_df2
#How fast the cases increased over the months?



plt.figure(figsize= (15,10))

plt.xticks(rotation = 90 ,fontsize = 10)

plt.yticks(fontsize = 15)

plt.xlabel("month",fontsize = 30)

plt.ylabel('confirmed_cases',fontsize = 30)

plt.title("Worldwide Confirmed Cases Over Time" , fontsize = 30)

total_cases = covid_df2.groupby('month')['new_confirmed_cases'].sum().reset_index()

total_cases1=total_cases.sort_values("new_confirmed_cases")

ax = sns.pointplot( x = total_cases1.month ,y = total_cases1.new_confirmed_cases , color = 'g')

ax.set(xlabel='month', ylabel='confirmed_cases')
#Impact on top 5 affected countries



covid_df3=covid_df2.groupby(['reporting_country_territory']).sum()

covid_df4=covid_df3[['new_confirmed_cases','new_total_deaths']].sort_values("new_confirmed_cases",ascending=False).head()

ax=covid_df4[['new_confirmed_cases','new_total_deaths']].plot(kind='bar',title="Compare Cases in top 5 country",figsize=(12,8))

ax.set_xlabel("reporting_country_territory",fontsize=12)

ax.set_ylabel("count")

plt.show()


#Create a new column to calculate mortality rate



covid_df4["Mortality_Rate"]=round(((covid_df4['new_total_deaths'])/(covid_df4['new_confirmed_cases']))*100,2)

covid_df5=covid_df4.sort_values("Mortality_Rate",ascending=False).head(10)
#Mortality rate in top 5 affected countries



plt.figure(figsize= (15,10))

plt.xticks(fontsize = 15)

plt.yticks(fontsize = 15)

plt.xlabel("Mortality_Rate",fontsize = 30)

plt.ylabel('Country',fontsize = 30)

plt.title("Top 5 countries having highest mortality rate" , fontsize = 30)

ax = sns.barplot(x = covid_df5['Mortality_Rate'],y=covid_df5.index)
#Identifying COVID-19 Transmission Type



covid_df6=covid_df2.groupby(['transmission_classification'])['new_confirmed_cases'].sum().reset_index()

covid_df7=covid_df6.replace(0,'Source Unknown')

covid_df8=covid_df7.set_index('transmission_classification')
#Which transmission type is more dangerous

#Plot using pie chart



plot=covid_df8.plot.pie(x=covid_df8.index,y='new_confirmed_cases',figsize=(8,8),colors=['green','yellow','red','blue','orange'])

plt.title('Transmission Classification',fontsize=25)
#Pictorial representation of impact of covid 19

# Using heatmap to show the effect of confirmed cases and total death on top 5 countries

sns.heatmap(covid_df4,cmap = plt.cm.Greens)
#Monthwise analysis of top affected country 



covid_df8=covid_df2[covid_df2['reporting_country_territory']=='Italy']



var1=['new_total_deaths','new_confirmed_cases']

var2=['month']



var3 = [(it1,it2)  for it1 in var1  for it2 in var2]



fig=plt.figure(figsize=(10,10))

for i,j in enumerate(var3):

     plt.subplot(4,1,i+1)

     sns.boxplot(x = j[1], y = j[0], data = covid_df8, notch = True)

#Thank you