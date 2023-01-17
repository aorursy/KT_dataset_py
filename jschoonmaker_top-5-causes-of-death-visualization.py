#Importing necessary libraries for data analysis and visualization

import matplotlib.pyplot as plt

import pandas as pd

import os

%matplotlib inline
#Importing and loading data frame top_five, with original data downloaded from data.gov on Jan 10, 2020

df = pd.read_csv('/kaggle/input/top_five.csv')

print(df)
#splitting the dataframe to only show the top five causes of death in 1900, then changing cause and number of deaths

#     columns to lists in order to prepare data for use in pie chart

df_1900 = df[df['Year'] == 1900]

df_1900_cause = df_1900.Cause.tolist()

df_1900_deaths = df_1900['Number of Deaths'].tolist()
#splitting the dataframe to only show the top five causes of death in 1950, changing data to list format

df_1950 = df[df['Year'] == 1950]

df_1950_cause = df_1950.Cause.tolist()

df_1950_deaths = df_1950['Number of Deaths'].tolist()
#splitting the dataframe to only show the top five causes of death in 2000, changing data to list format

df_2000 =df[df['Year'] == 2000]

df_2000_cause = df_2000.Cause.tolist()

df_2000_deaths = df_2000['Number of Deaths'].tolist()
#Settings to be used when plotting the pie charts

explode = (0.1, 0, 0, 0, 0)

colors = ['lightskyblue', 'lightgreen', 'gold', 'lightpink', 'lightcoral']
plt.figure(figsize=(40, 20))



#Pie chart for year 2000

plt.subplot(3, 1, 1)

plt.pie(df_2000_deaths, labels = df_2000_cause, autopct = '%1.1f%%', shadow=True, explode=explode, colors=colors, startangle=60)

plt.title('Top 5 Causes of Death in 2000', fontsize='xx-large', fontweight='bold')

#Pie chart for year 1950

plt.subplot(3, 1, 2)

plt.pie(df_1950_deaths, labels = df_1950_cause, autopct = '%1.1f%%', shadow=True, explode=explode, colors=colors, startangle=60)

plt.title('Top 5 Causes of Death in 1950', fontsize='xx-large', fontweight='bold')

#Pie chart for year 1900

plt.subplot(3, 1, 3)

plt.pie(df_1900_deaths, labels = df_1900_cause, autopct = '%1.1f%%', shadow=True, explode=explode, colors=colors, startangle=60)

plt.title('Top 5 Causes of Death in 1900', fontsize='xx-large', fontweight='bold')

plt.savefig(r'C:\Users\J\Desktop\Projects\DeathPie', bbox_inches='tight')

plt.show()
#Possible data to use for future analysis - total number of deaths associated with 'top five' causes 



#sum of the total number of 'top five' deaths in 1900

total_deaths_1900 = sum(df_1900['Number of Deaths'])



#sum of the total number of 'top five' deaths in 1950

total_deaths_1950 = sum(df_1950['Number of Deaths'])



#sum of the total number of 'top five' deaths in 2000

total_deaths_2000 = sum(df_2000['Number of Deaths'])



#quick plot of total number of 'top five' deaths in reported years

x=[1900, 1950, 2000]

y=[total_deaths_1900, total_deaths_1950, total_deaths_2000]

plt.cla()

plt.bar(x, y, width=20)

plt.xticks(x, ('1900', '1950', '2000'))

plt.title('Total number of Deaths attributed to Top 5 Causes of Death in 1900, 1950, and 2000')

plt.show()