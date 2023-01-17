%matplotlib inline

import pandas as pd

import matplotlib.pyplot as plt

import folium

import numpy as np

import seaborn as sns

import geopandas

import plotly.express as px
#Using pandas to read the csv file and encoding the file to ISO-88590-1

df = pd.read_csv('../input/forest-fires-in-brazil/amazon.csv', encoding = "ISO-8859-1")

df.head()
#creating a dictionary with translations of months

month_map={'Janeiro': 'January', 'Fevereiro': 'February', 'Março': 'March', 'Abril': 'April', 'Maio': 'May',

          'Junho': 'June', 'Julho': 'July', 'Agosto': 'August', 'Setembro': 'September', 'Outubro': 'October',

          'Novembro': 'November', 'Dezembro': 'December'}

#mapping our translated months

df['month']=df['month'].map(month_map)

df.head()
#Creating a pivot to get the total number of fires and the year

pivot1 = pd.pivot_table(df,values="number",index=["year"],aggfunc=np.sum)

pivot1.head()
from matplotlib.pyplot import MaxNLocator, FuncFormatter



plt.figure(figsize=(20,8))



ax = sns.lineplot(x = pivot1.index, y = 'number', data = pivot1, estimator = 'sum', color = 'orange', lw = 3, 

                  err_style = None)



plt.title('Total Fires in Brazil : 1998 - 2017', fontsize = 18)

plt.xlabel('Year', fontsize = 14)

plt.ylabel('Number of Fires', fontsize = 14)



ax.xaxis.set_major_locator(plt.MaxNLocator(19))

ax.set_xlim(1998, 2017)



ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

# Fires increased dramatically in the last 20 years, from 20,000 in 1998 to almost double in 2017. 

# What's also alarming is that there is also an increasing trend in the data, 

# so we can expect even more wildfires in the years to follow.



# 2003 and 2016 had the most wildfires throughout Brazil.
#Creating another pivot to plot the number of fires by month

pivot2 = pd.pivot_table(df,values="number",index=["month"],aggfunc=np.sum)

pivot2.head()
# Sorting by months chronologically

meses = df['month'].unique()

pivot2 = pivot2.reindex(meses)

pivot2
# Total burnings reported in Brazil from 1998 to 2017 by months

plt.figure(figsize=(20, 8))

ax = sns.barplot(x=pivot2.index, y="number", data=pivot2)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

plt.xlabel("Month")

plt.ylabel("Count of fires")

plt.title("Fire vs Month")
# The following analysis can be drawn from the plot:



# 1. February, March, April and May see the lowest number of forest fires

# 2. A sudden spike in June continuing the trend till November

# 3. July, August, October and November are the 4 months where maximum forest fires happen
#Creating another pivot to plot the fires by state

pivot3 = pd.pivot_table(df,values="number",index=["state"],aggfunc=np.sum)

pivot3.head()
#Plotting the graph

plt.figure(figsize=(20, 6))

ax = sns.barplot(x=pivot3.index, y="number", data=pivot3)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

plt.xlabel("State")

plt.ylabel("Count of fires")

plt.title("Total burnings reported in Brazil from 1998 to 2017 by States")
# The following analysis can be drawn from the plot:



# 1. Mato Grosso see a huge number of forest fires

# 2. Sergipe, Distrito Federal, Alagoas and Espirito Santo see the lowest number of forest fires
data_pivot = df.pivot_table(values='number', index='year', columns='month', aggfunc=np.sum)

data_pivot = data_pivot.loc[:,['January', 'Feburary', 'March', 'April', 'May', 'June', 'July','August','September', 'October', 'November', 'December']]



plt.figure(figsize=(15,8))

sns.heatmap(data_pivot, linewidths=0.05, vmax=9000, cmap='Oranges', fmt="1.0f", annot=True)

plt.title('Heatmap of number of fires in states in every month in years', fontsize=15)

plt.xlabel('Month')

plt.ylabel('Year')