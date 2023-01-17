
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import folium
import matplotlib.ticker as ticker
import scipy.interpolate as ip

import matplotlib.pyplot as plt

import calendar
import datetime


df_my_data = pd.read_csv("../input/file12/VIDEOGAMESDATASET.csv")   
df_my_data.head()  
df_my_data
df_my_data.columns
df_my_data.tail()
df_my_data = df_my_data[df_my_data.Year.notnull()]
df_my_data.Year = df_my_data['Year'].astype(int)
df_my_data = pd.read_csv("../input/file12/VIDEOGAMESDATASET.csv")
df_my_data = df_my_data[df_my_data.Year.notnull()]
df_my_data.Year = df_my_data['Year'].astype(int)
df_my_data = df_my_data.groupby(['Year'])
df_my_data = df_my_data['NA_Sales','EU_Sales','JP_Sales','Other_Sales'].sum()
df_my_data.plot(kind='bar',figsize=(15,8))
plt.title("Video game sales by year")
plt.xlabel("Year")
plt.ylabel("Total sales")
plt.legend(['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'], loc='best', frameon=True, fontsize=14)
plt.show()

df_my_data = pd.read_csv("../input/file12/VIDEOGAMESDATASET.csv")
df_my_data = df_my_data.groupby(['Genre'])
df_my_data = df_my_data['NA_Sales','EU_Sales','JP_Sales','Other_Sales'].sum()
df_my_data.plot(kind='bar',figsize=(15,8))
plt.xlabel('Genre',fontsize=16)
plt.ylabel('Total Sales',fontsize=16)
plt.title('Sum of most selling Games',fontsize=16)
plt.show()
df_my_data = pd.read_csv("../input/file12/VIDEOGAMESDATASET.csv") 
df_my_data = df_my_data.groupby(['Genre'])
df_my_data = df_my_data['NA_Sales','EU_Sales','JP_Sales','Other_Sales'].sum()
df_my_data.head(12)
#Plot Graph
ax = df_my_data.plot(
    kind='line', 
    figsize=(15, 8), 
    legend=True,
    ylim=(0,700),
    fontsize=14
    ,linewidth =4.8
)
plt.xlabel('Genre',fontsize=16)
plt.ylabel('Total Sales',fontsize=16)
plt.title('Sales in difference between countries',fontsize=16)

labels = 'Action', 'Adventure', 'Fighting', 'Misc', 'Platform', 'Puzzle', 'Racing', 'Role-Playing','Shooter', 'Simulation','Sports', 'Strategy' 
sizes = [3316, 1286, 848, 1739, 886, 582, 1249, 1488,1310, 867, 2346, 681 ]
explode = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)  

fig1, ax1 = plt.subplots(figsize=(15,8))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=False, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.legend(['Action', 'Adventure', 'Fighting', 'Misc', 'Platform', 'Puzzle', 'Racing', 'Role-Playing', 'Shooter', 'Simulation', 'Sports', 'Strategy'], loc='best', frameon=True, fontsize=14)
plt.show()
df_my_data = pd.read_csv("../input/file12/VIDEOGAMESDATASET.csv")
plt.figure(figsize=(15,8))
sns.lineplot(df_my_data.Platform, df_my_data.Global_Sales)
df_my_data.head(12)
plt.title("Platform used by most players")
plt.xlabel("Platform")
plt.legend(['Total'], loc='best', frameon=True, fontsize=10)
plt.ylabel("Total sales")
plt.show()
df_my_data = pd.read_csv("../input/file12/VIDEOGAMESDATASET.csv")   
df_my_data = df_my_data.groupby(['Platform'])

df_my_data = df_my_data['NA_Sales','EU_Sales','JP_Sales','Other_Sales'].sum()
df_my_data.head(12)
#Plot Graph
ax = df_my_data.plot(
    kind='bar', 
    figsize=(15, 8), 
    legend=True,
    ylim=(0,700),
    fontsize=14
    ,linewidth =4.8
)
plt.xlabel('Platform',fontsize=16)
plt.ylabel('Total Sales',fontsize=16)
plt.title('Sales difference between each platform',fontsize=16)
df_my_data = pd.read_csv("../input/file12/VIDEOGAMESDATASET.csv") 
df_my_data = df_my_data[df_my_data.Year.notnull()]
df_my_data.Year = df_my_data['Year'].astype(int)
df_my_data = df_my_data.groupby('Year')['NA_Sales'].sum()
df_my_data.head(12)
#Plot Graph
ax = df_my_data.plot(
    kind='bar', 
    color=('Purple'),
    figsize=(15, 8), 
    legend=True,
    xlim=(1980, 2020),
    ylim=(0,400),
    fontsize=14
    ,linewidth =4.8
)

plt.title("Sales in North America")
plt.legend(['Total'], loc='best', frameon=True, fontsize=10)
plt.xlabel("Year")
plt.ylabel("sales")
plt.xticks(rotation=90)
plt.grid(b=True, linewidth=0.5, axis='y');
plt.grid(b=True, linewidth=0.5, axis='y');
plt.show()
df_my_data = pd.read_csv("../input/file12/VIDEOGAMESDATASET.csv") 
df_my_data = df_my_data[df_my_data.Year.notnull()]
df_my_data.Year = df_my_data['Year'].astype(int)
df_my_data = df_my_data.groupby('Year')['EU_Sales'].sum()
df_my_data.head(12)
#Plot Graph
ax = df_my_data.plot(
    kind='bar', 
    color=('Blue'),
    figsize=(15, 8), 
    legend=True,
    xlim=(1980, 2020),
    ylim=(0,200),
    fontsize=14
    ,linewidth =4.8
)
plt.title("Sales in Europe")
plt.legend(['Total'], loc='best', frameon=True, fontsize=10)
plt.xlabel("Year")
plt.ylabel("sales")
plt.xticks(rotation=90)
plt.grid(b=True, linewidth=0.5, axis='y');
plt.grid(b=True, linewidth=0.5, axis='y');
plt.show()

df_my_data = pd.read_csv("../input/file12/VIDEOGAMESDATASET.csv")
df_my_data = df_my_data[df_my_data.Year.notnull()]
df_my_data.Year = df_my_data['Year'].astype(int)
df_my_data = df_my_data.groupby('Year')['JP_Sales'].sum()
df_my_data.head(12)
#Plot Graph
ax = df_my_data.plot(
    kind='bar', 
     color=('Red'),
    figsize=(15, 8), 
    legend=True,
    xlim=(1980, 2020),
    ylim=(0,80),
    fontsize=14
    ,linewidth =4.8
)
plt.title("Sales in japan")
plt.legend(['Total'], loc='best', frameon=True, fontsize=10)
plt.xlabel("Year")
plt.ylabel("sales")
plt.xticks(rotation=90)
plt.grid(b=True, linewidth=0.5, axis='y');
plt.grid(b=True, linewidth=0.5, axis='y');
plt.show()
df_my_data = pd.read_csv("../input/file12/VIDEOGAMESDATASET.csv")
df_my_data = df_my_data[df_my_data.Year.notnull()]
df_my_data.Year = df_my_data['Year'].astype(int)
df_my_data = df_my_data.groupby('Year')['Other_Sales'].sum()
df_my_data.head(12)
#Plot Graph
ax = df_my_data.plot(
    kind='bar', 
     color=('Green'),
    figsize=(15, 8), 
    legend=True,
    xlim=(1980, 2020),
    ylim=(0,90),
    fontsize=14
    ,linewidth =4.8
)
plt.title("Sales in other countries")
plt.legend(['Total'], loc='best', frameon=True, fontsize=10)
plt.xlabel("Year")
plt.ylabel("sales")
plt.xticks(rotation=90)
plt.grid(b=True, linewidth=0.5, axis='y');
plt.grid(b=True, linewidth=0.5, axis='y');
plt.show()