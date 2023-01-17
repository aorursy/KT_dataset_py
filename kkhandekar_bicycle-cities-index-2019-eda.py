# Generic

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Visualisation Libraries

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objects as go

import plotly.express as px

plt.style.use('seaborn-darkgrid')



# Warning

import warnings

warnings.filterwarnings("ignore")



# Garbage Collector

import gc



# Priority Queue

import heapq



# Tabulate

from tabulate import tabulate
url = '../input/bicycle-cities-index-2019/bike_usage_2019.csv'

data = pd.read_csv(url, header='infer')
print("Total Records: ", data.shape[0])
# Inspect

data.head()
# Stat Summary

data.describe().transpose()
# Custom Function to get top & last 3 cities and shows the summary and density distribution



def summary(x):

    

    hi = heapq.nlargest(3, zip(data[x],data['City']))    # Top 3 cities 

    lw = heapq.nsmallest(3, zip(data[x],data['City']))   # Last 3 cities

    

    hi_val, hi_cty = zip(*hi)  # Unzip high value list

    lw_val, lw_cty = zip(*lw)  # Unzip low value list

    

    # Define a new list

    high = []

    low = []

    

    # Iterating over unzipped list (high val)

    for i,j in zip(hi_val,hi_cty):

        #print(f'{j} has {i}')

        high.append([j,i])

    

    # Iterating over unzipped list (low val)

    for i,j in zip(lw_val,lw_cty):

        #print(f'{j} has {i}')

        low.append([j,i])

    

    # Print the tabulated results

    print(f'3 Cities With High {x.capitalize()} :\n')

    print(tabulate(high,headers=['City',x.capitalize()],tablefmt="pretty"))

    

    print ()

    

    # Print the tabulated results

    print(f'3 Cities With Low {x.capitalize()} :\n')

    print(tabulate(low,headers=['City',x.capitalize()],tablefmt="pretty"))

    

    



    fig = plt.figure(figsize=(15, 10))

    plt.subplots_adjust(hspace = 0.6)

    sns.set_palette('pastel')

    

    plt.subplot(221)

    ax1 = sns.distplot(data[x], color = 'r')

    plt.title(f'{x.capitalize()} Density Distribution')

    

    plt.subplot(222)

    ax2 = sns.violinplot(x = data[x], palette = 'Accent', split = True)

    plt.title(f'{x.capitalize()} Violinplot')

    

    plt.subplot(223)

    ax2 = sns.boxplot(x=data[x], palette = 'cool', width=0.7, linewidth=0.6)

    plt.title(f'{x.capitalize()} Boxplot')

    

    plt.subplot(224)

    ax3 = sns.kdeplot(data[x], cumulative=True)

    plt.title(f'{x.capitalize()} Cumulative Density Distribution')

    

    plt.show()
# Bicycle Usage

summary('Bicycle Usage')
# Bicycle Roads

summary('Bicycle Roads')
# Bicycle Rental Stations

summary('Bicycle Rental Stations')
# Bicycle Stolen

summary('Bicycle Stolen')
corr = data[['Bicycle Usage', 'Bicycle Roads','Bicycle Rental Stations', 'Bicycle Stolen']].corr()



plt.figure(figsize=(12, 8))

plt.title("Correlation Heatmap", fontsize=20)



ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True, annot=True

)
#Custom Function to Plot Bivariate Analysis of Columns



def bivar(x,y,dataframe):

    #fig = plt.figure(figsize=(15, 10))

    style = dict(size=10, color='gray')

    fig, ax = plt.subplots(figsize=(15, 10))

    sns.set_palette('pastel')

    

    hl = heapq.nlargest(1, zip(dataframe[x], dataframe[y], dataframe['City']))

    hlx, hly, hlc = zip(*hl)

    

    ll = heapq.nsmallest(1, zip(dataframe[x], dataframe[y], dataframe['City']))

    llx, lly, llc = zip(*ll)

    

       

    ax = sns.lineplot(data=dataframe[[x,y]], palette="tab10", linewidth=1.5)

    

    for i,j,k in zip(hlx, hly, hlc):

        ax.text(i,j, k + " << HIGH",**style)

    

    for i,j,k in zip(llx, lly, llc):

        ax.text(i,j, k + " << LOW",**style)

        

        

    plt.title(f'{x.capitalize()} & {y.capitalize()} Line Plot', fontsize=20)

    

   

    plt.show()
#Bivariate Analysis of Bicycle Usage & Bicycle Roads

bivar('Bicycle Usage','Bicycle Roads',data)
#Bivariate Analysis of Bicycle Usage & Bicycle Rental Stations

bivar('Bicycle Usage','Bicycle Rental Stations',data)
#Bivariate Analysis of Bicycle Roads & Bicycle Rental Stations

bivar('Bicycle Roads','Bicycle Rental Stations',data)
#Bivariate Analysis of Bicycle Roads & Bicycle Stolen

bivar('Bicycle Roads','Bicycle Stolen',data)
#Bivariate Analysis of Bicycle Rental Stations & Bicycle Stolen

bivar('Bicycle Rental Stations','Bicycle Stolen',data)