# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import scipy.stats

import seaborn as sns

import datetime

from pylab import rcParams
df = pd.read_csv('../input/weatherHistory.csv')

df.head(3)
df.info()
#Categorical variables:

categorical = df.select_dtypes(include = ["object"]).keys()

print(categorical)
#Quantitative variables:

quantitative = df.select_dtypes(include = ["int64","float64"]).keys()

print(quantitative)
#'Formatted Date' transformation:



df['Date'] = pd.to_datetime(df['Formatted Date'])

df['year'] = df['Date'].dt.year

df['month'] = df['Date'].dt.month

df['day'] = df['Date'].dt.day

df['hour'] = df['Date'].dt.hour
df.info()
df[quantitative].describe()
rcParams['figure.figsize'] = 8, 8

df[quantitative].hist()
df=df.drop('Loud Cover',axis=1)
pressure_median = df['Pressure (millibars)'].median()

      

def pressure(x):

    if x==0:

        return x + pressure_median

    else:

        return x

        

df["Pressure (millibars)"] = df.apply(lambda row:pressure(row["Pressure (millibars)"]) , axis = 1)



rcParams['figure.figsize'] = 5, 3

df['Pressure (millibars)'].hist()
rcParams['figure.figsize'] = 8, 5

sns.countplot(y=df['Summary'])
len(df['Summary'].unique()) #How many different 'Summary' categories are there?
summary_freq=pd.crosstab(index=df['Summary'],columns="count")  

summary_freq_rel = summary_freq/summary_freq.sum() 

summary_freq_rel.sort_values('count', ascending=False) #relative frequencies
#new categorical variable:

def cloud_categorizer(row):

   row = str(row).lower()

   category = ""

   if "foggy" in row:

       category = 5

   elif "overcast" in row:

       category = 4

   elif "mostly cloudy" in row:

       category = 3

   elif "partly cloudy" in row:

       category = 2

   elif "clear" in row:

       category = 1

   else:

       category = 0

   return category 



df["cloud (summary)"] = df.apply (lambda row:cloud_categorizer(row["Summary"]) , axis = 1)
rcParams['figure.figsize'] = 5, 5

sns.countplot(df['cloud (summary)']) 
sns.boxplot(x=df['cloud (summary)'], y=df['Visibility (km)']) 
def cloud_categorizer(row):

    row = str(row).lower()

    category = ""

    if "foggy" in row:

        category = 5

    elif "overcast" in row:

        category = 4

    elif "mostly cloudy" in row:

        category = 3

    elif "partly cloudy" in row:

        category = 2

    elif "clear" in row:

        category = 1

    else:

        category = 4 

    return category 



df["cloud (summary)"] = df.apply (lambda row:cloud_categorizer(row["Summary"]) , axis = 1)
ax=sns.countplot(df['cloud (summary)'])

ax.set_xticklabels(('1=Clear', '2=Partly Cloudy', '3=Mostly Cloudy', '4=Overcast', '5=Foggy'))
len(df['Daily Summary'].unique()) #number of categories
daily_summary_freq =pd.crosstab(index=df['Daily Summary'],columns="count") 

daily_summary_freqrel=daily_summary_freq/daily_summary_freq.sum()

daily_summary_freqrel.sort_values('count', ascending=False).head(10)#Show the 10 most common categories
#Let's create a new variable called 'cloud (daily summary)' using the same function we created for 'cloud (summary)'



df["cloud (daily summary)"] = df.apply (lambda row:cloud_categorizer(row["Daily Summary"]) , axis = 1)

rcParams['figure.figsize'] = 8, 5

ax=sns.countplot(df['cloud (daily summary)'])

ax.set_xticklabels(('1=Clear', '2=Partly Cloudy', '3=Mostly Cloudy', '4=Overcast', '5=Foggy'))
#Drawing a heatmap

def facet_heatmap(data, color, **kws):

    values=data.columns.values[3]

    data = data.pivot(index='day', columns='hour', values=values)

    sns.heatmap(data, cmap='coolwarm', **kws)  



#Joining heatmaps of every month in a year 

def weather_calendar(year,weather): #Year= Any year in DataFrame. Weather=Any quantitative variable

    dfyear = df[df['year']==year][['month', 'day', 'hour', weather]]

    vmin=dfyear[weather].min()

    vmax=dfyear[weather].max()

    with sns.plotting_context(font_scale=12):

        g = sns.FacetGrid(dfyear,col="month", col_wrap=3) #One heatmap per month

        g = g.map_dataframe(facet_heatmap,vmin=vmin, vmax=vmax)

        g.set_axis_labels('Hour', 'Day')

        plt.subplots_adjust(top=0.9)

        g.fig.suptitle('%s Calendar. Year: %s.' %(weather, year), fontsize=18)
weather_calendar(2006,'Temperature (C)')

weather_calendar(2008,'Wind Speed (km/h)')