# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
filename = "/kaggle/input/videogamesales/vgsales.csv"
df = pd.read_csv(filename) #Create dataframe
df.head() #Show first 5 rows
#Take a look at the non-null count of all the variables
df.info()
df.Genre.unique() #Show all the different genres
#Take a look at each of the variables in more depth
df.describe()
#Histograms of Variables
import seaborn as sns
from matplotlib.pyplot import plot
sns.lmplot(x='Year', y='Global_Sales', data=df, line_kws={'color': 'red'})
plt.ylim(0,40)
plt.xlabel("Year")
plt.ylabel("Global Sales (in Millions of Dollars)")
plt.title("Histogram of Global Sales Over Time")
df.info()
df.isna().sum() #Number of Missing Values for Each Column
df.dropna(inplace=True)
print(df.shape)
df.isna().sum()

df.info()
decade1 = df[(df.Year >= 1980) & (df.Year <= 1989)]
print(decade1.shape)
decade2 = df[(df.Year >= 1990) & (df.Year <= 1999)]
print(decade2.shape)
decade3 = df[(df.Year >= 2000) & (df.Year <= 2009)]
print(decade3.shape)
decade4 = df[(df.Year >= 2010) & (df.Year <= 2019)]
print(decade4.shape)
import seaborn as sns
box_plot = sns.boxplot(x='Genre', y='Global_Sales', data=decade1, width=.5, palette='Blues')
sns.set_context("paper")
sns.set_style('whitegrid')
sns.set(rc={'figure.figsize':(11.7,11.27)})
plt.ylim(0,5)
plt.xlabel("Genre of Video Game with Labeled Median Value")
plt.ylabel("Global Sales (Millions of Dollars)")
plt.title("Global Sales of Games by Genre From 1980 to 1989")

ax = box_plot.axes
lines = ax.get_lines()
categories = ax.get_xticks()

for cat in categories:
    # every 4th line at the interval of 6 is median line
    # 0 -> p25 1 -> p75 2 -> lower whisker 3 -> upper whisker 4 -> p50 5 -> upper extreme value
    y = round(lines[4+cat*6].get_ydata()[0],1) 

    ax.text(
        cat, 
        y, 
        f'{y}', 
        ha='center', 
        va='center', 
        fontweight='bold', 
        size=10,
        color='white',
        bbox=dict(facecolor='#445A64'))

box_plot.figure.tight_layout()
box_plot = sns.boxplot(x='Genre', y='Global_Sales', data=decade2, width=.5, palette='Blues')
sns.set_context("paper")
sns.set_style('whitegrid')
sns.set(rc={'figure.figsize':(11.7,11.28)})
plt.ylim(0,5)
plt.xlabel("Genre of Video Game with Labeled Median Value")
plt.ylabel("Global Sales (Millions of Dollars)")
plt.title("Global Sales of Games by Genre From 1980 to 1989")

ax = box_plot.axes
lines = ax.get_lines()
categories = ax.get_xticks()

for cat in categories:
    # every 4th line at the interval of 6 is median line
    # 0 -> p25 1 -> p75 2 -> lower whisker 3 -> upper whisker 4 -> p50 5 -> upper extreme value
    y = round(lines[4+cat*6].get_ydata()[0],1) 

    ax.text(
        cat, 
        y, 
        f'{y}', 
        ha='center', 
        va='center', 
        fontweight='bold', 
        size=10,
        color='white',
        bbox=dict(facecolor='#445A64'))

box_plot.figure.tight_layout()
box_plot = sns.boxplot(x='Genre', y='Global_Sales', data=decade3, width=.5, palette='Blues')
sns.set_context("paper")
sns.set_style('whitegrid')
sns.set(rc={'figure.figsize':(11.7,11.28)})
plt.ylim(0,5)
plt.xlabel("Genre of Video Game with Labeled Median Value")
plt.ylabel("Global Sales (Millions of Dollars)")
plt.title("Global Sales of Games by Genre From 1980 to 1989")

ax = box_plot.axes
lines = ax.get_lines()
categories = ax.get_xticks()

for cat in categories:
    # every 4th line at the interval of 6 is median line
    # 0 -> p25 1 -> p75 2 -> lower whisker 3 -> upper whisker 4 -> p50 5 -> upper extreme value
    y = round(lines[4+cat*6].get_ydata()[0],1) 

    ax.text(
        cat, 
        y, 
        f'{y}', 
        ha='center', 
        va='center', 
        fontweight='bold', 
        size=10,
        color='white',
        bbox=dict(facecolor='#445A64'))

box_plot.figure.tight_layout()
box_plot = sns.boxplot(x='Genre', y='Global_Sales', data=decade4, width=.5, palette='Blues')
sns.set_context("paper")
sns.set_style('whitegrid')
sns.set(rc={'figure.figsize':(11.7,11.28)})
plt.ylim(0,5)
plt.xlabel("Genre of Video Game with Labeled Median Value")
plt.ylabel("Global Sales (Millions of Dollars)")
plt.title("Global Sales of Games by Genre From 1980 to 1989")

ax = box_plot.axes
lines = ax.get_lines()
categories = ax.get_xticks()

for cat in categories:
    # every 4th line at the interval of 6 is median line
    # 0 -> p25 1 -> p75 2 -> lower whisker 3 -> upper whisker 4 -> p50 5 -> upper extreme value
    y = round(lines[4+cat*6].get_ydata()[0],1) 

    ax.text(
        cat, 
        y, 
        f'{y}', 
        ha='center', 
        va='center', 
        fontweight='bold', 
        size=10,
        color='white',
        bbox=dict(facecolor='#445A64'))

box_plot.figure.tight_layout()
d1pub = decade1[(decade1.Global_Sales >= 1.5)] #Split data up by publishers with global sales above 1.0 global sales
sns.barplot(x='Global_Sales', y='Publisher', data=d1pub)
plt.xlabel("Global Sales (Millions of Dollars)")
plt.ylabel("Publishers")
plt.title("Top Publishers From 1980 to 1989")
d2pub = decade2[(decade2.Global_Sales >= 1.5)]
sns.barplot(x='Global_Sales', y='Publisher', data=d2pub)
plt.xlabel("Global Sales (Millions of Dollars)")
plt.ylabel("Publishers")
plt.title("Top Publishers From 1990 to 1999")
d3pub = decade3[(decade3.Global_Sales >= 1.5)]
sns.barplot(x='Global_Sales', y='Publisher', data=d3pub)
plt.xlabel("Global Sales (Millions of Dollars)")
plt.ylabel("Publishers")
plt.title("Top Publishers From 2000 to 2009")
d4pub = decade4[(decade4.Global_Sales >= 1.5)]
sns.barplot(x='Global_Sales', y='Publisher', data=d4pub)
plt.xlabel("Global Sales (Millions of Dollars)")
plt.ylabel("Publishers")
plt.title("Top Publishers From 2010 to 2016")