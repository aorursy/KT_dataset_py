#lets call all the necessary libraries for the understanding plotting in Python

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

# To plot things in the note book

%matplotlib inline   

import seaborn as sns  # Provide high level interface and draw beautiful plots

import random

x = np.arange(0,50,1)

y = 5 + 2*x

y = y + np.random.normal(0,2,size=len(x))
# Create figure

plt.figure(figsize=(6, 6))

# Create line plot

plt.plot(x, y, label='x vs y')

# Add legend

plt.legend()



# Specify ticks for x- and y-axis

plt.xticks(np.arange(0, 50, 5))

plt.yticks(np.arange(0, max(y), 5))

ax = plt.gca()

# Add labels

plt.xlabel('Quantity X (arbitrary units)')

plt.ylabel('Quantity Y (arbitrary units)')

# Show plot

plt.show()

# Create figure

plt.figure(figsize=(6, 6))

# Create line plot

plt.scatter(x, y, label='x vs y',color='green', s=25, marker='*')

# Add legend

plt.legend()



# Specify ticks for x- and y-axis

plt.xticks(np.arange(0, 50, 5))

plt.yticks(np.arange(0, max(y), 5))

ax = plt.gca()

# Add labels

plt.xlabel('Quantity X (arbitrary units)')

plt.ylabel('Quantity Y (arbitrary units)')

# Show plot

plt.show()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 3),sharey=True, gridspec_kw={'hspace': 0, 'wspace': 0.1})

# if you want to plot one on top of other, then simply change nrows=2, ncols=1

axes[0].plot(x, y)

axes[0].set_xlabel('X')

axes[0].set_ylabel('y')



axes[1].scatter(x, y)

axes[1].set_xlabel('X')
plt.figure(figsize=(10, 6), dpi=50)

# Create scatter plot

sns.regplot(x=x, y=y)

plt.xlabel('Quantity X (arbitrary units)')

plt.ylabel('Quantity Y (arbitrary units)')

sales = pd.read_csv('../input/smartphone-data-sale/smartphone_sales.csv')

sales.head(5)
#lets drop first column

df = sales.drop('Unnamed: 0',axis=1)

df.head()
# Create figure

plt.figure(figsize=(10, 6), dpi=300)

# Create stacked area chart

labels = sales.columns[2:]



plt.stackplot(df['Quarter'].values, df.drop('Quarter',axis=1).T,labels=labels)

# Add legend

plt.legend()

# Add labels and title

plt.xlabel('Quarters')

plt.ylabel('Sales units in thousands')

plt.title('Smartphone sales units')
gaus_data = np.random.normal(0,1,1000)
plt.figure(figsize=(6, 4), dpi=100)

# Create histogram

plt.hist(gaus_data, bins=20,range=(-5,5))

plt.axvline(x=np.mean(gaus_data), color='g')

plt.axvline(x=np.percentile(gaus_data, 16), color='r', linestyle= '--')

plt.axvline(x=np.percentile(gaus_data, 84), color='r', linestyle= '--')

# Add labels and title

plt.xlabel('random variable (x)')

plt.ylabel('Frequency')

plt.title('Plotting data from Gaussian distribution with mean 0 and Sigma 1')



plt.figure(figsize=(6, 4), dpi=100)

# Create histogram

plt.boxplot(gaus_data)

# Add labels and title

ax = plt.gca()

ax.set_xticklabels(['Any name'])

plt.ylabel('X')

plt.title('Plotting Box plot of a random variable X')

# Show plot

plt.show()

#Multy variable Box plot

data = np.random.normal(0,2,size=3000).reshape(1000,3)

coloumn_names = ['X','Y','Z']

df = pd.DataFrame(data, columns=coloumn_names)
# Create figure

plt.figure(figsize=(6, 4), dpi=100)

# Create histogram

plt.boxplot([df.X.values,df.Y.values, df.Z.values])

# Add labels and title

ax = plt.gca()

ax.set_xticklabels(['Group X', 'Group Y', 'Group Z'])

plt.ylabel('Values')

plt.title('Multi Varialbe box plots')
# to make it more beautiful we can also use seaborn library

plt.figure(dpi=100)

# Set style

sns.set_style('whitegrid')

# Create boxplot

sns.boxplot(data=df)

# Despine

sns.despine(left=True, right=True, top=True)



#sns.boxplot(x = 'day', y = 'total_bill', data = tips) you can load data directly from sns library  

# to make it more beautiful we can also use seaborn library

plt.figure(dpi=100)

# Set style

sns.set_style('whitegrid')

# Create boxplot

sns.violinplot(data=df)

# Despine

sns.despine(left=True, right=True, top=True)



#sns.boxplot(x = 'day', y = 'total_bill', data = tips) you can load data directly from sns library
sns.heatmap(df.corr(), cmap=sns.light_palette("orange", as_cmap=True, reverse=True))

plt.title("Checking correlation between variables")

flight_details = pd.read_csv('../input/flight-details/flight_details.csv')
flight_details.head()
pivot_data = flight_details.pivot(index='Months', columns='Years', values='Passengers')
sns.heatmap(pivot_data, cmap=sns.light_palette("orange", as_cmap=True, reverse=True))

plt.title("Flight Passengers from 2001 to 2012")

#plt.xlabel("Years")

#plt.ylabel("Months")

plt.show()
youtube_views = pd.read_csv('../input/youtube-views/youtube.csv')
youtube_views.head()
fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(12, 8), dpi=50)

sns.barplot(x="subs", y="channels", data=youtube_views, palette="Blues_d", ax=ax1)

sns.barplot(x="views", y="channels", data=youtube_views, palette="Blues_d", ax=ax2)
from ipywidgets import interact, widgets
olymic_data = pd.read_csv('../input/olympia2016-data/olympia2016_athletes.csv')
olymic_data.head()
df = olymic_data.groupby(['sex'])["gold", "silver", "bronze"].sum()

df.reset_index(inplace=True)
fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5, 5), dpi=100)

df.plot.bar(x='sex',y=['gold','silver','bronze'], ax=ax1)
def get_random_color():

    return '%06x' % random.randint(0, 0xFFFFFF)
def interact_plots(sex='male',col='weight'):

    col = col

    val = olymic_data[olymic_data['sex']==sex][col].values

    val=val[~np.isnan(val)]

    color = '#'+get_random_color()

    ax = sns.distplot(val,color=color,label=sex)

    ax.set(xlabel=col, ylabel='frequency')

    ax.legend()

    return ax
interact(interact_plots,sex=['male','female'],col=['weight','height'])