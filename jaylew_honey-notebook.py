# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dataset = pd.read_csv('../input/honeyproduction.csv')
dataset.head()
dataset.info()
dataset.head(10)
#Find top producer state each year
top_prod_state = []
top_prod_amount = []
for year in dataset['year'].unique():
    top_state = list(dataset[dataset['totalprod'] == dataset[(dataset['year']==year)]['totalprod'].max()]['state'])[0]
    top_prod_state.append(top_state)
    top_prod_amount.append(dataset[dataset['year']==year]['totalprod'].max())
#Find lowest producer state each year
low_prod_state = []
low_prod_amount = []
for year in dataset['year'].unique():
    low_state = list(dataset[dataset['totalprod'] == dataset[(dataset['year']==year)]['totalprod'].min()]['state'])[0]
    low_prod_state.append(low_state)
    low_prod_amount.append(dataset[dataset['year']==year]['totalprod'].min())
#Overlayed bar plots showing the top/bottom producers of honey each year.
sns.set_style('whitegrid')
fig, ax1 = plt.subplots(1, 1, sharey=True, figsize=(15,6))

#Making the highest production bars
top_prod_plot = sns.barplot(x=dataset['year'].unique(), y=top_prod_amount, palette=sns.xkcd_palette(['blue']), alpha=0.5, log=True, ax=ax1)

#Definiting barwidths
barwidth = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]

#Labels for the highest producers
toplabel_ = top_prod_state
for bar, newwidth, label in zip(top_prod_plot.patches, barwidth, toplabel_):
    x = bar.get_x()
    width = bar.get_width()
    height = 0.60*(bar.get_height())
    centre = x+width/2.0
    bar.set_x(centre-newwidth/2.0)
    bar.set_width(newwidth)
    top_prod_plot.text(x+width/2.0, height-1, label, ha='center')

#Need to twiny to put the second plot on 'different' axes. Otherwise the .patches returns the same thing as above and the locations are off.
ax2 = ax1.twiny()

#Making lowest production bars
low_prod_plot = sns.barplot(x=dataset['year'].unique(), y=low_prod_amount, palette=sns.xkcd_palette(['red']), alpha=0.7, log=True, ax=ax2)

#Lowest producer labels
lowlabel_ = low_prod_state
for bar, newwidth, label in zip(low_prod_plot.patches, barwidth, lowlabel_):
    x = bar.get_x()
    width = bar.get_width()
    height = 0.60*(bar.get_height())
    centre = x+width/2.0
    bar.set_x(centre-newwidth/2.0)
    bar.set_width(newwidth)
    low_prod_plot.text(x+width/2.0, height, label, ha='center')

#Axes settings
ax1.set_ylim(1, 100000000)
ax1.set_xlabel('Year');
ax1.set_ylabel('Lbs')
ax2.get_xaxis().set_visible(False)
plt.title('Highest and Lowest Honey Production Per Year - Log Scale')

#Legend settings
top_patch = mpatches.Patch(color='blue', label='Most Honey Production', alpha=0.5)
bottom_patch = mpatches.Patch(color='red', label='Least Honey Production', alpha=0.7)
plt.legend(handles=[top_patch, bottom_patch], frameon=True, fancybox=True, loc=4, shadow=True)
plt.show()
#Create group based on the year, then find total production and total value
yearlyprodsum = dataset.groupby(['year'])['totalprod'].sum().values
yearlyvaluesum = dataset.groupby(['year'])['prodvalue'].sum().values
years = dataset['year'].unique()

fig, ax = plt.subplots(1,1,figsize=(15,6))
plt.plot(years, yearlyprodsum, 'b-', label='Lbs of Honey')
plt.plot(years, yearlyvaluesum, 'r-', label='Dollars of Value')
plt.title('Yearly Honey Production and Value - US')
plt.xlabel('Year')
plt.legend(frameon=True)
#Looking at the average yield per col each year, and the average price per Lb each year
avgyieldpercol = dataset.groupby(by='year')['yieldpercol'].mean().values
avgpriceperlb = dataset.groupby(by='year')['priceperlb'].mean().values
#Comparing averages of the yield per col and the price per lb

#Create figure object
fig, ax1 = plt.subplots(figsize=(15,8))

#Create initial plot and defining the first set of y-axes
sns.regplot(x=years, y=avgyieldpercol, color='magenta', ax=ax1)
ax1.set_ylabel('Average Honey Yield per Colony', color='magenta')
ax1.tick_params('y', colors='magenta')

#Twinning the x-axis and setting the second set of y-axes
ax2 = ax1.twinx()
sns.regplot(x=years, y=avgpriceperlb, color='green', ax=ax2)
ax2.set_ylabel('Average Price per Lb', color='green')
ax2.tick_params('y', colors='green')

#Global settings for the figure
plt.xlabel('Year')
plt.title('Comparison of Average Honey Yield per Colony and Average Price per Lb')
plt.show()
#Creating the groupby object for the states
grouped_state = dataset.groupby(by='state')
#Creating a dictionary of dfs containing each state's information
states = dataset['state'].unique()
states_dict = {key: None for key in states}
for name in states:
    group_df = grouped_state.get_group(name)
    states_dict[name] = group_df
#Stupid, but works to access a lot of data. Plot the column provided the column has a full 15 years of data. Can show large trends.
#Should try to create unique colours for each one...
plt.figure(figsize=(18,10))
for state in states:
    if len(states_dict[state]['yieldpercol']) < 15:
        pass
    else:
        plt.semilogy(years, states_dict[state]['numcol'], label=state);
        
plt.legend(ncol=2, bbox_to_anchor=(1.03,1.0))
plt.ylim((1e3,1e6))
plt.xlabel('Year')
plt.show()
#Plotting price per lb data over 4 separate plots
fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(18,10))

#Create a larger hidden subplot to create shared x/y labels
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
plt.grid(False)
plt.xlabel('Years')
plt.ylabel('Price per Lb (USD)')

#Building the plots
i=1
for state in states:
    if len(states_dict[state]['yieldpercol']) < 15:
        pass
    else:
        if i in np.arange(1,12,1):
            axes[0,0].plot(years, states_dict[state]['priceperlb'], label=state);    
        if i in np.arange(12,23,1):
            axes[0,1].plot(years, states_dict[state]['priceperlb'], label=state);
        if i in np.arange(23,34,1):
            axes[1,0].plot(years, states_dict[state]['priceperlb'], label=state);
        if i in np.arange(34,45,1):
            axes[1,1].plot(years, states_dict[state]['priceperlb'], label=state);
        i+=1;   

#Show legends
axes[0,0].legend()
axes[0,1].legend()
axes[1,0].legend()
axes[1,1].legend()

#Create a full titles
plt.suptitle('Cost of Honey Per Lb Each Year')
plt.show()

