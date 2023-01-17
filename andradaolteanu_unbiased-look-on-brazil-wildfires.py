# Import libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# Data Import

data = pd.read_csv('../input/forest-fires-in-brazil/amazon.csv', encoding='latin1')

data.head(5)
statistics = data['number'].describe().reset_index()



statistics.style.format({"number": "{:20,.0f}"}).hide_index().highlight_max(color='#FFB27A')
# Recode months into english

data['month'].replace(to_replace = 'Janeiro', value = 'Jan', inplace = True)

data['month'].replace(to_replace = 'Fevereiro', value = 'Feb', inplace = True)

data['month'].replace(to_replace = 'Março', value = 'Mar', inplace = True)

data['month'].replace(to_replace = 'Abril', value = 'Apr', inplace = True)

data['month'].replace(to_replace = 'Maio', value = 'May', inplace = True)

data['month'].replace(to_replace = 'Junho', value = 'Jun', inplace = True)

data['month'].replace(to_replace = 'Julho', value = 'Jul', inplace = True)

data['month'].replace(to_replace = 'Agosto', value = 'Aug', inplace = True)

data['month'].replace(to_replace = 'Setembro', value = 'Sep', inplace = True)

data['month'].replace(to_replace = 'Outubro', value = 'Oct', inplace = True)

data['month'].replace(to_replace = 'Novembro', value = 'Nov', inplace = True)

data['month'].replace(to_replace = 'Dezembro', value = 'Dec', inplace = True)
# Group data by year, state, month

year_mo_state = data.groupby(by = ['year','state', 'month']).sum().reset_index()



year_mo_state.head()
# Set up the defaults

sns.set_style('whitegrid')

from matplotlib.pyplot import MaxNLocator, FuncFormatter



plt.figure(figsize=(16,9))



# Make the plot

ax = sns.lineplot(x = 'year', y = 'number', data = year_mo_state, estimator = 'sum', color = '#FF5555', 

                  lw = 3.5, err_style = None , alpha = 0.85)

# Create a line

import pylab as p

p.arrow( 1998, 25000, 19, 15000, facecolor="#FFB27A", edgecolor="#FFB27A", head_width=0, head_length=0, length_includes_head = False,

       width = 0.15, alpha = 0.7, shape = "full")



# Make pretty

plt.title('Total Fires in Brazil : 1998 - 2017', fontsize = 25)

plt.xlabel('Year', fontsize = 20)

plt.ylabel('Number of Fires', fontsize = 20)

plt.xticks(fontsize = 15)

plt.yticks(fontsize = 15)



ax.xaxis.set_major_locator(plt.MaxNLocator(19))

ax.set_xlim(1998, 2017)



ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
# Figure size

plt.figure(figsize=(16,12))



# The plot

sns.boxplot(x = 'month', order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep','Oct', 'Nov', 'Dec'], 

            y = 'number', data = year_mo_state, palette = "coolwarm", saturation = 1, width = 0.9, fliersize=4, linewidth=2)



# Make pretty

plt.title('Fires in Brazil by Month', fontsize = 25)

plt.xlabel('Month', fontsize = 20)

plt.ylabel('Number of Fires', fontsize = 20)

plt.xticks(fontsize = 15)

plt.yticks(fontsize = 15)
# First create the data

year_mo_state_Amazon = data[data['state'] == 'Amazonas'].groupby(by = ['year','state', 'month']).sum().reset_index()



# Set up the figure size

plt.figure(figsize=(16,9))



# CReate the plot

ax = sns.lineplot(x = 'year', y = 'number', data = year_mo_state_Amazon, estimator = 'sum', color = '#FF2323', lw = 3.5, 

                  err_style = None, alpha = 0.85)



# Add line

p.arrow( 1998, 1500, 19, 0, facecolor="#FFB27A", edgecolor="#FFB27A", head_width=0, head_length=0, length_includes_head = False,

       width = 20, alpha = 0.7, shape = "full")



# Make pretty

plt.title('Total Fires in Amazon : 1998 - 2017', fontsize = 25)

plt.xlabel('Year', fontsize = 20)

plt.ylabel('Number of Fires', fontsize = 20)

plt.xticks(fontsize = 15)

plt.yticks(fontsize = 15)



ax.xaxis.set_major_locator(plt.MaxNLocator(19))

ax.set_xlim(1998, 2017)



ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
# Creating the top 10 dataframe

states_fires = data.groupby(by = 'state')['number'].sum().sort_values(ascending = False).head(10).reset_index()

states_fires = states_fires.sort_values(by = 'number', ascending = True)



states_fires
# Set figure size

plt.figure(figsize = (16, 9))



# plot

ax = sns.barplot(x = states_fires['state'], y = states_fires['number'], palette = "Reds", alpha = 0.85)



# Make pretty

plt.title("Top 10 most affected states", fontsize = 25)

plt.xlabel("State", fontsize = 20)

plt.ylabel("Sum of Wildfires", fontsize = 20)

plt.xticks(fontsize = 15)

plt.yticks(fontsize = 15)

plt.legend(fontsize = 15)
# Prepare the data

year_mo_state_top_states = data[data['state'].isin(['Amazonas','Mato Grosso','Paraiba','Sao Paulo','Rio'])].groupby(by = ['year','state', 'month']).sum().reset_index()



# The plot

plt.figure(figsize=(16,9))

ax = sns.lineplot(x = 'year', y = 'number', data = year_mo_state_top_states, hue = 'state', estimator = 'sum', lw = 3.5, 

                  err_style = None, palette = ["#FF2323", "#FFA653", "#FF6E6E", "#FF8411", "#FFDA11"])



# Make pretty

plt.title('Total Fires in Amazon : 1998 - 2017', fontsize = 25)

plt.xlabel('Year', fontsize = 20)

plt.ylabel('Number of Fires', fontsize = 20)

plt.xticks(fontsize = 15)

plt.yticks(fontsize = 15)



ax.xaxis.set_major_locator(plt.MaxNLocator(19))

ax.set_xlim(1998, 2017)



ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 14})