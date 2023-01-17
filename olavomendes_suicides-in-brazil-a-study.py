import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# Reading the dataset

suicides = pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')

suicides.head()
# Selecting only data from Brazil and assigning it to a variable

suicides_brazil = suicides[suicides.country == 'Brazil'].sort_values(by='year', ascending=False)

suicides_brazil.head()
# Setting the size of the figure

plt.figure(figsize=(10, 6))



#Plotting the comparison lines between suicides in Brazil and in the world

sns.lineplot(x=suicides_brazil['year'], y=suicides_brazil['suicides/100k pop'], color='b', label='Brazil')

sns.lineplot(x=suicides['year'],  y=suicides['suicides/100k pop'], color='r', label='World')



plt.title('Comparison of suicides between Brazil and the world')



# Defining the limit of the figure to show the years from 1987 to 2015, since Brazil does not have data for 2016

plt.xlim(1987, 2015)



plt.legend();
# Separating and adding the number of suicides by sex

sex_suicides_percent = suicides_brazil.groupby('sex')['suicides_no'].sum()



sex_suicides_percent
# Plotting a pie chart with the number of suicides by sex

colors_pie = ['red', 'cyan']

plt.pie(sex_suicides_percent, 

        labels=sex_suicides_percent.index,

        autopct='%.1f%%',

        shadow=True,

        colors=colors_pie,

        explode=[0.1, 0]);
# Separating and adding the number of suicides by age

colors_barh = ['y', 'm', 'b', 'c', 'g', 'r']

age = suicides_brazil.groupby('age')['suicides_no'].sum().sort_values()

x = age.index

age
# Setting the size of the figure

plt.figure(figsize=(10, 8))



# Plotting a horizontal bar chart

plt.barh(x, age.values,

        color=colors_barh);
# Separating the number of suicides by generation

colors_bar = ['y', 'm', 'b', 'c', 'g', 'r']

generation = suicides_brazil.groupby('generation')['suicides_no'].sum().sort_values()

generation
# Separating the number of suicides by generation

gen = generation.index



plt.figure(figsize=(10, 8))



# Plotting a graph showing the total number of suicides per generation

plt.bar(gen, generation.values,

       color=colors_bar);