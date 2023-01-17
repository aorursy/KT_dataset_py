# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns
Nobel = pd.read_csv('../input/archive.csv')

print(Nobel.head(10))
# Number of Nobel prizes awarded

display(len(Nobel.Prize)) 

# NUmber of Nobel prizes won by male and female

display(Nobel.Sex.value_counts())

# Top 10 countries in Nobel prize winning

Nobel['Birth Country'].value_counts().head(10)
# USA

Nobel['usa_born_winner'] = Nobel['Birth Country'] == 'United States of America'

Nobel['decade'] = (np.floor(pd.Series(Nobel.Year)/10)*10).astype(int)

prop_usa_winners = Nobel.groupby('decade', as_index = False)['usa_born_winner'].mean()



# displaying usa born winners per decade

display(prop_usa_winners)
#setting plotting theme

sns.set()



plt.rcParams['figure.figsize'] = [11,8]



#Plotting line plot for USA born winners

ax = sns.lineplot(prop_usa_winners['decade'],prop_usa_winners['usa_born_winner'])



# Adding %-formatting to the y- axis

from matplotlib.ticker import PercentFormatter

ax.yaxis.set_major_formatter(PercentFormatter())
# Calculating the proportion of female laureates per decade

Nobel['female_winner'] = Nobel.Sex == "Female"

prop_female_winners = Nobel.groupby(['decade','Category'],as_index=False)['female_winner'].mean()



# Plotting USA born winners with % winners on the y-axis



# Setting plotting theme

sns.set()

# and setting the size of all plots.

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [11, 7]



# Plotting USA born winners 

ax = sns.lineplot(x='decade', y='female_winner', hue='Category', data=prop_female_winners)

# Adding %-formatting to the y-axis

from matplotlib.ticker import PercentFormatter

ax.yaxis.set_major_formatter(PercentFormatter())
Nobel[Nobel['Sex'] == 'Female'].nsmallest(1,'Year',keep='first')
Nobel.groupby("Full Name").filter(lambda x: len(x) >= 2)['Full Name'].value_counts()
# Converting birth_date from String to datetime

Nobel['Birth Date'] = pd.to_datetime(Nobel["Birth Date"],errors='coerce')

# Calculating the age of Nobel Prize winners

Nobel['Age'] = Nobel['Year'] - Nobel['Birth Date'].dt.year



# Plotting the age of Nobel Prize winners



sns.lmplot(x='Year',y='Age',data=Nobel, lowess=True, aspect=2, line_kws={"color" : "black"})
sns.lmplot(x='Year',y='Age',data=Nobel, lowess=True, aspect=2, line_kws={"color" : "black"},row='Category')
display(Nobel.nlargest(1, "Age"))



Nobel.nsmallest(1, "Age")