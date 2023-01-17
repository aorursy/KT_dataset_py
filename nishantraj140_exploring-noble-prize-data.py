# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Reading the data

Nobel = pd.read_csv("../input/nobel-laureates/archive.csv")

# Taking a look at the dataset

display(Nobel.shape)

display(Nobel.columns)

display(Nobel.head(10))
# Number of Nobel prizes awarded

display(len(Nobel.Prize)) 

# NUmber of Nobel prizes won by male and female

display(Nobel.Sex.value_counts())

# Top 10 countries in Nobel prize winning

Nobel['Birth Country'].value_counts().head(10)
Nobel['usa_born_winner'] = Nobel['Birth Country'] == 'United States of America'

Nobel['decade'] = (np.floor(pd.Series(Nobel.Year)/10)*10).astype(int)

prop_usa_winners = Nobel.groupby('decade', as_index = False)['usa_born_winner'].mean()



# displaying usa born winners per decade

display(prop_usa_winners)
#setting plotting theme

sns.set()



plt.rcParams['figure.figsize'] = [11,8]



#Plotting line plot for USA born winners

ax = sns.lineplot(x='decade',y='usa_born_winner', data=prop_usa_winners)



# Adding %-formatting to the y- axis

ax.yaxis.set_major_formatter(PercentFormatter())
Nobel["Laureate Type"].value_counts()
Nobel.Category.value_counts()
# Display the number of nobel prizes given by category

sns.set(style='white')

ax = sns.countplot(x='Category', palette = 'GnBu_d', order = Nobel['Category'].value_counts().index, data=Nobel)

ax.set_title('Nobel prizes given in total by category from 1901 to 2016', fontsize=15)
# Display the number of prizes won by sex from 1901 to 2016

sns.set(style='white')

ax = sns.countplot(x='Sex', data=Nobel)

ax.set_title('Nobel prizes won by male and female between 1901 and 2016')

Nobel[Nobel['Sex'] == 'Female'].nsmallest(1,'Year')
Nobel.groupby("Full Name").filter(lambda x: len(x) >= 2)['Full Name'].value_counts()
# Converting birth_date from String to datetime

Nobel['Birth Date'] = pd.to_datetime(Nobel["Birth Date"],errors='coerce')

# Calculating the age of Nobel Prize winners

Nobel['Age'] = Nobel['Year'] - Nobel['Birth Date'].dt.year



# Plotting the age of Nobel Prize winners



sns.lmplot(x='Year',y='Age',data=Nobel, lowess=True, aspect=2, line_kws={"color" : "black"})