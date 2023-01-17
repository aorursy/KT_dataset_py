# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#reading data

data = pd.read_csv('../input/fifa-20-complete-player-datasets/FIFAdatas.csv')
# taking a look at the data

data.describe()
# still looking at the dataframe

data.shape
# checking if this df has duplicated values

data.duplicated()
# dropping all duplicated

data = data.drop_duplicates()
data.head()
# by taking a look at the data we noticed that we have to fix some strings like coluns Age, Wage and Value

# lets first  adjust column Age

data.info()
# converting age to float

# thanks to Samrat rai for helping me with this proccess

def Age_num(df_age):

    try:

        age = float(df_age[2:-2])

    except ValueError:

        age = NaN

    return age   

data['Age_Num'] = data['Age'].apply(Age_num)

print(data['Age_Num'])
# now we have to fix wage and value

def extract_wages(x):

    out = x.replace('€','')

    if 'K' in out:

        out = float(out.replace('K',''))*1000

    elif 'M' in out:

        out = float(out.replace('M',''))*100000

    return float(out)
# applying the function for Wage

data['Wage_num'] = data['Wage'].apply(lambda x: extract_wages(x))

data.head()
# applying for Value

data['Value_num'] = data['Value'].apply(lambda x: extract_wages(x))

data.head()
# let's check which position values the most



fig = plt.figure(figsize = (10,5))

plt.plot(data.groupby('Position')['Value_num'].mean().sort_values(ascending = False),'--bo')

plt.xlabel('Positions')

plt.ylabel('Mean of value')

plt.title('Value by positions')
#testing boxplot

# lets check the wage's distribuition by position

# we can see a lot of outliers, probably for cristiano ronaldo, messi and neymar

# for the other positions we can consider per exemplo: Virgil van Dijk

data.boxplot(column = 'Wage_num', by = 'Position');

plt.title('Wage by positions')

plt.suptitle('')

plt.grid('')
# checking which players have most wage

# it's strange because we see players that don't usually appear so much in midia, neymar and CR7 are one of those, and Sané is a bench player

data.sort_values(by = 'Wage_num', ascending= False).head()
# checking the top 8 teams with the higher overall rating

fig = plt.figure(figsize = (12,10));

ax = fig.add_axes([0.1, .1, 1, 0.4]);

plt.plot(data.groupby('Team')['Overall'].mean().sort_values(ascending = False).head(8),'-o');

plt.title('Overall dos tops 8 times do FIFA 20', fontsize = 16, color = 'w')

plt.xlabel('Clubes', fontsize = 14, color = 'w')

plt.ylabel('Overall', fontsize=  14, color = 'w')
# again we see something strange, really Real Sociedad ? It's a good team but probably their data may be weird. Let's check

data[data['Team'] == 'Real Sociedad']
#lets plot only teams that at least have more than 3 players listed

counts_team = data['Team'].value_counts().sort_values(ascending = False)

counts_team[counts_team >= 3]



group_by_filter = pd.DataFrame(data[data['Team'].isin(counts_team[counts_team >3].index)].groupby('Team')['Overall'].mean().sort_values(ascending = False))

# first line is in index, we have to reset

group_by_filter = group_by_filter.reset_index()
# In this graphic we see which teams has higher overall ratings

fig = plt.figure(figsize = (12,10));

plt.bar(x = group_by_filter['Team'], height = group_by_filter['Overall'],

        color = 'blue',

        width = 0.8,

        alpha = 0.6,

        edgecolor = 'black')

plt.title('Top clubes com o maior overall',fontsize = 24, color = 'w');



plt.xlabel('Clubes',fontsize = 16, color = 'w');

plt.ylabel('Overall',fontsize = 16, color = 'w');
