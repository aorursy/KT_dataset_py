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
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
df = pd.read_csv('../input/data.csv')



#looking at the data

print(df.head(10))
#let's see the type of columns that we have in our data

print(df.columns)

print('TOTAL NUMBER OF COLUMNS : {}'.format(len(df.columns)))
print(df.info())
sns.set(style ="dark", palette="colorblind", color_codes=True)

plt.figure(figsize = (16,10))

sns.countplot(df['Preferred Foot'])

plt.title('COUNT OF PREFERRED FOOT')
sns.set(style ="dark", palette="colorblind", color_codes=True)

plt.figure(figsize = (16,12))

sns.violinplot(y = df['SprintSpeed'])

plt.title('SPRINT SPEED DISTRIBUTION')
print(max(df['SprintSpeed'].values))

print(min(df['SprintSpeed'].values))



sns.set(style ="dark", palette="colorblind", color_codes=True)

plt.figure(figsize = (20, 16))

sns.countplot(y = df['SprintSpeed'].values[:100])

plt.ylabel('SPRINT SPEEDS', fontsize = 16)

plt.title('SPRINT SPEEDS OF THE PLAYERS', fontsize = 20)
df1 = df['Nationality'].head()

df2 = df['Value'].head()

df3 = df['Name'].head()

conc_data = pd.concat([df1,df2,df3],axis =1) # axis = 0 : adds dataframes in row

print(conc_data)
df['Potential'].plot(kind = 'line', color = 'g', label = 'Reactions', linewidth = 1, alpha = 0.5,grid = True,linestyle = ':')

df['Overall'].plot(color = 'r',label = 'Overall', linewidth = 1, alpha = 0.5, grid = True, linestyle = '--')
# Histogram: number of players's age

sns.set(style ="dark", palette="colorblind", color_codes=True)

plt.figure(figsize=(16,8))

sns.distplot(df.Age, bins = 58, kde = False, color='r')

plt.xlabel("Player\'s age", fontsize=16)

plt.ylabel('Number of players', fontsize=16)

plt.title('Histogram of players age', fontsize=20)

plt.show()
# Compare six clubs in relation to age

club_names = ('Real Madrid', 'Liverpool', 'Juventus', 'Manchester United', 'FC Barcelona')

df_club = df.loc[df['Club'].isin(club_names) & df['Age']]



fig, ax = plt.subplots()

fig.set_size_inches(15, 10)

ax = sns.boxplot(x = "Club", y = "Age", data = df_club);

ax.set_title('Distribution of age in some clubs', fontsize=20);
# let's see the total number of players that we have at different playing positions

plt.figure(figsize = (16, 10))

sns.countplot(y = df['Position'], data = df, palette = 'plasma');

plt.title('Count of players on the position', fontsize=20);
#let's see wo are the top players with 'LEFT' foot as their preffered foot

data_left = df[df['Preferred Foot'] == 'Left'][['Name', 'Overall', 'Age', 'Potential', 'FKAccuracy']]

print(data_left.head(10))