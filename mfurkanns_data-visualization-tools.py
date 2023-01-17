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
players = pd.read_csv('../input/data.csv')

players.head()
players.info()
players
turkish_players = players[players.Nationality=='Turkey']

turkish_players = turkish_players.head(25)

for a,i in turkish_players.iterrows():

    print(i.Name)

plt.figure(figsize=(15,10))

sns.barplot(x=turkish_players['Name'], y=turkish_players['Overall'])

plt.xticks(rotation= 45)

plt.xlabel('Turkish Players')

plt.ylabel('Overall')

plt.title('First 25 Turkish Players')

plt.show()
countries = players.Nationality.unique()

countries
population = []

average = []





for i in range(0,164):

    population.append(0)

    average.append(0)



for a,i in players.T.iteritems():

    for ind,coun in enumerate(countries):

        if(i.Nationality == coun):

            population[ind]+=1

            average[ind]+=i.Overall

            break



for i in range(0,164):

    average[i] = average[i]/population[i]

            

list_label = ["country","population","average"]

list_col = [countries,population,average]

zipped = list(zip(list_label,list_col))

data_dict = dict(zipped)

dfa = pd.DataFrame(data_dict)





index = (dfa['average'].sort_values(ascending=False)).index.values

sorted_dfa = dfa.reindex(index)



plt.figure(figsize=(15,10))

sns.barplot(x=sorted_dfa['country'].head(25), y=sorted_dfa['average'].head(25))

plt.xticks(rotation= 45)

plt.xlabel('Countries')

plt.ylabel('Overall')

plt.title('Average of 25 Countries at Football')

plt.show()