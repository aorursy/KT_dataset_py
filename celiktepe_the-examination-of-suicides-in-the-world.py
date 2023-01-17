import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from collections import Counter

import os

print(os.listdir("../input"))
data = pd.read_csv('../input/master.csv')

data.info()
data.describe()
data.head()
data.columns
data.country.unique()
country_List = list(data.country.unique())

suicide_ratio = []

for x in country_List:

    tmp = data[data['country'] == x]

    averages = sum(tmp.suicides_no) / len(tmp)

    suicide_ratio.append(int(averages))

    

data_List = pd.DataFrame({'country' : country_List, 'suicide_ratio' : suicide_ratio})

new_index = (data_List['suicide_ratio'].sort_values(ascending=False)).index.values

sorted_data = data_List.reindex(new_index)



plt.figure(figsize = (18, 10))

sns.barplot(x = sorted_data['country'], y = sorted_data['suicide_ratio'])

plt.xticks(rotation = 90)

plt.xlabel('Countries')

plt.ylabel('Ratio')

plt.title('Suicide Ratio')

plt.show()
year_list = data.year

year_count = Counter(year_list)

most = year_count.most_common(15)



x, y = zip(*most)

x , y = list(x), list(y)



plt.figure(figsize = (15, 8))

sns.barplot(x = x, y = y, palette = sns.cubehelix_palette(len(x)), order = x)

plt.xlabel('Years')

plt.ylabel('Frequency')

plt.title('the years which was existed the most suicides')

plt.show()
data.head()
data.year = data.year.astype(float)

yearList = list(data.country.unique())

overall = []

for x in yearList:

    tmp = data[data.country == x]

    average = sum(tmp.year) / len(tmp)

    overall.append(average)

data1 = pd.DataFrame({'country': yearList, 'averages': overall})

# sorting

newIndex = (data1['averages'].sort_values(ascending=True)).index.values

sorted_Data = data1.reindex(newIndex)

filtred_Data = sorted_Data.loc[:15]

# visualization

plt.figure(figsize = (15, 10))

sns.barplot(x = filtred_Data['country'], y = filtred_Data['averages'])

plt.xticks(rotation = 90)

plt.show()
county_List = list(data.country.unique())

Silent = []

Millenials = []

Boomers = []

Generation_Z = []

G_I_Generation = []



for i in country_List:

    tmp = data[data.country == i]

    dl = Counter(tmp.generation)

    Silent.append(dl['Silent'])

    Millenials.append(dl['Millenials'])

    Boomers.append(dl['Boomers'])

    Generation_Z.append(dl['Generation Z'])

    G_I_Generation.append(dl['G.I. Generation'])



f,vs = plt.subplots(figsize = (9, 18))

sns.barplot(x = Silent, y = country_List, color = 'red', alpha = 0.6, label = 'Silent')

sns.barplot(x = Millenials, y = country_List, color = 'blue', alpha = 0.5, label = 'Millenials')

sns.barplot(x = Boomers, y = country_List, color = 'green', alpha = 0.6, label = 'Boomers')

sns.barplot(x = Generation_Z, y = country_List, color = 'orange', alpha = 0.7, label = 'Generation Z')

sns.barplot(x = G_I_Generation, y = country_List, color = 'cyan', alpha = 0.7, label = 'G.I. Generation')



vs.legend(loc = 'lower right', frameon = True)

vs.set(xlabel='Generation number', ylabel='Countries',title = "The Number of Countries According to Generation ")

            

         
data.head()