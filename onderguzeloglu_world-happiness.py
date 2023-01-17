# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from collections import Counter

%matplotlib inline

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
world_happiness_2015 = pd.read_csv('/kaggle/input/world-happiness/2015.csv').head(25)

world_happiness_2016 = pd.read_csv('/kaggle/input/world-happiness/2016.csv').head(25)

world_happiness_2017 = pd.read_csv('/kaggle/input/world-happiness/2017.csv').head(25)

world_happiness_2018 = pd.read_csv('/kaggle/input/world-happiness/2018.csv').head(25)

world_happiness_2019 = pd.read_csv('/kaggle/input/world-happiness/2019.csv').head(25)
world_happiness_2019.head()
world_happiness_2019.info()
world_happiness_2019['Country or region'].value_counts()
# rate of happiness to regions in 2019

country_list = list(world_happiness_2019['Country or region'])

happiness_score = []

for i in country_list:

    x = world_happiness_2019[world_happiness_2019['Country or region']== i ]

    happiness_score_rate = sum(x['Score'])/len(x)

    happiness_score.append(happiness_score_rate)

#sorting

data = pd.DataFrame({'country_list':country_list,'happiness_score':happiness_score})

new_index = (data['happiness_score'].sort_values(ascending = False)).index.values

sorted_data = data.reindex(new_index).head(25)



#visualization

plt.figure(figsize = (15,10))

sns.barplot(x = sorted_data['country_list'], y = sorted_data['happiness_score'])

plt.xticks(rotation = 45)

plt.xlabel = ('Country or Region')

plt.ylabel = ('Happines score')

plt.title = ('Happiness score to each Counrty or regions')
world_happiness_2017.head()
# rate of freedom each countries in 2017



countries = list(world_happiness_2017['Country'])

freedom_ratio = []

for i in countries:

    x = world_happiness_2017[world_happiness_2017['Country'] == i ]

    freedom_rate = sum(x['Freedom']) / len(x)

    freedom_ratio.append(freedom_rate)

#sorting

data1 = pd.DataFrame({'countries' : countries,'freedom_ratio' : freedom_ratio})

new_index1 = (data1['freedom_ratio'].sort_values(ascending = False)).index.values

sorted_data1 = data1.reindex(new_index1).head(15)



#visualization



plt.figure(figsize=(20,10))

ax = sns.barplot(x = sorted_data1['countries'], y = sorted_data1['freedom_ratio'], palette = sns.cubehelix_palette(len(x)))

plt.xlabel = 'Countries'

plt.ylabel = 'Freedom ratio'

plt.title = 'rate of freedom each countries in 2017'
world_happiness_2016.head()
#generosity,happiness score, health(life expectancy) rates to each countries

countries = list(world_happiness_2016['Country'])

generosity_ratio = []

happiness_score = []

Family_ratio = []

for i in countries:

    x = world_happiness_2016[world_happiness_2016['Country'] == i]

    generosity_ratio.append(sum(x.Generosity)/len(x))

    happiness_score.append(sum(x['Happiness Score'])/len(x))

    Family_ratio.append(sum(x['Family'])/len(x))

    

#visualization

f,ax = plt.subplots(figsize = (50,70))

sns.barplot(x = generosity_ratio, y = countries, color = 'green', alpha = 0.5, label = 'Generosity')

sns.barplot(x = happiness_score, y = countries, color = 'red', alpha = 0.5, label = 'Happiness score')

sns.barplot(x = Family_ratio, y = countries, color = 'cyan', alpha = 0.5, label = 'Family')

ax.legend(loc = ('low right'),frameon = True)

ax.set(xlabel = 'Generosity,Happiness Score,Health (Life Expectancy)',ylabel = 'Countries', title = 'generosity,happiness score, health(life expectancy) rates to each countries')

plt.show()
world_happiness_2019.head()
countries = list(world_happiness_2019['Country or region'])

generosity_score = []

happiness_score = []

for i in countries:

    x = world_happiness_2019[world_happiness_2019['Country or region'] == i ]

    generosity_score.append(sum(x.Generosity) / len(x))

    happiness_score.append(sum(x.Score)/len(x))

    

#sorting

data = pd.DataFrame({'countries': countries,'happiness_score': happiness_score,'generosity_score':generosity_score})

new_index = (data['happiness_score'].sort_values(ascending=False)).index.values

sorted_data2 = data.reindex(new_index)



# visualization

f,ax = plt.subplots(figsize = (20,10))

sns.pointplot(x='countries', y = 'generosity_score', data = data, color = 'lime', alpha = 0.8)

sns.pointplot(x='countries', y = 'happiness_score', data = data, color = 'red', alpha = 0.8)

plt.text(40,0.6,'generosity',color='red',fontsize = 17,style = 'italic')

plt.text(40,0.55,'happiness score',color='lime',fontsize = 18,style = 'italic')

plt.xlabel = 'countries'

plt.ylabel = 'Values'

plt.title = 'freedom choice vs happiness score'

plt.grid()
g = sns.jointplot(data['generosity_score'],data['happiness_score'], kind = "kde", size = 7)

plt.savefig('graph.png')

plt.show()
#Different usage of parameters but same plot with previous one

g = sns.jointplot("generosity_score","happiness_score", data = data, size = 5, ratio=3,color="r")
world_happiness_2016
labels = world_happiness_2016.Region.value_counts('Happiness Score').index

colors = ['grey','blue','yellow','green','brown']

explode = [0,0,0,0,0,0]

sizes = world_happiness_2016.Region.value_counts('Happiness Score').values



# visual pie charm

plt.figure(figsize=(7,7))

plt.pie(sizes, explode=explode,labels=labels, colors = colors,autopct = '%1.1f%%')

plt.title("Happiness score to Regions",color = 'blue', fontsize = 15)
sns.lmplot(x = "generosity_score", y="happiness_score", data=data)

plt.show()
#Visualization of high school graduation rate vs poverty rate of each state with different style of seaborn code

# cubehelix plot

sns.kdeplot(data.generosity_score,data.happiness_score, shade=True, cut = 3)

plt.show()
pal = sns.cubehelix_palette(2, rot=-.5,dark=.3)

sns.violinplot(data = data, palette=pal,inner="points")

plt.show()


f,ax = plt.subplots(figsize = (5,5))

sns.heatmap(data.corr(),annot = True, linewidths=.5, fmt='.1f',ax = ax)

plt.show()
#sns.boxplot(x = "generosity_score", y = "happiness_score", hue = "generosity_score", data = data, palette="PRGn")

#plt.show()
# manner of death(olum sekli) : ates edilerek, ates edilerek ve sok tabancasi ile

# gender

# age

sns.swarmplot(x = "generosity_score", y = "happiness_score", hue = "happiness_score", data = data)

plt.show()
# pair plot

sns.pairplot(data)

plt.show()
# kill properties

# Manner of death

#sns.countplot(kill.gender)

#sns.countplot(kill.manner_of_death)

#plt.title("gender",color = 'blue',fontsize=15)