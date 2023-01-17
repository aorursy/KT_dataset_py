import pandas as pd   
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('bmh')
%matplotlib inline
plt.rcParams['figure.dpi'] = 100
pokedata = pd.read_csv("../input/pokemon-with-stats-plus-gen-7/Pokemon_all.csv")
pokedata.head()                    
pokedata.tail()
pokedata.columns = pokedata.columns.str.upper()
pokedata['TYPE 1'] = pokedata['TYPE 1'].str.capitalize()
pokedata['TYPE 2'] = pokedata['TYPE 2'].str.capitalize()
pokedata.drop_duplicates('#', keep='first', inplace=True)
pokedata['TYPE 2'].fillna(value='None', inplace=True)
pokedata.head()
pokedata.tail()
pokedata['#'].count()
sns.factorplot(
    x='GENERATION', 
    data=pokedata,
    size=5,
    aspect=1.2,
    kind='count'
).set_axis_labels('Generation', '# of Pokemon')

plt.show()
pokedata['LEGENDARY'].value_counts()
fig = plt.figure(figsize=(7,7))

colours = ["aqua", "orange"]
pokeLeg = pokedata[pokedata['LEGENDARY']==True]
pokeNon = pokedata[pokedata['LEGENDARY']==False]

legDist = [pokeLeg['NAME'].count(),pokeNon['NAME'].count()]
legPie = plt.pie(legDist,
                 labels= ['Legendary', 'Non Legendary'], 
                 autopct ='%1.1f%%', 
                 shadow = True,
                 colors=colours,
                 startangle = 45,
                 explode=(0, 0.1))

colours = ["aqua", "orange"]
g = sns.factorplot(
    x='GENERATION', 
    data=pokedata,
    kind='count', 
    hue='LEGENDARY',
    palette=colours, 
    size=5, 
    aspect=1.5,
    legend=False,
    ).set_axis_labels('Generation', '# of Pokemon')

g.ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),  shadow=True, ncol=2, labels=['NON LEGENDARY','LEGENDARY'])
plt.show()

pokedata['TYPE 1'].unique()
fig = plt.figure(figsize=(15,15))

fig.add_subplot(211)
pokedata['TYPE 1'].value_counts().plot(kind='pie', 
                                       autopct='%1.1f%%',
                                       pctdistance=0.9)

fig.add_subplot(212)
pokedata['TYPE 2'].value_counts().plot(kind='pie', 
                                       autopct='%1.1f%%',
                                       pctdistance=0.9)

plt.show()
sns.factorplot(
    y='TYPE 1',
    data=pokedata,
    kind='count',
    order=pokedata['TYPE 1'].value_counts().index,
    size=4,
    aspect=1.5,
    color='green'
).set_axis_labels('# of Pokemon', 'Type 1')

sns.factorplot(
    y='TYPE 2',
    data=pokedata,
    kind='count',
    order=pokedata['TYPE 2'].value_counts().index,
    size=4,
    aspect=1.5,
    color='purple'
).set_axis_labels('# of Pokemon', 'Type 2');
plt.subplots(figsize=(10, 10))

sns.heatmap(
    pokedata[pokedata['TYPE 2']!='None'].groupby(['TYPE 1', 'TYPE 2']).size().unstack(),
    linewidths=1,
    annot=True,
    cmap="Blues"
)

plt.xticks(rotation=35)
plt.show()
