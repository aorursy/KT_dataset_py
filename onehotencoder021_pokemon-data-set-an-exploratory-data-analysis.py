#Import the necessary libraries

import numpy as np

import pandas as pd

import seaborn as sb

from scipy import stats

import matplotlib.pyplot as plt

import regex as re
#Read in the dataset

df = pd.read_csv("../input/PokemonData.csv") 
df.head()
df.describe()
df.isna().sum()
df.Type2.fillna("None",inplace=True)

df['Total'] = df.HP + df.Attack + df.Defense+df.Speed + df.SpAtk+df.SpDef
df.drop_duplicates('Num', keep='first', inplace=True)
Dex = df[['Num', 'Name', 'Type1', 'Type2', 'Generation', 'Legendary']]



statistics = pd.merge(df, Dex, on='Num').loc[:, ['Num', 'HP', 'Attack', 'Defense', 'SpAtk', 'SpDef', 'Speed',

          'Total']]

plt.figure(figsize=(15,10))

sb.heatmap(df.corr(),annot = True)
df.Name = df.Name.apply(lambda x: re.sub(r'(.+)(Mega.+)',r'\2',x))

df.Name = df.Name.apply(lambda x: re.sub(r'(.+)(Primal.+)',r'\2',x))

df.Name = df.Name.apply(lambda x: re.sub(r'(HoopaHoopa)(.+)','Hoopa'+r'\2',x))
NL_Poke = df.loc[(df['Legendary']==False)]

L_Poke = df.loc[(df['Legendary']==True)]
#Pie chart for pokemon - Legendary vs Non Legendary

Split = [NL_Poke['Name'].count(),L_Poke['Name'].count()]

LegPie = plt.pie(Split,labels=['Not Legendary','Legendary'],autopct='%1.1f%%',shadow=True)

plt.title('Legendary vs Non-Legendary')

fig=plt.gcf()
plt.figure(figsize=(6,3))

sb.kdeplot(df["Total"],legend=False,color="blue",shade=True)
sb.kdeplot(df["HP"],legend = False,color="blue",shade=True)
sb.kdeplot(df["Attack"],legend = False,color="blue",shade=True)
sb.kdeplot(df["Defense"],legend = False,color="blue",shade=True)

sb.kdeplot(df["Speed"],legend = False,color="blue",shade=True)
plt.figure(figsize=(20,10))

sb.countplot(x='Type1',data = df)
plt.figure(figsize=(20,10))

sb.countplot(x='Type2',data = df)
sb.catplot(x='Generation', data=df,col='Type1',kind='count',col_wrap=3).set_axis_labels('Generation', 'Pokemons');
plt.figure(figsize = (15,10))

dualTypes = df[df['Type2'] != 'None']

sb.heatmap( dualTypes.groupby(['Type1', 'Type2']).size().unstack(),linewidths=1,annot=True)
plt.figure(figsize=(20,10))

Defhist = sb.distplot(df['Defense'],color='red',hist=True)

Atthist = sb.distplot(df['Attack'],color='teal',hist=True)

Atthist.set(title='Distribution of Defense and Attack',xlabel = 'Defense:red , Attack:teal')

FigHist = Atthist.get_figure()
plt.figure(figsize=(20,10))

SpDefHist = sb.distplot(df['SpDef'],color='red',hist=True)

SpAttHist = sb.distplot(df['SpAtk'],color='teal',hist=True)

SpAttHist.set(title='Distribution of Sp Defense and Sp attack',xlabel='SpDef : red , SpAtk : teal')

Fighist = SpAttHist.get_figure()

stats = ['Total','HP','Attack','Defense','SpAtk','SpDef','Speed']





def maxStats(df,cols):

    st = ''

    for col in cols:

        stat = df[col].max()

        name=df[df[col] == df[col].max()]['Name'].values[0]

        gen = df[df[col] == df[col].max()]['Generation'].values[0]

        st += name + " of Generation "+str(gen)+" has the best "+col+" stat of "+str(stat)+".\n"

        

    return st

print(maxStats(NL_Poke,stats))
#Compare base stats of all generations

plt.figure(figsize=(20,10))

bp = sb.boxplot(x='Generation',y='Total',data=NL_Poke)

plt.title('Base Stat Total',fontsize=17)

plt.xlabel('Generation',fontsize=12)

plt.ylabel('Total',fontsize=12)
df.sort_values('Total',ascending=False).head(30)
stdStats = statistics.drop('Total', axis='columns').set_index('Num').apply(

    lambda x: (x - x.mean()) / x.std())

stdStats['strength'] = stdStats.sum(axis='columns')
stdStats.reset_index(inplace=True)

pd.merge(Dex, stdStats, on='Num').sort_values('strength', ascending=False).head(30)
pd.merge( Dex[~Dex['Legendary']],stdStats, on='Num').sort_values('strength', ascending=False).head(10)
joined = pd.merge(Dex,stdStats,on='Num')

medians = joined.groupby(['Type1', 'Type2']).median().loc[:, 'strength']

plt.figure(figsize=(20,10))

sb.heatmap(medians.unstack(),linewidths=1, cmap='RdYlBu_r');
medians.reset_index().sort_values('strength', ascending=False).head()
joined = pd.merge(Dex[Dex['Legendary']==False],stdStats,on='Num')

medians = joined.groupby(['Type1', 'Type2']).median().loc[:, 'strength']

plt.figure(figsize=(20,10))

sb.heatmap(medians.unstack(),linewidths=1, cmap='RdYlBu_r');
medians.reset_index().sort_values('strength', ascending=False).head()
joined = pd.merge(Dex[Dex['Legendary']==False],stdStats,on='Num')

plt.figure(figsize=(20,10))

sb.heatmap( joined.groupby('Type1').median().loc[:, 'HP':'Speed'], linewidths=1,cmap='RdYlBu_r')