import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

import seaborn as sns

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

pokemons=pd.read_csv('../input/pokemon/Pokemon.csv')
pokemons.sample(7)
pokemons.columns
pokemons.info()
del pokemons['Type 2']

pokemons.rename(columns={'Type 1':'Type'},inplace=True)
pokemons.head()
pokemons.describe()
len(pokemons.Name.unique())

# We got 800 DIFFERENT pokemons.
pokemons[pokemons.duplicated()]

#We don't have duplicated values in our data frame.
len(pokemons['Type'].unique())

#There are 18 different types  of pokemons
pokemons['Generation'].value_counts()
sns.countplot(x='Generation',data=pokemons,palette='nipy_spectral')

plt.title('Number of pokemons grouped by generation')
pokemons.Type.value_counts()
plt.figure(figsize=(15,5))

sns.countplot(pokemons.Type,palette='twilight')
sns.set_style('darkgrid')

plt.figure(figsize=(10,6))

sns.boxplot(data=pokemons.drop(['#','Total','Generation','Legendary'],axis=1),fliersize=3,palette='seismic')

plt.title('Boxplots for stats')
plt.figure(figsize=(10,6))

sns.violinplot(data=pokemons.drop(['#','Total','Generation','Legendary'],axis=1),palette='rocket')

plt.title('Violinplots for stats')
pokemons.groupby('Type').sum()
pokemons.groupby('Type').sum().HP



# In this series the types are alphabetically ordered.
pokemons['Type'].unique()
list_types=pokemons['Type'].unique().tolist() # Convert the array of types into a list

list_types.sort() # Sorting the list of strings alphabetically

list_types
plt.style.use('ggplot')

plt.style.use('seaborn-darkgrid')



stats=pokemons[['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']]

k=1

m=0

palette=['magma','ocean','vlag','copper','mako','winter']

plt.figure(figsize=(17,17))

for i in stats:

    plt.subplot(3,2,k)

    k=k+1

    sns.barplot(x=pokemons.groupby('Type').sum()[i],y=list_types,palette=palette[m])

    m=m+1

    plt.title(str('Total of '+i))
k=1;

m=0;

plt.figure(figsize=(15,30))

for i in stats:

    plt.subplot(6,1,k);

    k=k+1;

    sns.stripplot(x=pokemons.Type,y=pokemons[i],palette='Dark2');

    plt.title(str('Total of ')+i + str(' for each type'))
k=1;

plt.figure(figsize=(17,22))

for i in list_types:

    plt.subplot(6,3,k);

    k=k+1;

    sns.barplot(x=pokemons[pokemons.Type==i].sum().drop(['#','Name','Type','Generation','Legendary','Total']).values,

                y=pokemons[pokemons.Type==i].sum().drop(['#','Name','Type','Generation','Legendary','Total']).index,

                palette='inferno');

    plt.title(i)

    plt.xlim(0,8500)

        
pok_melt=pd.melt(pokemons,id_vars=['Name','Type','Legendary'],value_vars=['HP','Defense','Attack','Sp. Atk','Sp. Def','Speed'])

pok_melt.head()
plt.figure(figsize=(17,22))

k=1

for i in list_types:

    plt.subplot(6,3,k)

    k=k+1

    sns.swarmplot(x=pok_melt.variable,y=pok_melt[pok_melt.Type==i].value,palette='gist_stern')

    plt.title(i)

    plt.xlabel('')
df=pd.DataFrame()

for i in stats:

    df[i]=pokemons.groupby('Type').describe()[i]['mean']
df
plt.figure(figsize=(16,20))

k=1

m=0

for i in stats:

    plt.subplot(3,2,k)

    k=k+1

    sns.barplot(x=df[i],y=df.index,palette=palette[m])

    m=m+1

    plt.title(str('Mean of total ')+ i +str(' for each type'))

    plt.xlabel(i)
k=1;

plt.figure(figsize=(16,25))

for i in list_types:

    plt.subplot(6,3,k);

    k=k+1;

    sns.barplot(x=df.loc[i,:].values,y=df.loc[i,:].index, palette='Paired');

    plt.title(i)

    plt.xlim(0,130)

    plt.ylabel('Mean')
plt.figure(figsize=(15,5))

sns.barplot(x=pokemons.groupby('Type').sum().Total.sort_values(ascending=False).index

            ,y=pokemons.groupby('Type').sum().Total.sort_values(ascending=False),palette='cool')

plt.title('Total of all stats for each type of pokemon')
plt.figure(figsize=(15,5))

sns.barplot(x=pokemons.groupby('Type').mean().Total.sort_values(ascending=False).index,

            y=pokemons.groupby('Type').mean().Total.sort_values(ascending=False).values,palette='twilight_shifted')

plt.title('Mean of the total of all stats for each type of pokemon')
best_stats=[]

for i in list_types:

    best_stats.append(df.loc[i,:].sort_values(ascending=False).index[0])
m=0

for k in best_stats:

    print('Best stat of type ',list_types[m],' is ',k)

    m=m+1
pokemons[pokemons.Name.str.contains('Mega')]
mega_pokemons = ['Mega'+poke.split('Mega')[1] for poke in pokemons[pokemons.Name.str.contains('Mega')].Name]

mega_pokemons
pokemons=pokemons.replace(to_replace=pokemons[pokemons.Name.str.contains('Mega')].Name.values,value=mega_pokemons)
for n in list_types:

    print(str('TYPE ')+n.upper())

    for i in stats:

        name=pokemons[(pokemons.Type==n)].sort_values(by=i,ascending=False).Name.values[0]

        print(str('Best ')+i+(' pokemon is ')+name)

    print('*****************************************')

sns.countplot(x='Generation',data=pokemons,palette='seismic')

plt.title('Number of pokemons grouped by generation')

plt.ylabel('Number of pokemons')
pokemons.groupby('Generation').sum()
plt.figure(figsize=(15,15))

k=1

for i in stats:

    plt.subplot(3,2,k)

    x=sns.swarmplot(x='Generation',y=i,data=pokemons,palette='plasma')

    k=k+1

    plt.title(i+str(' for each generation'))

    
k=1

plt.figure(figsize=(17,15))

for i in stats:

    plt.subplot(3,2,k)

    sns.boxplot(y=pokemons[i],x=pokemons.Generation)

    k=k+1

    plt.title(i+str(' for each generation'))
from bokeh.io import output_notebook, show, output_file

from bokeh.plotting import figure

from bokeh.layouts import row, gridplot

output_notebook()



p1=figure(plot_width=400,plot_height=200,title='HP for each generation')

p1.circle(x=[1,2,3,4,5,6],y=pokemons.groupby('Generation').sum().HP,size=3,color='red')

p1.line(x=[1,2,3,4,5,6],y=pokemons.groupby('Generation').sum().HP,line_width=1,color='red')



p2=figure(plot_width=400,plot_height=200,title='Attack for each generation')

p2.circle(x=[1,2,3,4,5,6],y=pokemons.groupby('Generation').sum().Attack,size=3,color='red')

p2.line(x=[1,2,3,4,5,6],y=pokemons.groupby('Generation').sum().Attack,line_width=1,color='red')

    

p3=figure(plot_width=400,plot_height=200,title='Defense for each generation')

p3.circle(x=[1,2,3,4,5,6],y=pokemons.groupby('Generation').sum().Defense,size=3,color='red')

p3.line(x=[1,2,3,4,5,6],y=pokemons.groupby('Generation').sum().Defense,line_width=1,color='red')



p4=figure(plot_width=400,plot_height=200,title='Sp. Atk for each generation')

p4.circle(x=[1,2,3,4,5,6],y=pokemons.groupby('Generation').sum()['Sp. Atk'],size=3,color='red')

p4.line(x=[1,2,3,4,5,6],y=pokemons.groupby('Generation').sum()['Sp. Atk'],line_width=1,color='red')



p5=figure(plot_width=400,plot_height=200,title='Sp. Def for each generation')

p5.circle(x=[1,2,3,4,5,6],y=pokemons.groupby('Generation').sum()['Sp. Def'],size=3,color='red')

p5.line(x=[1,2,3,4,5,6],y=pokemons.groupby('Generation').sum()['Sp. Def'],line_width=1,color='red')



p6=figure(plot_width=400,plot_height=200,title='Speed for each generation')

p6.circle(x=[1,2,3,4,5,6],y=pokemons.groupby('Generation').sum().Speed,size=3,color='red')

p6.line(x=[1,2,3,4,5,6],y=pokemons.groupby('Generation').sum().Speed,line_width=1,color='red')



grid=gridplot([p1,p2,p3,p4,p5,p6],ncols=2)

show(grid)
len(pokemons[pokemons.Legendary==True])

# There are 65 Legendary pokemons

# 8.125% pokemons are Legendary
pokemons.groupby('Generation').sum().Legendary

# Generations 3 ,5 & 4 have the most legendary pokemons
sns.barplot(x=pokemons.groupby('Generation').sum().Legendary.index,

            y=pokemons.groupby('Generation').sum().Legendary.values,palette='CMRmap')
pokemons.groupby('Type').sum().Legendary.sort_values(ascending=False)
plt.figure(figsize=(15,10))

sns.barplot(x=pokemons.groupby('Type').sum().Legendary.sort_values(ascending=False).index,

              y=pokemons.groupby('Type').sum().Legendary.sort_values(ascending=False).values,palette='Paired')
k=1;

m=0;

plt.figure(figsize=(15,30))

for i in stats:

    plt.subplot(6,1,k);

    k=k+1;

    sns.swarmplot(x='Type',y=i,palette='Dark2',hue='Legendary',data=pokemons);

    plt.title(str('Total of ')+i + str(' for each type'))
plt.figure(figsize=(17,22))

k=1

for i in list_types:

    plt.subplot(6,3,k)

    k=k+1

    sns.swarmplot(x=pok_melt.variable,y=pok_melt[pok_melt.Type==i].value,palette='Dark2',

                  hue=pok_melt[pok_melt.Type==i].Legendary)

    plt.title(i)

    plt.xlabel('')

    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.102), loc='lower left',ncol=2, mode="expand", borderaxespad=0.)

   
legend=pokemons[pokemons.Legendary==True]

for i in stats:

    print('Number of legendary pokemons with ',i, ' higher than the average:',

          len(legend[legend[i]>pokemons[i].mean()]),'\nPercentage:', round(len(legend[legend[i]>pokemons[i].mean()])/65*100,2),

           '\n**************')
plt.figure(figsize=(15,10))

sns.heatmap(pokemons.drop(['#'],axis=1).corr(),annot=True,cmap="YlGnBu")
sns.pairplot(pokemons.drop(['#','Legendary','Generation'],axis=1))