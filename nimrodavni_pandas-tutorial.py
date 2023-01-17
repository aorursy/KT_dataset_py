import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
import os

print(os.listdir("../input/"))
df =pd.read_csv("../input/Pokemon.csv")
df.head()
df.head(3)
df.columns
df[["Name","Type 1"]]
df = df.set_index('Name')
df.head()
df=df.drop(['#'],axis=1)

df.head()
df[df.index.str.contains("Mega")]
df.index.str.contains("Mega")
df[df.index.str.contains("Mega")].head()
df.index = df.index.str.replace(".*(?=Mega)", "")

df.head(10)
df[df.index.str.contains("Mega")].head()
df.columns = df.columns.str.upper().str.replace('_', '')

df.head()
df[df['LEGENDARY'] == True].head(20)
print('The columns of the dataset are: ',','.join(list(df.columns)))

print('The shape of the dataframe is: ',df.shape)
df.describe()
ser = df['TYPE 1']

ser
ser['Bulbasaur']
ser.index
ser.values
df['TYPE 2'].fillna(df['TYPE 1'], inplace=True) #fill NaN values in Type2 with corresponding values of Type

df.head(100)
df.loc['Bulbasaur'] #retrieves complete row data from index with value Bulbasaur
df.iloc[0] #retrieves complete row date from index 0 ; integer version of loc
# df[(df['TYPE 1']=='Fire' or df['TYPE 1']=='Dragon') and (df['TYPE 2']=='Dragon' or df['TYPE 2']=='Fire')].head(3)
df[((df['TYPE 1']=='Fire') | (df['TYPE 1']=='Dragon')) & ((df['TYPE 2']=='Dragon') | (df['TYPE 2']=='Fire'))].head(3)
print(f'The pokemon with the msot HP is {df["HP"].idxmax()} with {df["HP"].max()} HP')
df.sort_values('TOTAL',ascending=False).head()
df.sort_values(['TOTAL','ATTACK'],ascending=False).head()
print('The unique  pokemon types are',df['TYPE 1'].unique()) #shows all the unique types in column

print('The number of unique types are',df['TYPE 1'].nunique()) #shows count of unique values 
df['TYPE 1'].value_counts()
df.groupby(['TYPE 1']).size()  #same as value_counts
df.groupby(['TYPE 1']).groups # our groups and which pokemon is in which
df.groupby(['TYPE 1']).groups.keys() # only the group names
df.groupby(['TYPE 1']).first()
df.groupby(['TYPE 1'])['SPEED'].max() # Highest speed from each group
df.groupby(['TYPE 1'])['HP'].mean() # mean HP per Type 1 group
df.groupby(['TYPE 1'])['LEGENDARY'].sum() # Number of legendaries from each type
df.groupby(['TYPE 1','TYPE 2']).size()
df["LEGENDARY"] == True # the condition
df[df["LEGENDARY"] == True].head()
df.loc[df["LEGENDARY"] == True,"SPEED"].head()
df.loc[df["LEGENDARY"] == True,"SPEED"] += 10
df.loc[df["LEGENDARY"] == True,"SPEED"].head()
df.loc[df["LEGENDARY"] == True,"SPEED"] -= 10
# get the data ready

pokemon_per_generation = df["GENERATION"].value_counts().sort_index()

pokemon_per_generation
pokemon_per_generation.plot.bar()
# or, alternatively

pokemon_per_generation.plot(kind='bar')
# Add title

plt.title('Number of pokemon in each generation')



# Set our x/y labels

plt.xlabel('Generation')

plt.ylabel('Number of pokemon')



pokemon_per_generation.plot(kind='bar')

plt.show()
legendaries_per_generation = df[df['LEGENDARY'] == True]["GENERATION"].value_counts().sort_index()

non_legendaries_per_generation = df[df['LEGENDARY'] == False]["GENERATION"].value_counts().sort_index()



# Concat 2 series to a dataframe with 2 columns

pd.concat([non_legendaries_per_generation,legendaries_per_generation],axis=1,keys=['non_legendaries','legendaries']).plot.bar()



# Add title

plt.title('Number of pokemon / legendaries in each generation')



# Set our x/y labels

plt.xlabel('Generation')

plt.ylabel('Number of pokemon')



# add legend

plt.legend(('Normal', 'Legendary'))



plt.show()
plt.bar(non_legendaries_per_generation.index, non_legendaries_per_generation.values)

# we use the bottom argument to start the legendaries bar from the end of the non legendaries

plt.bar(legendaries_per_generation.index, legendaries_per_generation.values,

             bottom=non_legendaries_per_generation.values)



# Add title

plt.title('Number of pokemon / legendaries in each generation')



# Set our x/y labels

plt.xlabel('Generation')

plt.ylabel('Number of pokemon')



# add legend

plt.legend(('Normal', 'Legendary'))



plt.show()
df["TYPE 1"].value_counts().plot.barh()
df.hist(column='ATTACK')
type_gen_total=df.groupby(['GENERATION','TYPE 1']).count().reset_index()

type_gen_total=type_gen_total[['GENERATION','TYPE 1','TOTAL']]

# pivot is a very useful method, it reshapes our data based on certin columns, we specify which columns will be the index, columns and values

# in this example we make our generation the row index, the columns are each pokemon type and the values are their total in each generation

type_gen_total=type_gen_total.pivot(index='GENERATION',columns='TYPE 1',values='TOTAL')

type_gen_total.plot(marker='o')

fig=plt.gcf()

fig.set_size_inches(12,6)

plt.show()
type_gen_total[["Fire","Water","Electric","Grass","Ice","Fighting","Poison","Ground","Flying","Psychic","Bug","Rock","Ghost","Dragon","Dark","Steel","Fairy"]].plot(marker='o'

,color=["#EE8130","#6390F0","#F7D02C","#7AC74C","#96D9D6","#C22E28","#A33EA1","#E2BF65","#A98FF3","#F95587","#A6B91A","#B6A136","#735797","#6F35FC","#705746","#B7B7CE","#D685AD"])

fig=plt.gcf()

fig.set_size_inches(12,6)

plt.show()
fire = df[(df['TYPE 1']=='Fire') | ((df['TYPE 2'])=="Fire")] #fire contains all fire pokemon

water = df[(df['TYPE 1']=='Water') | ((df['TYPE 2'])=="Water")]  #all water pokemon

ax = fire.plot.scatter(x='ATTACK', y='DEFENSE', color='Red', label='Fire')

water.plot.scatter(x='ATTACK', y='DEFENSE', color='Blue', label='Water', ax=ax);
strong=df.sort_values(by='TOTAL', ascending=False) #sorting the rows in descending order

strong.drop_duplicates(subset=['TYPE 1'],keep='first') #since the rows are now sorted in descending oredr

#thus we take the first row for every new type of pokemon i.e the table will check TYPE 1 of every pokemon

#The first pokemon of that type is the strongest for that type

#so we just keep the first row
df['TYPE 1'].value_counts().plot.pie()
# Add title

plt.title('Number of pokemon in each generation')



# Set our x/y labels

plt.xlabel('Generation')

plt.ylabel('Number of pokemon')

sns.barplot(pokemon_per_generation.index,pokemon_per_generation.values)



plt.show()
# Add title

plt.title('Number of pokemon in each generation')



# Set our x/y labels

plt.xlabel('Generation')

plt.ylabel('Number of pokemon')

#sns.barplot(pokemon_per_generation.index,pokemon_per_generation.values)

sns.countplot('GENERATION',data=df,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))



plt.show()
plot_data = df["TYPE 1"].value_counts()

sns.barplot(plot_data.values,plot_data.index,palette='plasma')
sns.distplot(df['ATTACK'])
plot_data = df.groupby(['GENERATION','TYPE 1']).count().reset_index()

sns.lineplot(x='GENERATION',y='TOTAL',hue='TYPE 1',data=plot_data,marker='o')

fig=plt.gcf()

fig.set_size_inches(15,9)

plt.legend(loc='upper right')

plt.show()
fire_and_water = df[(df['TYPE 1'].isin(['Fire','Water']))]

sns.scatterplot(x='ATTACK', y='DEFENSE',data=fire_and_water, hue='TYPE 1')
df2=df.drop(['GENERATION','TOTAL'],axis=1)

sns.boxplot(data=df2)

plt.ylim(0,200)  #change the scale of y axix

plt.show()
plt.subplots(figsize = (15,5))

plt.title('Attack by Type1')

sns.boxplot(x = "TYPE 1", y = "ATTACK",data = df)

plt.ylim(0,200)

plt.show()
plt.subplots(figsize = (15,5))

plt.title('Attack by Type2')

sns.boxplot(x = "TYPE 2", y = "ATTACK",data=df)

plt.show()
plt.subplots(figsize = (15,5))

plt.title('Defence by Type')

sns.boxplot(x = "TYPE 1", y = "DEFENSE",data = df)

plt.show()
plt.subplots(figsize = (20,10))

plt.title('Attack by Type1')

sns.violinplot(x = "TYPE 1", y = "ATTACK",data = df)

plt.ylim(0,200)

plt.show()
plt.subplots(figsize = (20,10))

plt.title('Attack by Type1')

sns.violinplot(x = "TYPE 1", y = "DEFENSE",data = df)

plt.ylim(0,200)

plt.show()
plt.subplots(figsize = (15,5))

plt.title('Strongest Genaration')

sns.violinplot(x = "GENERATION", y = "TOTAL",data = df)

plt.show()
plt.figure(figsize=(10,6)) #manage the size of the plot

sns.heatmap(df.corr(),annot=True) #df.corr() makes a correlation matrix and sns.heatmap is used to show the correlations heatmap

plt.show()