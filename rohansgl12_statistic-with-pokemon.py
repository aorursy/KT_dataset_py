# importing all important packages

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



plt.style.use('bmh')
dfP = pd.read_csv('../input/pokemon/Pokemon.csv')# reading csv file and saving it to a variable





dfP.head() #printing first 5 entries of dataset
dfP.info() #describing every column type and entries
print(dfP['Attack'].mean())

dfP['Name'].where(dfP['Attack'] < dfP['Attack'].mean()) #printing names which have attack greater than average
dfP.set_index('Name' , inplace = True)# setting index to name column
## The index of Mega Pokemons contained extra and unneeded text. Removed all the text before "Mega"  

dfP.index = dfP.index.str.replace(".*(?=Mega)", "")

dfP.head(10)
dfP_missing = dfP.isna() # if True means value in the dataframe is Nan or Null

for col in dfP_missing.columns:

    print(dfP_missing[col].value_counts())

    print('\n')

#we can see in column Type2 386 values are Nan    

    
dfP['Type 2'].fillna(dfP['Type 1'] , inplace= True)
dfP.drop(['#'] , axis = 1 , inplace=True) # drop the column or axis=1, row or axis =0

#you can use the above syntax with inplace = True or dfP = dfP.drop(['#'] , axis =1,inplace=False)
dfP.loc[['Venusaur','Bulbasaur']] #retrieves complete row information from index with value Bulbasaur and Venusaur

print(dfP.iloc[0:1,])#retrieves complete row date from index 0 ; integer version of loc
#filtering pokemons using logical operators

dfP[((dfP['Type 1']=='Grass') | (dfP['Type 2'] == "Poison") )].head()
print("MAx HP:",dfP['HP'].argmax())  #returns the pokemon with highest HP

dfP.sort_values(by=['Total'] , ascending=True , axis=0)#this arranges the dataframe in ascending order of the Totals
print('The unique  pokemon types are',dfP['Type 1'].unique()) #shows all the unique types in column

print('The number of unique types are',dfP['Type 1'].nunique()) #shows count of unique values 
df_summary = dfP.describe() #summary of the pokemon dataframe

df_summary
bins = range(0,200,10)

plt.hist(dfP["Attack"] , bins , histtype='bar',rwidth=1.4 , color='#808000')

plt.xlabel('Attack')

plt.ylabel('Count')

plt.plot()

plt.axvline(dfP['Attack'].mean() , linestyle = 'dashed' , color = 'r')

plt.show()
water = dfP[(dfP['Type 1'] == 'Water') | (dfP['Type 2'] == 'Water')]

grass = dfP[(dfP['Type 1'] == 'Grass') | (dfP['Type 2'] == 'Grass')]



plt.scatter(water.Defense.head(50) , water.Speed.head(50) , color = 'B',label = 'Water' , marker = '*')

plt.scatter(grass.Defense.head(50),grass.Speed.head(50),color='G',label="Grass")



plt.xlabel('Defense')

plt.ylabel('Speed')

plt.legend()

plt.plot()

plt.show()

labels = 'Water', 'Normal', 'Grass', 'Bug', 'Psychic', 'Fire', 'Electric', 'Rock', 'Other'

sizes = [112, 98, 70, 69, 57, 52, 44, 44, 175]

colors = ['Y', 'B', '#00ff00', 'C', 'R', 'G', 'silver', 'white', 'M']

explode = (0, 0, 0.2, 0, 0.1, 0, 0, 0, 0)  # only "explode" the 3rd slice 

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=90)

plt.axis('equal')

plt.title("Percentage of Different Types of Pokemon")

plt.plot()

fig=plt.gcf()

fig.set_size_inches(7,7)

plt.show()
df = dfP.drop(['Generation' , 'Total'] , axis = 1)

sns.boxplot(data = df)

plt.ylim(0,300) #changing the scale of y axis

plt.show()
plt.subplots(figsize = (15,5))

plt.title('Attack by Type1')

sns.boxplot(x = "Type 1", y = "Attack",data = df)

plt.ylim(0,200)

plt.show()
plt.subplots(figsize = (15,5))

plt.title('Attack by Type2')

sns.boxplot(x = "Type 2", y = "Attack",data=df)

plt.show()
plt.subplots(figsize = (15,5))

plt.title('Defence by Type')

sns.boxplot(x = "Type 1", y = "Defense",data = df)

plt.show()
plt.figure(figsize=(10,6)) #resize  the plot

sns.heatmap(dfP.corr(),annot=True) #df.corr() makes a correlation matrix and sns.heatmap is used to show the correlations heatmap

plt.show()
