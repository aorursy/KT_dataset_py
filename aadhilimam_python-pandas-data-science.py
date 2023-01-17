import pandas as pd
df = pd.read_csv('../input/pokemon.csv')
#print the top rows

print(df.head())
#print the bottom 3 rows

df.tail(3)
#Read the headers

df.columns
#Reach each column

df[['Name','Type 1','HP']][0:5]
#Reach Each row

df.iloc[3] 
#Reach Each row with range

df.iloc[3:10] 
#Read each row with range 

for index , row in df.iterrows():

    print(index,row[['Name' , 'Total']])
#Read the specific location

df.iloc[2,1]
df.loc[df['Type 1'] == 'Fire']
#describe data

df.describe()
df.sort_values(['Type 1' , 'HP'], ascending=True)
df['Total'] = df['HP'] + df['Attack'] + df['Defense'] + df['Sp. Atk'] + df['Sp. Def'] + df['Speed']  
df.head(5)
#drop the column / delete the column

df = df.drop(columns= ['Total'])
df.head()
#create column using iloc

df['Total'] = df.iloc[:,4:10].sum(axis=1)
df.head()
#arrange the Dataframe

cols = list(df.columns)

df = df[cols[0:4] + [cols[-1]] + cols[4:12]]

df.head()
#exporting the new dataframe to csv

df.to_csv('../input/modified.csv', index=False)
#exporting the new dataframe to exel

df.to_excel('modified_exel.xlsx' , index=False)
#exporting the new dataframe to text seperated by text

df.to_csv('modified_text.txt' , index=False , sep='\t')
df.loc[(df['Type 1'] == 'Grass') & (df['Type 2'] == 'Poison')]
df.loc[(df['Type 1'] == 'Grass') | (df['Type 2'] == 'Poison')]
new_df = df.loc[(df['Type 1'] == 'Grass') & (df['Type 2'] == 'Poison') & (df['HP'] > 70) ]

new_df
#resetting the index 

new_df = new_df.reset_index(drop=True)

new_df
df.loc[df['Name'].str.contains('Mega')]
#drop the name mega

df.loc[~df['Name'].str.contains('Mega')]
import re

df.loc[df['Type 1'].str.contains('fire|grass', flags= re.I , regex = True)]
#Start Names with 'pi'

df.loc[df['Name'].str.contains('^pi[a-z]*', flags= re.I , regex = True)]
df.loc[df['Type 1'] == 'Flamer' , 'Type 1'] = 'Fire' 
df
df.loc[df['Type 1'] == 'Flamer' , 'Legendary'] = True

df
df = pd.read_csv('modified.csv')
df.loc[df['Total'] > 500 , ['Generation','Legendary']] = ['Test 1 ','Test 2']
df
df = pd.read_csv('modified.csv')

df.groupby(['Type 1']).mean()
df.groupby(['Type 1']).mean().sort_values('Attack' , ascending=False)
df.groupby(['Type 1']).sum()
df.groupby(['Type 1']).count()
df['count'] = 1

df.groupby(['Type 1' , 'Type 2']).count()['count']