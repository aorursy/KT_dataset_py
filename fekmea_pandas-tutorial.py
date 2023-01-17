import pandas as pd



df = pd.read_csv('../input/pokemon/PokemonData.csv')

df.head(5)



# df_xlsx = pd.read_excel('pokemon_data.xlsx')

# print(df_xlsx.head(3))



# df = pd.read_csv('pokemon_data.txt', delimiter='\t')



# print(df.head(5))







df['HP'].head(5)
#### Read Headers

df.columns



## Read each Column

#df[['Name', 'Type 1', 'HP']].head(5)



## Read Each Row# 

#df.iloc[0:4]

#for index, row in df.iterrows():

#    print(index, row['Name'])

#df.loc[df['Type 1'] == "Grass"]



## Read a specific location (R,C)

#print(df.iloc[2,1])

#df.describe()

#df.sort_values(['Type 1' ,'HP'] , ascending=[1,0])



df.sort_values(['Type1', 'HP'], ascending=[1,0])



df
df['Total'] = df['HP'] + df['Attack'] + df['Defense']



#df = df.drop(columns=['Total'])



#df['Total'] = df.iloc[:, 4:10].sum(axis=1)



#cols = list(df.columns)

#df = df[cols[0:4] + [cols[-1]]+cols[4:12]]



df.head(5)
#df.to_csv('modified.csv', index=False)



#df.to_excel('modified.xlsx', index=False)



df.to_csv('../PokemonData_output.txt', index=False, sep='\t')





new_df = df.loc[(df['Type1'] == 'Grass') & (df['Type2'] == 'Poison') & (df['HP'] > 70)]



new_df.reset_index(drop=True, inplace=True)



import re

df.loc[df['Name'].str.contains('^pi[a-z]' , flags=re.I , regex=True)]

new_df

new_df.to_csv('filtered.csv')









df.loc[df['Total'] > 500, ['Generation','Legendary']] = ['Test 1', 'Test 2']



 



#df = pd.read_csv('modified.csv')



df
#df = pd.read_csv('filtered.csv')



#df['count'] = 1



#df.groupby(['Type1', 'Type2']).count()['count']








