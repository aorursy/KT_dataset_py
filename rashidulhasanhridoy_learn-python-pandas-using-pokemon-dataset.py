import pandas as pd
df = pd.read_csv('../input/pokemon/Pokemon.csv')
df
print(str(len(df)))
df.head(5)
df.tail(5)
df.shape
df.columns
df.dtypes
df['Name']
df['Name'][0:5]
df[['Name', 'Generation', 'Speed']]
df.iloc[1]
df.iloc[543]
df.iloc[9:12]
df.iloc[10,5]
df.iloc[9,2]
df.loc[df['Type 1'] == 'Water']
df.loc[(df['Type 1'] == 'Water') & (df['Type 2'] == 'Dark')]
df.loc[(df['Type 1'] == 'Water') & (df['Type 2'] == 'Dark') & (df['HP'] > 60)]
array = ['Dark', 'Ghost', 'Ground']

df.loc[(df['Type 1'] == 'Fire') & df['Type 2'].isin(array)]
array = ['Fire', 'Water', 'Rain']

df.loc[df['Type 1'].isin(array)]
df.describe()
df.sort_values('Name')
df.sort_values(['Name', 'HP', 'Generation'])
df.sort_values('Name', ascending = False)
df.sort_values(['Name', 'Type 1', 'Type 2'], ascending = [1, 0, 1])
df.head(5)
df['Sum'] = df['Total'] + df['HP'] + df['Attack']

df.head(5)
df['Mul'] = df['HP'] * df['Attack']

df.tail(5)
df.columns
df = df.drop(columns=['Sum'])

df.head(3)
df['Sum2'] = df.iloc[:, 4:8].sum(axis=1)

df.head(3)
df = df.drop(columns=['Sum2', 'Mul'])

df.head(5)
df.loc[df['Type 1'] == 'Dark']
df.loc[df['Type 2'] == 'Fire']
df.loc[(df['Type 1'] == 'Bug') & (df['Type 2'] == 'Fire')]
df.loc[(df['Type 1'] == 'Dark') & (df['Type 2'] == 'Fire') & (df['HP'] > 20)]
df.loc[((df['Type 1'] == 'Dark') | (df['Type 2'] == 'Dark')) & (df['Total'] > 500)]
df_new = df.loc[(df['Type 1'] == 'Dark') & (df['Type 2'] == 'Fire') & (df['HP'] > 10)]

df_new

#df_new.to_csv('New Pokemon')
df_new1 = df.loc[(df['Type 1'] == 'Dark') | (df['Defense'] >= 600)]

df_new1
df.loc[df['Name'].str.contains('Houndoom')]
df.loc[df['Name'].str.contains('ye')]
df.loc[df['Name'].str.contains('Sableye') & (df['Total'] > 390)]
df.loc[df['Name'].str.contains('Houndoom')]
df.loc[~df['Name'].str.contains('Houndoom')][0:5]
import re

df.loc[df['Type 1'].str.contains('Grass|Ghost', regex = True)]
df.loc[df['Type 1'].str.contains('poison|flying', flags = re.I, regex=True)]
df.head(5)
df.loc[df['Type 1'] == 'Grass', 'Type 1'] = 'Poison'

df.head(5)
df.loc[df['Type 1'] == 'Poison', 'Type 1'] = 'Grass'

df.head(5)
df.loc[df['HP'] >= 50, ['Total', 'Attack']] = 'Test0'

df.head(5)
df.loc[df['Total'] == 'Test0', ['Total', 'Attack']] = [12, 45]

df.head(5)
df.loc[df['Total'] == 12, ['Total', 'Attack']] = [2, 5]

df.head(5)
df.groupby('Type 1').mean()
df.groupby('Type 1').mean().sort_values('HP', ascending = False)
df.groupby('Type 1').sum()
df.groupby('Type 1').count()
df['Count'] = 1

df.groupby(['Type 1']).count()['Count']
df['Count'] = 1

df.groupby(['Type 1', 'Type 2']).count()['Count']