import pandas as pd



df = pd.read_csv('../input/pokemon_data.csv')



print(df.head(10))
# Read columns

df.columns



print(df[['Name', 'Type 1', 'HP']][0:10])
# Read rows



# Read a specific row

print(df.iloc[0:4])
# Read a specific Location (Row, Column)

print(df.iloc[3,1])
for index, row in df.iterrows():

  # print(index, row)

    print(index, row['Name'])
# Load data by specific by category

df.loc[df['Type 1'] == 'Grass']
df.sort_values('Name', ascending=False)
df.sort_values(['Type 1', 'HP'], ascending=[1, 0])