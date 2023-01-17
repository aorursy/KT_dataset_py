import pandas as pd
obj = pd.Series([4, 7, -5, 3])

obj
animales = ['Tortuga', 'Zorro', 'Paloma', 'Elefante']

tipo = ['reptil', 'mamífero', 'ave', 'mamífero']

obj = pd.Series(tipo, index=animales)

obj
obj[0]
obj['Tortuga']
d = {'tipo_vivienda': ['casa', 'departamento'],

     'm2': [125, 59],

     'Barrio': ['San Martin', 'Florida'],

     'Precio (kUSD)': [200, 130]

    }

df = pd.DataFrame(data=d)

df
df = pd.read_csv('../input/FIFA 2018 Statistics.csv')

df
df = pd.read_table('../input/FIFA 2018 Statistics.csv', sep=',')

df
pd.read_table('../input/FIFA 2018 Statistics.csv', sep=',', header=None)
pd.read_table('../input/FIFA 2018 Statistics.csv', sep=',', header=None, skiprows=1, nrows=3)
df = pd.read_csv("../input/FIFA 2018 Statistics.csv")
df.head()
df.tail()
df.columns
df.describe()
df.info()
df.shape
df.duplicated()
df.Team.unique()
df.Team.value_counts()
df["Team"].head(3)
df.Team.head(3)
df.iloc[3,:]
df.iloc[:,1]
df.iloc[7:10, 1:6]
sel = df.columns.isin(["Team", "Opponent"])

print(sel)

df.iloc[:6, sel]
df.loc[[0,1,2,3,4,5],['Team', 'Opponent']]
df.loc[ df["Ball Possession %"] > 70 ]
df.loc[(df.index >= 1) & (df.index < 7)]
df.sort_values(['Date', 'Team'])