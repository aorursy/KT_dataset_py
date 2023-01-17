import pandas as pd
pd.set_option('max_columns', None)
pokemon = pd.read_csv("../input/pokemon/pokemon.csv")
pokemon.head(3)
pokemon['type1'].value_counts().plot.bar()
pokemon['hp'].value_counts().sort_index().plot.line()
pokemon[pokemon['weight_kg']<600]['weight_kg'].plot.hist()
#pokemon['weight_kg'].plot.hist()