import pandas as pd

import seaborn as sns

import matplotlib as plt

sns.set_style("whitegrid")
df = pd.read_csv('../input/ign.csv')

df['genre'].value_counts()

#As we can see, strategy genre is one of the most numerous genres in IGN database, securing top-6 in our database.
#Lets compare scores of all games and of stratefy titles

scores = df['score_phrase'].value_counts()

scores.plot.pie(figsize=(6,6))
df_strategy = df[(df['genre']=='Strategy')]

strategy_scores = df_strategy['score_phrase'].value_counts()

strategy_scores.plot.pie(figsize=(6,6))

#As we can see, strategy games have more titles with mark 'Good' and 'Great'

#comparing means

print('All Games Mean Score: {0}, Strategy Genre Mean Score: {1}'.format(df['score'].mean(),df_strategy['score'].mean()))
table_strategy = df_strategy.groupby(["genre", "release_year"]).size().unstack().T

table_strategy.reset_index(inplace = True)



table_strategy[["Strategy"]].plot(x = table_strategy['release_year'])
