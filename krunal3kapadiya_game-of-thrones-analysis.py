import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

sns.set(style="darkgrid")
# importing the libraries
df_battles = pd.read_csv("../input/battles.csv")
df_character_predictions = pd.read_csv("../input/character-predictions.csv")
df_character_deaths = pd.read_csv("../input/character-deaths.csv")
# displaying data deaths
df_battles.head()
df_battles = df_battles.drop(['attacker_2', 'attacker_3', 'attacker_4', 'defender_2', 'defender_3', 'defender_4', 'note'], axis=1)
sum(df_battles.duplicated())
df_battles.isnull().sum()
# selecting data and run the model
df_battles_per_year = df_battles.groupby('year',as_index=False).sum()
plt.bar(df_battles_per_year['year'], df_battles_per_year['battle_number'])
plt.title('Number of battles per years')
plt.xlabel('Years')
plt.ylabel('Number of battles')
most_pop_genres = df_battles['attacker_commander'].str.cat(sep=', ').split(', ')
most_pop_genres = pd.Series(most_pop_genres).value_counts(ascending=False) 
graph = most_pop_genres.plot.bar()
graph.set_title("Most Attacker commander", fontsize=18, fontweight='bold')
graph.set_xlabel("Number of Battles", fontsize=16)
graph.set_ylabel("List of Commanders", fontsize=16)
graph.set_xlim(right=10)
graph.legend(['Commander'], loc = "upper right")
most_pop_genres = df_battles['defender_commander'].str.cat(sep=', ').split(', ')
most_pop_genres = pd.Series(most_pop_genres).value_counts(ascending=False) 
graph = most_pop_genres.plot.bar()
graph.set_title("Most Defender commander", fontsize=18, fontweight='bold')
graph.set_xlabel("Number of Battles", fontsize=16)
graph.set_ylabel("List of Commanders", fontsize=16)
graph.set_xlim(right=10)
graph.legend(['Commander'], loc = "upper right")
df_battles["region"].value_counts().plot(kind = 'bar')
df_battles['location'].value_counts().plot(kind = 'bar')
df_battles['attacker_king'].value_counts().plot(kind = 'bar')
df_battles['defender_king'].value_counts().plot(kind = 'bar')
df_battles['battle_type'].value_counts().plot(kind = 'bar')
df_battles.groupby('year').sum()[['major_death','major_capture']].plot.bar(rot=0)