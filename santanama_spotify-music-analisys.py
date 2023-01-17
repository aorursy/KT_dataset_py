#I'm start in data analisys and I'm applying here some skills that I learned



import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



df = pd.read_csv('../input/top50spotify2019/top50.csv', encoding = 'latin-1', index_col=0)



#Renaming

df.columns = ['Track', 'Artist', 'Genre', 'BPM', 'Energy', 'Danceability', 'Loudness', 

               'Liveness', 'Valence', 'Length', 'Acousticness', 'Speechiness', 'Popularity']



print('setup completed')





df.head()
#Shape of dataset

print(df.shape)
#Identifying types

df.dtypes

#Turning "Popularity" into objeto

df.Popularity.astype('object')





#Most popular artist

art_pop = df.groupby('Artist')['Popularity'].mean().sort_values(ascending = False)

print(art_pop)



#Most popular genre

pop_gen = df.groupby('Genre')['Popularity'].mean().sort_values(ascending = False)

print(pop_gen)
#Music per artist



df.groupby('Artist').size()
#Artists and their most played genre 

df.groupby('Artist')['Genre'].max()
#Most popular music and its duration



mus_pop = df.groupby('Track')['Popularity','Length'].max().sort_values(by= 'Popularity', ascending= False)

print(mus_pop)



#https://docs.python.org/3/library/functions.html
#Pop artits popularity

pop_an = df.loc[df.Genre == 'dance pop']

group_pop = pop_an.groupby('Artist')['Popularity'].mean().sort_values(ascending = False)

print(group_pop)



plt.figure(figsize = (12,6))

sns.lineplot(data = group_pop)

plt.title('POP Artists Popularity')

plt.show()