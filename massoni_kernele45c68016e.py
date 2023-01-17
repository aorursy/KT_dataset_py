import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

import seaborn as sns



from surprise import Reader, Dataset, KNNBasic

from surprise.model_selection import train_test_split

from surprise import Dataset

from surprise import accuracy
hits = pd.read_csv('../input/hits.csv')

musics = pd.read_csv('../input/music_data.csv')

genre = pd.read_csv('../input/genre.csv')

state = pd.read_csv('../input/state.csv')

target = pd.read_csv('../input/target.csv')
hits.columns
musics.columns
genre.columns
state.columns
musics.head()
hits.head()
df = hits.merge(genre, on='genre_id').merge(state, on='state_id').merge(musics, on='music_id')
df.head()
df.groupby(by="genre")["user_id"].count().sort_values(ascending=False)
plt.figure(figsize=(15,10))

df.groupby(by="genre")["user_id"].count().sort_values(ascending=False).plot.bar()

plt.xticks(rotation=50)

plt.xlabel("Gênero")

plt.ylabel("Número de plays")

plt.show()
df.groupby(by="state")["user_id"].count().sort_values(ascending=False)
plt.figure(figsize=(15,10))

df.groupby(by="state")["user_id"].count().sort_values(ascending=False).plot.bar()

plt.xticks(rotation=50)

plt.xlabel("Estados")

plt.ylabel("Número de plays")

plt.show()
corrmat = df.corr()

sns.set(font_scale=1)

fig, ax = plt.subplots(figsize=(10,10))

sns.heatmap(corrmat, vmax=1, vmin=-1, square=True, annot=True);
fig, ax = plt.subplots(figsize=(10,10))



ax.scatter(df['duration'], df['value'])



ax.set_title('Music Dataset')

ax.set_xlabel('Duration')

ax.set_ylabel('Value')
fig, ax = plt.subplots(figsize=(10,10))



ax.scatter(df['duration'], df['plays'])



ax.set_title('Music Dataset')

ax.set_xlabel('Duration')

ax.set_ylabel('Plays')
fig, ax = plt.subplots(figsize=(20,10))



ax.hist(df['duration'])



ax.set_title('Duration Frequency')

ax.set_xlabel('Duration')

ax.set_ylabel('Frequency')
fig, ax = plt.subplots(figsize=(20,10))



ax.hist(df['id_artist'])



ax.set_title('Artist Frequency')

ax.set_xlabel('Artist')

ax.set_ylabel('Frequency')
df.query('duration==0')
df = df.drop(df.loc[df.duration == 0].index)

df.query('duration==0')
df["rating"] = df.value / df.duration

df_grouped = df.groupby(by=["user_id","music_id"])["rating"].sum().reset_index()
df_grouped.head()
fig, ax = plt.subplots(figsize=(20,10))



ax.hist(df_grouped['rating'])



ax.set_title('Rating Frequency')

ax.set_xlabel('Rating')

ax.set_ylabel('Frequency')
fig, ax = plt.subplots(figsize=(10,10))



ax.scatter(df['duration'], df['rating'])



ax.set_title('Music Dataset')

ax.set_xlabel('Duration')

ax.set_ylabel('Rating')
targetList = [int(i) for i in target.user_id]
print(targetList)
reader = Reader()

data = Dataset.load_from_df(df_grouped[['user_id', 'music_id', 'rating']],reader)

    

trainset, testset = train_test_split(data, test_size=.20)



algo = KNNBasic()



algo.fit(trainset)

predictions = algo.test(testset)



accuracy.rmse(predictions)

    

musics = df_grouped.music_id.unique()

thisdict = {}
trainset = data.build_full_trainset()

algo.fit(trainset)
def get_recommendations(n):

    for userID in targetList:

        lista = list([])

        lista_aux = list([])

        for music in musics:

            aux = algo.predict(userID,music)

            lista.append((aux[1],aux[3]))

        lista.sort(key=lambda tup: tup[1],reverse=True)

        for i in range(n):

            lista_aux.append(lista[i][0])

        thisdict[userID] = list(lista_aux)

        

    return thisdict
result = get_recommendations(5)
print(result)