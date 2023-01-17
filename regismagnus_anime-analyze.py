import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

import seaborn as sns



from sklearn.preprocessing import LabelEncoder
anime_data = pd.read_csv('../input/anime-recommendations-database/anime.csv', index_col='anime_id')

anime_data.info()
#anime_data['rating'].fillna(-1, inplace=True)

#change Unknown as NaN

anime_data.replace("Unknown", np.nan, inplace=True)

#remove all NaN values

anime_data.dropna(inplace=True)



#convert episodes to numeric

anime_data['episodes']=pd.to_numeric(anime_data['episodes'])



total_rows=anime_data.shape[0]
#capture all genres

genres=[]

for group_genres in anime_data['genre']:

    if not pd.isna(group_genres):

        split_genres=group_genres.split(',')

        for genre in split_genres:

            genre=genre.strip()

            if genres.count(genre)==0:

                genres.append(genre)

#create cols by genres

for genre in genres:

    anime_data['genre_'+genre]=[False for i in range(anime_data.shape[0])]
def set_genre(row):

    '''set True in the col of relative genre'''

    if not pd.isna(row['genre']):

        genres=row['genre'].split(',')

        for genre in genres:

            genre=genre.strip()

            row['genre_'+genre] = True

    return row

#set genre True in correct places

anime_data = anime_data.apply(set_genre, axis=1)
print(anime_data['type'].unique())

print(genres)
anime_data.head()
for p in range(7, 2, -1):

    total=anime_data.query("rating <"+str(p))['rating'].count()

    print("Anime less than rating %.1f: %d relative the %.2f%% of the data" % (p, total, ((total*100)/total_rows)))
total=anime_data.query("rating >=7")['rating'].count()

print("Anime more or equal than rating 7.0: %d relative the %.2f%% of the data" % (total, ((total*100)/total_rows)))

for p in range(8, 10):

    total=anime_data.query("rating >"+str(p))['rating'].count()

    print("Anime more than rating %.1f: %d relative the %.2f%% of the data" % (p, total, ((total*100)/total_rows)))
total=anime_data.query("rating<=7 and rating>=6")['rating'].count()

print('Anime with rating between 7 and 6 inclusive: %d relative the %.2f%% of the data' % (total, ((total*100)/total_rows)))
plt.figure(figsize=(14,6))

plt.title("Compare Rating per Members")

sns.lineplot(x=anime_data['rating'], y=anime_data['members'])
plt.figure(figsize=(14,6))

plt.title("Compare Type per Members")

sns.barplot(x=anime_data['type'], y=anime_data['members'])
plt.figure(figsize=(14,6))

plt.title("Compare Rating per Members by each type")



for val in anime_data['type'].unique():

    sns.lineplot(x=anime_data[anime_data['type']==val]['rating'], y=anime_data[anime_data['type']==val]['members'], label=val)

plt.legend()
plt.figure(figsize=(14,6))

plt.title("Compare Episodes per Members by each type (without TV)")



for val in anime_data['type'].unique():

    if val=='TV':

        continue

    sns.lineplot(x=anime_data[anime_data['type']==val]['rating'], y=anime_data[anime_data['type']==val]['members'], label=val)

plt.legend()
plt.figure(figsize=(14,6))

plt.title("Compare Episodes per Members")

sns.lineplot(x=anime_data['episodes'], y=anime_data['members'])
anime_data.describe()
anime_data[anime_data['episodes']>1000]
anime_data[anime_data['rating']==10].head()
#create a new database based in genre

genre_cols=['genre', 'rating', 'members', 'episodes']

genre_data=pd.DataFrame(columns=genre_cols)

for genre in genres:

    col_name="genre_" + genre

    aux_data = anime_data.groupby(col_name).agg({ 'rating': 'mean', 'members': 'sum', 'episodes': 'sum' })

    aux_data['genre']=genre

    genre_data=genre_data.append(aux_data[aux_data.index==True][genre_cols], ignore_index=True)

genre_data.set_index('genre', inplace=True)



#I don't know why, but members is object in this part

genre_data['members']=pd.to_numeric(genre_data['members'])

genre_data['episodes']=pd.to_numeric(genre_data['episodes'])
genre_data.info()
genre_data_top = genre_data.sort_values('rating', ascending=False)[:10]

plt.figure(figsize=(14,6))

plt.title("Genrer per rating Top 10")

sns.barplot(x=genre_data_top.index, y=genre_data_top['rating'])
genre_data_members_top = genre_data.sort_values('members', ascending=False)[:10]



plt.figure(figsize=(14,6))

plt.title("Genrer per members Top 10")

sns.barplot(x=genre_data_members_top.index, y=genre_data_members_top['members'])