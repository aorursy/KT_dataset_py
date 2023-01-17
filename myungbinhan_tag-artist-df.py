import pandas as pd

import numpy as np



df_train = pd.read_json('../input/arena-res/train.json')

df_song_meta = pd.read_json('../input/arena-res/song_meta.json')



df_train_explode = df_train.explode('tags').explode('songs')

df_song_meta_explode = df_song_meta.explode('artist_name_basket')



tag_list = list(df_train['tags'].explode().value_counts().index)

tag_list = pd.DataFrame(tag_list)

tag_list.columns = ['index']



artist_list = df_song_meta['artist_name_basket'].explode().value_counts().index

artist_list = pd.DataFrame(artist_list)

artist_list.columns = ['index']

artist_list = artist_list[artist_list['index'].isin(tag_list['index'].to_list())]



df_artist_isin = df_song_meta_explode[df_song_meta_explode['artist_name_basket'].isin(artist_list['index'].to_list())]







df_tag_artist_score = pd.DataFrame()



df_tag_artist_score['artist'] = df_artist_isin['artist_name_basket'].value_counts().index

df_tag_artist_score['isin_tag'] = None

df_tag_artist_score['isin_tag_and_artist'] = None

df_tag_artist_score['score'] = None

df_tag_artist_score['id'] = df_tag_artist_score.index



df_tag_artist_score
def isin_tag(row):

    print(row['id'], '-', (row['id']+1)/len(df_tag_artist_score)*100,"%")

    i = row['artist']

    return len(df_train_explode[df_train_explode['tags'] == i])
def isin_tag_and_artist(row):

    print(row['id'], '-', (row['id']+1)/len(df_tag_artist_score)*100,"%")

    i = row['artist']

    return len(df_train_explode[(df_train_explode['tags'] == i) & (df_train_explode['songs'].isin(df_artist_isin[df_artist_isin['artist_name_basket'] == i]['id'].to_list()))])
# def calc_score(row):

#     print(row['id'], '-', (row['id']+1)/len(df_tag_artist_score)*100,"%")

#     i = row['artist']

#     a = df_train_explode[df_train_explode['tags'] == i]

#     b = df_train_explode[(df_train_explode['tags'] == i) & (df_train_explode['songs'].isin(df_artist_isin[df_artist_isin['artist_name_basket'] == i]['id'].to_list()))]

#     result = len(b)/len(a)

#     return result
def calc_score(row):

    print(row['id'], '-', (row['id']+1)/len(df_tag_artist_score)*100,"%")

    return (row['isin_tag_and_artist']/row['isin_tag'])*100
df_tag_artist_score['isin_tag'] = df_tag_artist_score.apply(isin_tag, axis=1)

df_tag_artist_score
df_tag_artist_score['isin_tag_and_artist'] = df_tag_artist_score.apply(isin_tag_and_artist, axis=1)

df_tag_artist_score
df_tag_artist_score['score'] = df_tag_artist_score.apply(calc_score, axis=1)

df_tag_artist_score
# i = '윤종신'

# df_artist_isin[df_artist_isin['artist_name_basket'] == i]['id']
df_tag_artist_score['artist_songs'] = None



def artist_songs(row):

    print(row['id'], '-', (row['id']+1)/len(df_tag_artist_score)*100,"%")

    i = row['artist']

    return len(df_artist_isin[df_artist_isin['artist_name_basket'] == i]['id'])



df_tag_artist_score['artist_songs'] = df_tag_artist_score.apply(artist_songs, axis=1)

df_tag_artist_score
df_tag_artist_score.columns
df_tag_artist_score2 = df_tag_artist_score[['artist', 'isin_tag_and_artist', 'isin_tag', 'score', 'artist_songs', 'id']]

df_tag_artist_score2
df_tag_artist_score = df_tag_artist_score2

df_tag_artist_score
# import pandas as pd

# df_tag_artist_score = pd.DataFrame()

# df_tag_artist_score['id'] = df_tag_artist_score.index

import distutils.dir_util



distutils.dir_util.mkpath("arena-res/fe/")

df_tag_artist_score.to_csv('arena-res/fe/tag_artist_score.csv', index=False)

pd.read_csv('arena-res/fe/tag_artist_score.csv')
df_tag_artist_score.sort_values('score', ascending=False)
df_tag_artist_score[df_tag_artist_score['score'] == 100.0].sort_values('artist_songs')
df_tag_artist_score[df_tag_artist_score['score']==0.0].sort_values('artist_songs')
# score_list = []



# count = 0



# for i in list(df_artist_isin['artist_name_basket'].value_counts().index[:100]):

#     a = df_train_explode[df_train_explode['tags'] == i]

#     b = df_train_explode[(df_train_explode['tags'] == i) & (df_train_explode['songs'].isin(df_artist_isin[df_artist_isin['artist_name_basket'] == i]['id'].to_list()))]

#     result = len(b)/len(a)

    

#     df_tag_artist_score

#     print(i,"\t",len(b)/len(a))

#     score_list.append(result)

    

# print()

# print(f"mean: {np.mean(score_list)}")
df_tag_artist_score