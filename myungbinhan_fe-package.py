import pandas as pd

import numpy as np

from sklearn import preprocessing

import distutils.dir_util



from scipy import sparse

from sklearn.metrics.pairwise import cosine_similarity

from tqdm import tqdm

import os

import distutils.dir_util

import io

import json



def fe_package():

    df_train_origin = pd.read_json('../input/arena-res/train.json')

    df_song_meta_origin = pd.read_json('../input/arena-res/song_meta.json')

    df_val_origin = pd.read_json('../input/arena-res/val.json')

    

    df_fe_genre_origin = pd.read_csv('../input/arena-res/fe/fe_genre.csv')

    

    df_fe_tr_id_origin = pd.read_csv('../input/arena-res/fe/fe_tr_id.csv')

    df_fe_tr_va_te_id_origin = pd.read_csv('../input/arena-res/fe/fe_tr_va_te_id.csv')

    

    

    



    # plstvector_percentage

    df_genre = df_fe_genre_origin.copy()

    df_id = df_fe_tr_va_te_id_origin.copy()

    df_val = df_val_origin.copy()[['id', 'songs']]



    def df_to_sparse(df):

        data = np.ones(len(df))

        row_indices = df[df.columns[0]]

        col_indices = df[df.columns[1]]

        return sparse.csr_matrix((data, (row_indices, col_indices)))



    genre_sparse = df_to_sparse(df_genre)



    vec_total = sparse.csr_matrix(np.array([[]]).reshape(0, 30)) # empty sparse



    for idx in tqdm(df_val.index):

        song_list = df_val.iloc[idx]['songs']

        plst_vec_sum = genre_sparse[song_list, :].sum(axis=0)

        if len(song_list) == 0:

            vec_total = sparse.vstack([vec_total, plst_vec_sum])

        else:

            plst_vec_sum = plst_vec_sum / len(song_list)

            plst_vec_sparse = sparse.csr_matrix(plst_vec_sum)

            vec_total = sparse.vstack([vec_total, plst_vec_sparse])





    # genre_gn_all.json 에서 대장르만 가져옴

    with open('../input/arena-res/genre_gn_all.json') as json_file:

        json_data = json.load(json_file)

    genre_list = []

    for k,v in json_data.items():

    #     print(k[-4:])

        if int(k[-4:]) %100 == 0:

            print(k,v)

            genre_list.append(k)



    df_plst_genre = pd.DataFrame(vec_total.A, columns=genre_list)



    distutils.dir_util.mkpath("arena-res/plst_percentage/")

    df_plst_genre.to_csv('arena-res/plst_percentage/genre.csv', index=False)

    

    print('plstvector_percentage')

    print(df_plst_genre)

    

    

    # 파일 불러오기



    

    

    

#     df_genre_dtl = pd.read_json('../input/arena-res/song_meta.json')[['id', 'song_gn_dtl_gnr_basket']].rename(columns={'id':'songs'}).explode('song_gn_dtl_gnr_basket')

#     df_id = pd.read_csv('../input/arena-res/fe/fe_tr_id.csv')

#     df_id = df_id[['songs']].drop_duplicates()

#     df_song_meta = pd.read_json('../input/arena-res/song_meta.json')[['id', 'album_id']]



#     df_artist_id_basket = pd.read_json('../input/arena-res/song_meta.json')[['id', 'artist_id_basket']]



#     df_genre = pd.read_json('../input/arena-res/song_meta.json')[['id', 'song_gn_gnr_basket']].rename(columns={'id':'songs'}).explode('song_gn_gnr_basket')



    le = preprocessing.LabelEncoder()

    distutils.dir_util.mkpath("arena-res/fe/")

    

    

    #feature_genre_dtl

    

    df_genre_dtl = df_song_meta_origin.copy()[['id', 'song_gn_dtl_gnr_basket']].rename(columns={'id':'songs'}).explode('song_gn_dtl_gnr_basket')

    df_genre_dtl[df_genre_dtl['song_gn_dtl_gnr_basket'].isna()]

    df_genre_dtl=df_genre_dtl.dropna() # 값이 없으면 줄을 지워야함

    df_genre_dtl['genre_dtl'] = le.fit_transform(df_genre_dtl['song_gn_dtl_gnr_basket'])

    df_genre_dtl[df_genre_dtl.song_gn_dtl_gnr_basket == 'GN1802']

    df_genre_dtl[['songs', 'genre_dtl']].to_csv('arena-res/fe/fe_genre_dtl.csv', index=False)

    print('feature_genre_dtl')

    print(df_genre_dtl)

    

    

    # feature_id

    

    df_tr_id = df_train_origin.copy()[['songs', 'id']].explode('songs')

    df_tr_id['songs'].isna().sum()



    df_tr_id[['songs', 'id']].to_csv('arena-res/fe/fe_tr_id.csv', index=False)

    print('feature_genre_dtl')

    print(df_tr_id)  

    

    

    #feature_album

    

    df_id = df_fe_tr_id_origin.copy()

    df_id = df_id[['songs']].drop_duplicates()

    

    df_song_meta = df_song_meta_origin.copy()[['id', 'album_id']]



    df_album = pd.merge(df_id, df_song_meta, left_on='songs', right_on='id').drop(['id'],axis=1)



    df_album['album'] = le.fit_transform(df_album['album_id'])

    df_album[['songs', 'album']].to_csv('arena-res/fe/fe_album.csv', index=False)

    

    print('feature_genre_dtl')

    print(df_album)

    

    

    #fe_artist



    df_artist_id_basket = df_song_meta_origin.copy()[['id', 'artist_id_basket']]

    df_artist_id_basket.explode('artist_id_basket').id.value_counts()



    df_artist_id_basket_ex = df_artist_id_basket.explode('artist_id_basket')

    

    df_song_artist = pd.merge(df_id, df_artist_id_basket_ex, left_on='songs', right_on='id').drop(['id'],axis=1)

    df_song_artist[df_song_artist['songs'] == 143370]



    df_song_artist['artist'] = le.fit_transform(df_song_artist['artist_id_basket'])

    df_song_artist[['songs', 'artist']].to_csv('arena-res/fe/fe_artist.csv', index=False)



    print('fe_artist')

    print(df_song_artist)

    

    

    #feature_genre



    df_genre = df_song_meta_origin.copy()[['id', 'song_gn_gnr_basket']].rename(columns={'id':'songs'}).explode('song_gn_gnr_basket')

    df_genre[df_genre['song_gn_gnr_basket'].isna()]

    df_genre=df_genre.dropna() # 값이 없으면 줄을 지워야함



    df_genre['genre'] = le.fit_transform(df_genre['song_gn_gnr_basket'])



    df_genre[df_genre.song_gn_gnr_basket == 'GN1800']

    df_genre[['songs', 'genre']].to_csv('arena-res/fe/fe_genre.csv', index=False)



    print('feature_genre')

    print(df_genre)

    

    #pd.read_csv('arena-res/fe/fe_genre.csv')

    

fe_package()

print('a')
# df_test_test = pd.read_json('../input/arena-res/song_meta.json')[0:1]

# df_test_test



# a = df_test_test.copy()

# b = df_test_test.copy()
# a['id'] = 10    



# print(df_test_test)

# print(a)

# print(b)
