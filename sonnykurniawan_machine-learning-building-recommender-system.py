#import library yang dibutuhkan

import pandas as pd

import numpy as np



#lakukan pembacaan dataset

movie_df = pd.read_csv('../input/title.basics.tsv', sep='\t')#untuk menyimpan title_basics.tsv



rating_df = pd.read_csv('../input/title.ratings.tsv', sep='\t')#untuk menyimpan title.ratings.tsv

#Menampilkan 5 data teratas movie_df

print("5 data teratas")

print(movie_df.head())

#Melihat tipe data setiap kolom

print("Tipe Data setiap Kolom")

print(movie_df.info())

#Pengecekan Data dengan Nilai NULL

print("Data dengan Nilai NULL")

print(movie_df.isnull().sum())
#Pengecekan terhadap bentuk data dari kolom primaryTitle dan originalTitle yang bernilai NULL

print(movie_df.loc[(movie_df['primaryTitle'].isnull()) | (movie_df['originalTitle'].isnull())])
#Mengupdate movie_df dengan membuang data-data bernilai NULL

movie_df = movie_df.loc[(movie_df['primaryTitle'].notnull()) & (movie_df['originalTitle'].notnull())]



#Menampilkan jumlah data setelah data dengan nilai NULL dibuang

print(len(movie_df))
#Pengecekan terhadap bentuk data dari kolom genres yang bernilai NULL

print(movie_df.loc[movie_df['genres'].isnull()])
#Mengupdate movie_df dengan membuang data-data bernilai NULL

movie_df = movie_df.loc[movie_df['genres'].notnull()]



#Menampilkan jumlah data setelah data dengan nilai NULL dibuang

print(len(movie_df))
#mengubah nilai '\\N' pada startYear menjadi np.nan dan cast kolomnya menjadi float64

movie_df['startYear'] = movie_df['startYear'].replace('\\N', np.nan)

movie_df['startYear'] = movie_df['startYear'].astype('float64')

print(movie_df['startYear'].unique()[:5])



#mengubah nilai '\\N' pada endYear menjadi np.nan dan cast kolomnya menjadi float64

movie_df['endYear'] = movie_df['endYear'].replace('\\N', np.nan)

movie_df['endYear'] = movie_df['endYear'].astype('float64')

print(movie_df['endYear'].unique()[:5])



#mengubah nilai '\\N' pada runtimeMinutes menjadi np.nan dan cast kolomnya menjadi float64

movie_df['runtimeMinutes'] = movie_df['runtimeMinutes'].replace('\\N', np.nan)

movie_df['runtimeMinutes'] = movie_df['runtimeMinutes'].astype('float64')

print(movie_df['runtimeMinutes'].unique()[:5])

#Membuat fungsi transform_to_list

def transform_to_list(x):

    if ',' in x: 

    #ubah menjadi list apabila ada data pada kolom genre

        return x.split(',')

    else: 

    #jika tidak ada data, ubah menjadi list kosong

        return []



movie_df['genres'] = movie_df['genres'].apply(lambda x: transform_to_list(x))
#Menampilkan 5 data teratas rating_df

print("5 data teratas")

print(rating_df.head())

#Melihat tipe data setiap kolom

print("Tipe Data setiap Kolom")

print(rating_df.info())

#Pengecekan Data dengan Nilai NULL

print("Data dengan Nilai NULL")

print(rating_df.isnull().sum())
#Lakukan join pada kedua table

movie_rating_df = pd.merge(movie_df, rating_df, on='tconst', how='inner')



#Tampilkan 5 data teratas

print(movie_rating_df.head())



#Tampilkan tipe data dari tiap kolom

print(movie_rating_df.info())
#Menghilangkan semua nilai NULL dari kolom startYear dan runtimeMinutes

movie_rating_df = movie_rating_df.dropna(subset=['startYear','runtimeMinutes'])



#Untuk memastikan bahwa sudah tidak ada lagi nilai NULL

print(movie_rating_df.info())
C = movie_rating_df['averageRating'].mean()

print(C)
#Mengambil numVotes di atas 80% populasi

m = movie_rating_df['numVotes'].quantile(0.8)

print(m)
#Membuat Fungsi Weighted Rating

def imdb_weighted_rating(df, var=0.8):

    v = df['numVotes']

    R = df['averageRating']

    C = df['averageRating'].mean()

    m = df['numVotes'].quantile(var)

    df['score'] = (v/(m+v))*R + (m/(m+v))*C #Rumus IMDb 

    return df['score']

    

imdb_weighted_rating(movie_rating_df)



#melakukan pengecekan dataframe

print(movie_rating_df.head())
#Membuat fungsi simple_recommender

def simple_recommender(df, top=100):

    df = df.loc[df['numVotes'] >= m] #Filter numVotes yang lebih dari m

    df = df.sort_values(by='score', ascending=False) #urutkan dari nilai tertinggi ke terendah

    

    #Ambil data 100 teratas

    df = df[:top]

    return df

    

#Ambil data 25 teratas     

print(simple_recommender(movie_rating_df, top=25))
df = movie_rating_df.copy()



def user_prefer_recommender(df, ask_adult, ask_start_year, ask_genre, top=100):

    #ask_adult = yes/no

    if ask_adult.lower() == 'yes':

        df = df.loc[df['isAdult'] == 1]

    elif ask_adult.lower() == 'no':

        df = df.loc[df['isAdult'] == 0]



    #ask_start_year = numeric

    df = df.loc[df['startYear'] >= int(ask_start_year)]



    #ask_genre = 'all' atau yang lain

    if ask_genre.lower() == 'all':

        df = df

    else:

        def filter_genre(x):

            if ask_genre.lower() in str(x).lower():

                return True

            else:

                return False

        df = df.loc[df['genres'].apply(lambda x: filter_genre(x))]



    df = df.loc[df['numVotes'] >= m] #Mengambil film dengan m yang lebih besar dibanding numVotes

    df = df.sort_values(by='score', ascending=False)

    

    #jika kamu hanya ingin mengambil 100 teratas

    df = df[:top]

    return df

#Menampilkan rekomendasi movie melakukan filter berdasarkan isAdult, startYear, dan genres.

print(user_prefer_recommender(df,

                              ask_adult = 'no',

                              ask_start_year = 2000,

                              ask_genre = 'drama'

                             ))