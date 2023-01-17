import pandas as pd
df = pd.read_csv('../input/music_dataset.csv', index_col=0)
df.head(10)
df.info()
df.columns
df.set_axis(['user_id', 'track_name', 'artist_name', 'genre_name', 'city', 'time', 'weekday'], axis = 'columns', inplace = True)
df.columns
df.isnull().sum()
df['track_name'] = df['track_name'].fillna('unknown')
df['artist_name'] = df['artist_name'].fillna('unknown')
df.isnull().sum()
df.dropna(subset = ['genre_name'], inplace = True)
df.isnull().sum()
df.duplicated().sum()
df = df.drop_duplicates().reset_index(drop = True)
df.duplicated().sum()
genres_list = df['genre_name'].unique()
def find_genre(genre_name):

    i = 0

    for element in genres_list:

        if element == genre_name:

            i += 1

    return i
find_genre('hip')
find_genre('hop')
find_genre('hip-hop')
def find_hip_hop(df, wrong):

    df['genre_name'] = df['genre_name'].replace(wrong, 'hiphop')

    number_of_errors = df[df['genre_name'] == wrong]['genre_name'].count()

    return number_of_errors
find_hip_hop(df, 'hip')
df.info()
df.groupby('city')['genre_name'].count()
df.groupby('weekday')['genre_name'].count()
def number_tracks(df, day, city):

    track_list = df[(df['weekday'] == day) & (df['city'] == city)]

    track_list_count = track_list['genre_name'].count()

    return track_list_count
number_tracks(df, 'Monday', 'Moscow')
number_tracks(df, 'Monday', 'Saint-Petersburg')
number_tracks(df, 'Wednesday', 'Moscow')
number_tracks(df, 'Wednesday', 'Saint-Petersburg')
number_tracks(df, 'Friday', 'Moscow')
number_tracks(df, 'Friday', 'Saint-Petersburg')
header = ['city', 'monday', 'wednesday', 'friday']

data_by_days = [

    ['Moscow', 15347, 10865, 15680],

    ['Saint-Petersburg', 5519, 6913, 5802],

]

table = pd.DataFrame(data = data_by_days, columns = header)

print(table)
moscow_general = df[df['city'] == 'Moscow']

print(moscow_general)
spb_general = df[df['city'] == 'Saint-Petersburg']
# объявление функции genre_weekday() с параметрами df, day, time1, time2

# в переменной genre_list сохраняются те строки df, для которых одновременно:

# 1) значение в столбце 'weekday' равно параметру day,

# 2) значение в столбце 'time' больше time1 и

# 3) меньше time2.

# в переменной genre_list_sorted сохраняются в порядке убывания  

# первые 10 значений Series, полученной подсчётом числа значений 'genre_name'

# сгруппированной по столбцу 'genre_name' таблицы genre_list

# функция возвращает значение genre_list_sorted



def genre_weekday(df, day, time1, time2):

    genre_list = df[(df['weekday'] == day) & (df['time'] > time1) & (df['time'] < time2)]

    genre_list_sorted = genre_list.groupby('genre_name')['genre_name'].count().head(10)

    return genre_list_sorted
genre_weekday(moscow_general, 'Monday', '07:00:00', '11:00:00')
genre_weekday(spb_general, 'Monday', '07:00:00', '11:00:00')
genre_weekday(moscow_general, 'Friday', '17:00:00', '23:00:00')
genre_weekday(spb_general, 'Friday', '17:00:00', '23:00:00')
moscow_genres = moscow_general.groupby('genre_name')['genre_name'].count().sort_values(ascending = False)
moscow_genres.head(10)
spb_genres = spb_general.groupby('genre_name')['genre_name'].count().sort_values(ascending = False)
spb_genres.head(10)