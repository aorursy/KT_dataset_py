import pandas as pd
df1 = pd.DataFrame({
    'name': ['John Smith', 'Jane Doe', 'Joe Schmo'],
    'address': ['123 Main St.', '456 Maple Ave.', '789 Broadway'],
    'age': [34, 28, 51]
})
print(df1)
df2 = pd.DataFrame([
    ['John Smith', '123 Main St.', 34],
    ['Jane Doe', '456 Maple Ave.', 28],
    ['Joe Schmo', '789 Broadway', 51]
    ],
    columns=['name', 'address', 'age'])
print(df2)
# save data to a CSV
df1.to_csv('new-csv-file.csv')

# load CSV file into a DataFrame in Pandas
df3 = pd.read_csv('../input/sample-csv-file/sample.csv')

print(df3)
df4 = pd.read_csv('../input/imdb-data/IMDB-Movie-Data.csv')

# print first 3 rows of DataFrame (Default 5)
print(df4.head(3))

# print statistics for each columns
print(df4.info())
# Select column 'Title'
imdb_title = df4.Title
print(imdb_title.head())
# Select column 'Runtime (Minutes)'
imdb_runtime_minutes = df4['Runtime (Minutes)']
print(imdb_runtime_minutes.head())
imdb_data = df4[['Title', 'Runtime (Minutes)']]
print(imdb_data.head())
# select fourth row
sing_movie = imdb_data.iloc[3]
print(sing_movie)
# select last third row
last_three_movies = imdb_data.iloc[-3:]
print(last_three_movies)
# select rows with runtime less than 75
short_movies = imdb_data[imdb_data['Runtime (Minutes)'] < 75]
print(short_movies)
# select rows with runtime between 60 and 80
medium_length_movies = imdb_data[(imdb_data['Runtime (Minutes)'] > 60) &
                                 (imdb_data['Runtime (Minutes)'] < 80)]
print(medium_length_movies)
# select rows with title in the list
fav_movies = imdb_data[imdb_data.Title.isin([
    'Wolves at the Door', 'Guardians of the Galaxy'
])]
print(fav_movies)
# reset indices without changing the source DF
fav_movies = fav_movies.reset_index(drop=True)
print(fav_movies)

# reset indices in the source DF
medium_length_movies.reset_index(drop=True, inplace=True)
print(medium_length_movies)