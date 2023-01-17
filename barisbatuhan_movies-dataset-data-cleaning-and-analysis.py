import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from ast import literal_eval

import json

%matplotlib inline
# reads the csv metadata and prints the head

df = pd.read_csv("../input/the-movies-dataset/movies_metadata.csv", low_memory=False)
df.head(5)
drop_df = ["homepage", "poster_path", "video", "imdb_id", "overview", "original_title", 

           "spoken_languages", "tagline"]

df = df.drop(drop_df, axis=1) # drops the selected columns

df = df.drop_duplicates(keep='first') # removes the duplicates from existing dataframe

df.dropna(how="all",inplace=True) # if each column is NaN or null in a row, drops this row
df.shape

df.info()
df.dropna(subset=["title"], inplace=True)

df["id"] =pd.to_numeric(df['id'], errors='coerce', downcast="integer")

df["popularity"] =pd.to_numeric(df['popularity'], errors='coerce', downcast="float") 

df["budget"] =pd.to_numeric(df['budget'], errors='coerce', downcast="float") 

df['release_date'] = pd.to_datetime(df['release_date'])

df['release_year'] = df['release_date'].dt.year
df['belongs_to_collection'] = df['belongs_to_collection'].fillna("None")

df['belongs_to_collection'] = (df['belongs_to_collection'] != "None").astype(int)
df["adult"].value_counts()
df.drop(["adult"], inplace=True, axis=1)
df.info()
df["status"].fillna(df["status"].value_counts().idxmax(), inplace=True)

df["runtime"] = df["runtime"].replace(0, np.nan)

df["runtime"].fillna(df["runtime"].mean(), inplace=True) 
df.dropna(subset=["release_date"],inplace=True)

df.dropna(subset=["original_language"],inplace=True)
# converts json list to list of inputs (from the label specified with 'wanted' parameter)

def json_to_arr(cell, wanted = "name"): 

    cell = literal_eval(cell)

    if cell == [] or (isinstance(cell, float) and cell.isna()):

        return np.nan

    result = []

    counter = 0

    for element in cell:

        if counter < 3:

            result.append(element[wanted])

            counter += 1

        else:

            break

    return result[:3]
df[['genres']] = df[['genres']].applymap(json_to_arr)

df[['production_countries']] = df[['production_countries']].applymap(lambda row: 

                                                                     json_to_arr(row, "iso_3166_1"))

df[['production_companies']] = df[['production_companies']].applymap(json_to_arr)
df['budget'] = df['budget'].replace(0 , pd.np.nan)

df['revenue'] = df['revenue'].replace(0 , pd.np.nan)
print("Number of rows with budget < 100: ", len((df[(df["budget"].notna())&(df["budget"] < 100)])))

print("Number of rows with budget > 100 and < 1000: ", len(df[(df["budget"].notna())&(df["budget"] > 100)

                                                              &(df["budget"] < 1000)]))

print("Number of rows with budget > 1000 and < 10000: ", len(df[(df["budget"].notna())&(df["budget"] > 1000)

                                                              &(df["budget"] < 10000)]))
def scale_money(num):

    if num < 100:

        return num * 1000000

    elif num >= 100 and num < 1000:

        return num * 10000

    elif num >= 1000 and num < 10000:

        return num *100

    else:

        return num
df[['budget', 'revenue']] = df[['budget', 'revenue']].applymap(scale_money)
sns.heatmap(df.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')
print("NaN Genres Count: ", len(df[df["genres"].isna()]))

print("NaN Revenue Count: ", len(df[df['revenue'].isna()])) 

print("NaN Budget Count: ", len(df[df['budget'].isna()])) 

print("NaN Production Company Count: ", len(df[df["production_companies"].isna()]))

print("NaN Production Country Count: ", len(df[df["production_countries"].isna()]))
# returns the values and occurance times or "limiter" amount of different parameters in a 2D list

def list_counter(col, limiter = 9999, log = True):

    result = dict()

    for cell in col:

        if isinstance(cell, float):

            continue

        for element in cell:

            if element in result:

                result[element] += 1

            else:

                result[element] = 1

    if log:

        print("Size of words:", len(result))

    result = {k: v for k, v in sorted(result.items(), key=lambda item: item[1], reverse=True)}

    if log:

        print("Sorted result is:")

    counter = 1

    sum_selected = 0

    total_selected = 0

    rest = 0

    returned = []

    for i in result: 

        if counter > limiter:

            total_selected += result[i]

        else:

            counter += 1

            sum_selected += result[i]

            total_selected += result[i]

            if log:

                print(result[i], " - ", i) 

            returned.append([i, result[i]])

    if log:

        print("Covered:", sum_selected, "out of", total_selected, "\n")

    return returned
genres_occur = list_counter(df["genres"].values, log=False)

genres = pd.DataFrame.from_records(genres_occur, columns=["genres", "count"])

genres.plot(kind = 'bar', x="genres")
countries_occur = list_counter(df["production_countries"].values, log=False)

countries = pd.DataFrame.from_records(countries_occur, columns=["countries", "count"])

countries.head(20).plot(kind = 'bar', x="countries")
companies_occur = list_counter(df["production_companies"].values, log=False)

companies = pd.DataFrame.from_records(companies_occur, columns=["companies", "count"])

companies.head(20).plot(kind = 'bar', x="companies")
def fill_na_with_list(cell, data):

    if isinstance(cell, float):

        return data

    else:

        return cell
df[['genres']] = df[['genres']].applymap(lambda row:

                                        fill_na_with_list(row, [genres_occur[0][0]]))

df[['production_countries']] = df[['production_countries']].applymap(lambda row: 

                                        fill_na_with_list(row, [countries_occur[0][0]]))
df.shape

df.info()
df["profit"] = df["revenue"] - df["budget"]

df[["popularity", "revenue", "budget", "runtime", "vote_average","profit", "release_year"]].describe()
min_val = df["budget"].min()

max_val = df["budget"].max()

df[["budget", "revenue", "profit"]] = df[["budget", "revenue", "profit"]].apply(lambda x: 

                                                            x / (max_val - min_val))
vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')

vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')

C = vote_averages.mean()

m = vote_counts.quantile(0.75)

def weighted_rating(data):

    v = data['vote_count'] + 1 # added +1

    R = data['vote_average']

    return (v / (v + m) * R) + (m / (m + v) * C)



df['weighted_rating'] = df.apply(weighted_rating, axis=1)
df_kwrd = pd.read_csv("../input/the-movies-dataset/keywords.csv")

df_kwrd.head()
df_kwrd["keywords"] = df_kwrd[['keywords']].applymap(json_to_arr)
df_kwrd.dropna(inplace=True)
keywords_occur = list_counter(df_kwrd["keywords"].values, log=False)

keywords = pd.DataFrame.from_records(keywords_occur, columns=["keywords", "count"])

keywords.head(20).plot(kind = 'bar', x="keywords")
df = pd.merge(df, df_kwrd, on=['id'], how='left')
df.info()
df_cr = pd.read_csv("../input/the-movies-dataset/credits.csv")

df_cr.head()
df_cr["cast"] = df_cr[['cast']].applymap(json_to_arr)
def get_director(x):

    x = literal_eval(x)

    for i in x:

        if i == "[]" or isinstance(i, float):

            return np.nan

        if i['job'] == 'Director':

            return i['name']

    return np.nan



df_cr['director'] = df_cr['crew'].apply(get_director)

df_cr.drop(["crew"], axis=1, inplace=True)
print("Entries with no cast:", len(df_cr[df_cr["cast"].isna()]))

print("Entries with no directors:", len(df_cr[df_cr["director"].isna()]))

print("Entries missing both:", len(df_cr[(df_cr["cast"].isna())&(df_cr["director"].isna())]))

df_cr.drop(df_cr[(df_cr["cast"].isna())&(df_cr["director"].isna())].index, inplace=True)
df = pd.merge(df, df_cr, on=['id'], how='left')
df.shape

df.info()
df.head(3)
df.sort_values('weighted_rating', ascending=False)[["title", "director", "genres", "profit", 

                                                    "popularity", "weighted_rating"]].head(10)
df.sort_values('popularity', ascending=False)[["title", "director", "genres", "profit", 

                                                    "popularity", "weighted_rating"]].head(10)
df.sort_values('profit', ascending=False)[["title", "director", "genres", "profit", 

                                                    "popularity", "weighted_rating"]].head(10)
sns.heatmap(df.corr(), cmap = 'YlGnBu')

df.drop(["id"], axis=1).corr()
g = sns.scatterplot(x="vote_count", y="profit", data=df[["profit", "vote_count"]])
g = sns.scatterplot(x="budget", y="revenue", data=df[["budget", "revenue"]])
g = sns.scatterplot(x="vote_count", y="popularity", data=df[["popularity", "vote_count"]])
g = sns.scatterplot(x="popularity", y="weighted_rating", data=df[["popularity", "weighted_rating"]])
df_genres = df[["title", "genres", "popularity", "budget", "revenue", "vote_count", "weighted_rating"]]
df_genres.head()
genres = list_counter(df_genres["genres"].values, log=False)
def list_to_col(data, col_name, col_list, limiter = 9999):

    counter = 0

    selected_items = set()

    for item in col_list:

        if counter >= limiter:

            break

        item = item[0]

        data[item] = 0

        selected_items.add(item)

        counter += 1

    

    for index, row in data.iterrows():

        for item in row[col_name]:  

            if item in selected_items:

                data.at[index, item] = 1

    data.drop([col_name], axis=1, inplace=True)

    return data
df_genres = list_to_col(df_genres, "genres", genres)

df_genres
def binary_mean_dataset_generator(data, col_list, limiter = 9999):

    counter = 0

    items = []

    for item in col_list:

        if counter >= limiter:

            break

        items.append(item[0])

        counter += 1

    rows = []

    for item in items:

        value = data[data[item] == 1].mean()

        rows.append([item, value[0], value[1], value[2], value[3], value[4]])  

    

    df_genres_means = pd.DataFrame(rows, columns=["type", "popularity", "budget", "revenue", 

                                            "vote_count", "rating"])

    return df_genres_means
df_means_genres = binary_mean_dataset_generator(df_genres, genres)

df_means_genres
plt.rcdefaults()

fig, ax = plt.subplots()

y_pos = np.arange(len(df_means_genres))

ax.barh(y_pos, df_means_genres['rating'], align='center')

ax.set_yticks(y_pos)

ax.set_yticklabels(df_means_genres['type'])

ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('Rating')

ax.set_title('Average Rating w.r.t. Genres')

plt.show()
plt.rcdefaults()

fig, ax = plt.subplots()

y_pos = np.arange(len(df_means_genres))

ax.barh(y_pos, df_means_genres['popularity'], align='center')

ax.set_yticks(y_pos)

ax.set_yticklabels(df_means_genres['type'])

ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('Popularity')

ax.set_title('Popularity w.r.t. Genres')

plt.show()
plt.rcdefaults()

fig, ax = plt.subplots()

y_pos = np.arange(len(df_means_genres))

ax.barh(y_pos, df_means_genres['vote_count'], align='center')

ax.set_yticks(y_pos)

ax.set_yticklabels(df_means_genres['type'])

ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('Vote Count')

ax.set_title('Vote Count w.r.t. Genres')

plt.show()
sns.set(style="whitegrid")

f, ax = plt.subplots(figsize=(10, 5))



sns.set_color_codes("muted")

sns.barplot(x="revenue", y="type", data=df_means_genres[['type', 'budget', 'revenue']],

            label="Revenue", color="b")



sns.set_color_codes("pastel")

sns.barplot(x="budget", y="type", data=df_means_genres[['type', 'budget', 'revenue']],

            label="Budget", color="b")



# Add a legend and informative axis label

ax.legend(ncol=2, loc="lower right", frameon=True)

ax.set(xlim=(0, 0.5), ylabel="Movie Types",

       xlabel="Average Budget And Revenue w.r.t. Genres")

sns.despine(left=True, bottom=True)
sns.heatmap(df_genres[["popularity", "budget", "revenue", "vote_count", "weighted_rating"]].corr(), 

            cmap = 'YlGnBu')
df_countries = df[["title", "production_countries", "popularity", "budget", "revenue", "vote_count", "weighted_rating"]]

countries = list_counter(df_countries["production_countries"].values, limiter=10, log=False)

df_countries = list_to_col(df_countries, "production_countries", countries, 10)

df_means_ct = binary_mean_dataset_generator(df_countries, countries)

df_means_ct 
plt.rcdefaults()

fig, ax = plt.subplots()

y_pos = np.arange(len(df_means_ct))

ax.barh(y_pos, df_means_ct['rating'], height=0.5, align='center')

ax.set_yticks(y_pos)

ax.set_yticklabels(df_means_ct['type'])

ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('Rating')

ax.set_title('Average Rating w.r.t. Countries')

plt.show()
plt.rcdefaults()

fig, ax = plt.subplots()

y_pos = np.arange(len(df_means_ct))

ax.barh(y_pos, df_means_ct['popularity'], align='center')

ax.set_yticks(y_pos)

ax.set_yticklabels(df_means_ct['type'])

ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('Popularity')

ax.set_title('Popularity w.r.t. Countries')

plt.show()
plt.rcdefaults()

fig, ax = plt.subplots()

y_pos = np.arange(len(df_means_ct))

ax.barh(y_pos, df_means_ct['vote_count'], align='center')

ax.set_yticks(y_pos)

ax.set_yticklabels(df_means_ct['type'])

ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('Vote Count')

ax.set_title('Average Vote Count w.r.t. Countries')

plt.show()
sns.set(style="whitegrid")

f, ax = plt.subplots(figsize=(8, 3))

sns.set_color_codes("pastel")

sns.barplot(x="revenue", y="type", data=df_means_ct[['type', 'budget', 'revenue']],

            label="Revenue", color="b")



sns.set_color_codes("muted")

sns.barplot(x="budget", y="type", data=df_means_ct[['type', 'budget', 'revenue']],

            label="Budget", color="b")



# Add a legend and informative axis label

ax.legend(ncol=2, loc="lower right", frameon=True)

ax.set(xlim=(0, 0.3), ylabel="Movie Types",

       xlabel="Average Budget And Revenue w.r.t. Countries")

sns.despine(left=True, bottom=True)
sns.heatmap(df_countries[["popularity", "budget", "revenue", "vote_count", "weighted_rating"]].corr(), 

            cmap = 'YlGnBu')
df_dir= df[["title", "director", "popularity", "budget", "revenue", "vote_count", "weighted_rating"]]

df_dir.dropna(subset=["director"], inplace=True)

directors = df_dir["director"].value_counts()

directors = directors.index.to_list()
def str_to_col(data, col_name, col_list, limiter = 9999):

    counter = 0

    selected = set()

    for item in col_list:

        if counter >= limiter:

            break

        data[item] = 0

        selected.add(item)

        counter += 1

    for index, row in data.iterrows():

        item = row[col_name]

        if(item in selected):

            data.at[index, item] = 1

    data.drop([col_name], axis=1, inplace=True)

    return data
def str_mean_dataset_generator(data, col_list, limiter = 9999):

    counter = 0

    items = []

    for item in col_list:

        if counter >= limiter:

            break

        items.append(item)

        counter += 1

    rows = []

    for item in items:

        value = data[data[item] == 1].mean()

        rows.append([item, value[0], value[1], value[2], value[3], value[4]])  

    

    df_genres_means = pd.DataFrame(rows, columns=["type", "popularity", "budget", "revenue", 

                                            "vote_count", "rating"])

    return df_genres_means
df_dir = str_to_col(df_dir, "director", directors[:15], 15)

df_means_dir = str_mean_dataset_generator(df_dir, directors[:15])

df_means_dir 
plt.rcdefaults()

fig, ax = plt.subplots()

y_pos = np.arange(len(df_means_dir))

ax.barh(y_pos, df_means_dir['rating'], height=0.5, align='center')

ax.set_yticks(y_pos)

ax.set_yticklabels(df_means_dir['type'])

ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('Rating')

ax.set_title('Average Rating w.r.t. Directors')

plt.show()
plt.rcdefaults()

fig, ax = plt.subplots()

y_pos = np.arange(len(df_means_dir))

ax.barh(y_pos, df_means_dir['popularity'], align='center')

ax.set_yticks(y_pos)

ax.set_yticklabels(df_means_dir['type'])

ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('Popularity')

ax.set_title('Popularity w.r.t. Directors')

plt.show()
plt.rcdefaults()

fig, ax = plt.subplots()

y_pos = np.arange(len(df_means_dir))

ax.barh(y_pos, df_means_dir['vote_count'], align='center')

ax.set_yticks(y_pos)

ax.set_yticklabels(df_means_dir['type'])

ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('Vote Count')

ax.set_title('Vote Count w.r.t. Directors')

plt.show()
sns.set(style="whitegrid")

f, ax = plt.subplots(figsize=(8, 3))

sns.set_color_codes("pastel")

sns.barplot(x="revenue", y="type", data=df_means_dir[['type', 'budget', 'revenue']],

            label="Revenue", color="b")



sns.set_color_codes("muted")

sns.barplot(x="budget", y="type", data=df_means_dir[['type', 'budget', 'revenue']],

            label="Budget", color="b")



# Add a legend and informative axis label

ax.legend(ncol=2, loc="upper right", frameon=True)

ax.set(xlim=(0, 0.3), ylabel="Movie Types",

       xlabel="Average Budget And Revenue w.r.t. Directors")

sns.despine(left=True, bottom=True)
sns.heatmap(df_dir[["popularity", "budget", "revenue", "vote_count", "weighted_rating"]].corr(), 

            cmap = 'YlGnBu')
df_key = df[["title", "keywords", "popularity", "budget", "revenue", "vote_count", "weighted_rating"]]

df_key.dropna(subset=["keywords"], inplace=True)

keywords = list_counter(df_key["keywords"].values, 20, log=False)

df_key = list_to_col(df_key, "keywords", keywords)

df_means_key = binary_mean_dataset_generator(df_key, keywords, 20)

df_means_key
plt.rcdefaults()

fig, ax = plt.subplots()

y_pos = np.arange(len(df_means_key))

ax.barh(y_pos, df_means_key['rating'], align='center')

ax.set_yticks(y_pos)

ax.set_yticklabels(df_means_key['type'])

ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('Rating')

ax.set_title('Average Rating w.r.t. Keywords')

plt.show()
plt.rcdefaults()

fig, ax = plt.subplots()

y_pos = np.arange(len(df_means_key))

ax.barh(y_pos, df_means_key['popularity'], align='center')

ax.set_yticks(y_pos)

ax.set_yticklabels(df_means_key['type'])

ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('Popularity')

ax.set_title('Popularity w.r.t. Keywords')

plt.show()
plt.rcdefaults()

fig, ax = plt.subplots()

y_pos = np.arange(len(df_means_key))

ax.barh(y_pos, df_means_key['vote_count'], align='center')

ax.set_yticks(y_pos)

ax.set_yticklabels(df_means_key['type'])

ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('Vote Count')

ax.set_title('Vote Count w.r.t. Keywords')

plt.show()
sns.set(style="whitegrid")

f, ax = plt.subplots(figsize=(10, 5))



sns.set_color_codes("muted")

sns.barplot(x="revenue", y="type", data=df_means_key[['type', 'budget', 'revenue']],

            label="Revenue", color="b")



sns.set_color_codes("pastel")

sns.barplot(x="budget", y="type", data=df_means_key[['type', 'budget', 'revenue']],

            label="Budget", color="b")



# Add a legend and informative axis label

ax.legend(ncol=2, loc="lower right", frameon=True)

ax.set(xlim=(0, 0.5), ylabel="Movie Types",

       xlabel="Average Budget And Revenue w.r.t. Keywords")

sns.despine(left=True, bottom=True)
sns.heatmap(df_key[["popularity", "budget", "revenue", "vote_count", "weighted_rating"]].corr(), 

            cmap = 'YlGnBu')
df_cast_dir = df[["director", "cast"]].dropna()

df_cast_dir.head()
director_list = df_cast_dir["director"].value_counts()

director_list = director_list.index.to_list()

df_cast_dir = str_to_col(df_cast_dir, "director", director_list[:10], 10)
cast = list_counter(df_cast_dir["cast"].values, 10, log=False)

df_cast_dir = list_to_col(df_cast_dir, "cast", cast, 10)
df_cast_dir = df_cast_dir.loc[(df_cast_dir!=0).any(axis=1)]
df_cast_dir.shape
sns.heatmap(df_cast_dir.corr(), cmap = 'YlGnBu')