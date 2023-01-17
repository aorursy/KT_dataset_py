# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/tmdb-box-office-prediction/train.csv', index_col="id")

data
data.info()
data_original = data.copy(deep=True)
data["from_collection"] = data["belongs_to_collection"].notna().astype('int') 

data["from_collection"]
data.groupby("from_collection")["revenue"].mean()
data["has_homepage"] = data["homepage"].notna().astype('int')

data["has_homepage"]
data.groupby("has_homepage")["revenue"].mean()
data.describe()
data.loc[data["popularity"] > 100]

#now I know there's no popularity anomalies, it is just reeealy popular films 
data[data["runtime"] < 60]

#strange runtimes 
data.loc[data["runtime"] == 0, "runtime"] = np.nan  #will handle it later along with other NaNs
data.info()
data["is_short_title"] = (data["title"].apply(len) < 15)#.astype('int')

data["is_short_title"].value_counts()
data.groupby("is_short_title")["revenue"].mean()

#i guess it's useless
data.drop("is_short_title", axis=1, inplace=True)
data.drop('belongs_to_collection', axis=1, inplace=True)

data.drop('homepage', axis=1, inplace=True)
data["Keywords"][3]
data["poster_path"]
data["cast"]
data["cast"][1]
type(data["cast"][1])
data["cast"].fillna("[]", inplace=True) 
data["cast"] = data["cast"].apply(eval) #will parse string data
type(data["cast"][1])
#check for other unparsed lists

for x in data.columns[data.dtypes == "object"]:

    print(x)

    print(data[x].dropna().reset_index().iloc[1])

    print(type(data[x].dropna().reset_index().iloc[1].values[1]))

    print()
#parse that string data (except cast because it's already parsed)

list_features = ["cast", "genres", "production_companies", "production_countries", 

                 "spoken_languages", "Keywords", "crew"]

for x in list_features[1:]:

    data[x].fillna("[]", inplace=True) 

    data[x] = data[x].apply(eval) #will parse string data
data.info()
#now we need to extract useful information from those lists 

all_actors = []

for movie_cast in data["cast"]:

    for member in movie_cast:

        all_actors.append(member["id"])

all_actors = pd.Series(all_actors)

popular_actors = all_actors.value_counts()[:250].index.values

popular_actors
all_directors = [] 

for movie_crew in data["crew"]:

    for member in movie_crew:

        if member["job"] == "Director":

            all_directors.append(member["id"])

all_directors = pd.Series(all_directors)

popular_directors = all_directors.value_counts()[:250].index.values

popular_directors
all_prod_companies = []

for movie in data["production_companies"]:

    for company in movie:

        all_prod_companies.append(company["id"])

all_prod_companies = pd.Series(all_prod_companies)

popular_companies = all_prod_companies.value_counts()[:10].index.values

popular_companies
all_prod_countries = []

for movie in data["production_countries"]:

    for country in movie:

        all_prod_countries.append(country["iso_3166_1"])

all_prod_countries = pd.Series(all_prod_countries)

popular_countries = all_prod_countries.value_counts()[:5].index.values

popular_countries
all_genres = []

for genre_list in data["genres"]: 

    for genre in genre_list:

        all_genres.append(genre["name"])

all_genres = pd.Series(all_genres)

popular_genres = np.append(all_genres.value_counts()[:10].index.values, "Animation")

popular_genres
def count_popular_actors(movie_cast):

    number_of_popular_actors = 0 

    for member in movie_cast:

        if member["id"] in popular_actors: number_of_popular_actors += 1 

    return number_of_popular_actors if number_of_popular_actors < 8 else 8 #movies with number of popular actors > 8 are strange in this dataset

def check_has_popular_director(movie_crew):

    for member in movie_crew: 

        if member["job"] == "Director" and member["id"] in popular_directors: return True     

    return False #if we iterated through all members and haven't found any popular directors

def check_has_popular_company(movie_companies):

    for company in movie_companies: 

        if company["id"] in popular_companies: return True

    return False

def check_has_popular_country(movie_countries):

    for country in movie_countries: 

        if country["iso_3166_1"] in popular_countries: return True   

    return False

def check_genre(movie_genres, target_genre): 

    for genre in movie_genres:

        if target_genre == genre["name"]: return True

    return False



data["has_popular_director"] = data["crew"].apply(check_has_popular_director)

data["number_of_popular_actors"] = data["cast"].apply(count_popular_actors)

data["from_popular_company"] = data["production_companies"].apply(check_has_popular_company)

data["from_popular_country"] = data["production_countries"].apply(check_has_popular_country)

for genre in popular_genres: data["Is"+"".join(genre.split(" "))] = data["genres"].apply(check_genre, target_genre=genre)

print(data.groupby("has_popular_director")["revenue"].mean())

print(data.groupby("number_of_popular_actors")["revenue"].mean())

print(data.groupby("from_popular_company")["revenue"].mean())

print(data.groupby("from_popular_country")["revenue"].mean())

for genre in popular_genres: print(data.groupby("Is"+"".join(genre.split(" ")))["revenue"].mean())
data.info()
data["budget"].describe().astype('int')
data.loc[data["budget"] == data["budget"].max()] 
data["budget"].describe()
data.loc[data["budget"] == 0, "budget"] = data["budget"].mean()
data["budget"].describe()
data.info()
data.drop(["cast", "crew", "title", "tagline", "spoken_languages", "production_companies", "Keywords", "poster_path", "status",

           "production_countries", "overview", "original_title", "original_language", "imdb_id", "genres"], axis=1, inplace=True)
data["runtime"] = data["runtime"].fillna(data["runtime"].mean()).astype("int")
data_original.info()
data.info()
data_original.iloc[2]
data.iloc[2]
(data["release_date"][3])
def get_year(release_date):

    century = "20" if release_date[-2] in ["0", "1"] else "19"

    return int(century+release_date[-2:])

def get_season(release_date):

    int_month = int(release_date.split("/")[0])

    if int_month < 3: return "Winter"

    if int_month < 6: return "Spring"

    if int_month < 9: return "Summer"

    if int_month < 12: return "Autumn"

    return "Christmas" #yeah, I know, very logical..

data["release_year"] = data["release_date"].apply(get_year)

data["release_season"] = data["release_date"].apply(get_season)
data["release_season"].value_counts()
data.groupby("release_season")["revenue"].mean().sort_values()
data["ReleaseWinter"] = data["release_season"] == "Winter"

data["ReleaseSpring"] = data["release_season"] == "Spring" 

data["ReleaseSummer"] = data["release_season"] == "Summer" 

data["ReleaseAutumn" ] = data["release_season"] == "Autumn" 

data["ReleaseChristmas"] = data["release_season"] == "Christmas" 
data.drop("release_season", axis=1, inplace=True) 
data.info()
import matplotlib.pyplot as plt 

plt.scatter(data["release_year"], data["revenue"])
data["yearBand"] = pd.cut(data["release_year"], 5)

data.groupby("yearBand")["revenue"].mean().sort_values()
data.loc[ data["release_year"] <= 1940, "release_year"] = 1 

data.loc[(data["release_year"] <= 1960) & (data["release_year"] > 1940), "release_year"] = 2 

data.loc[(data["release_year"] <= 1980) & (data["release_year"] > 1960), "release_year"] = 3 

data.loc[(data["release_year"] <= 1997) & (data["release_year"] > 1980), "release_year"] = 4 

data.loc[(data["release_year"] > 1997), "release_year"] = 5 
data["release_year"].describe()
data["release_year"].hist()
data.drop("yearBand", axis=1, inplace=True)
data.drop(["release_date"], axis=1, inplace=True)
plt.scatter(data.index, data["revenue"])
plt.scatter(data.index, data["popularity"])
data[data["popularity"] > 50] 
data[data["revenue"] > 1.0e9]
data["revenue"].describe()
data["popularity"].describe()
data_original[data["popularity"] > data["popularity"].quantile(0.75) + 4*data["popularity"].std()]["title"]
data = data.astype('float')
data.info()
sum_season_counts = {} 

for x in ["Winter", "Spring", "Summer", "Autumn", "Christmas"]:

    sum_season_counts[x] = data["Release"+x].value_counts()[1]

sum_season_counts
print(list_features)

print(popular_actors)

print(popular_directors)

print(popular_companies)

print(popular_countries)

print(popular_genres)

budget_mean = data["budget"].mean()

print(budget_mean)

runtime_mean = data["runtime"].mean()

print(runtime_mean)



def prepare_data(data_unprep):

    data = data_unprep.copy()

    

    data["from_collection"] = data["belongs_to_collection"].notna()

    data["has_homepage"] = data["homepage"].notna()

    

    data.loc[data["runtime"] == 0, "runtime"] = np.nan

    

    for x in list_features:

        data[x].fillna("[]", inplace=True)

        data[x] = data[x].apply(eval) #will parse string data



    data["has_popular_director"] = data["crew"].apply(check_has_popular_director)

    data["number_of_popular_actors"] = data["cast"].apply(count_popular_actors)

    data["from_popular_company"] = data["production_companies"].apply(check_has_popular_company)

    data["from_popular_country"] = data["production_countries"].apply(check_has_popular_country)

    for genre in popular_genres: data["Is"+"".join(genre.split(" "))] = data["genres"].apply(check_genre, target_genre=genre)

    

    data.loc[data["budget"] == 0, "budget"] = budget_mean

    data["runtime"] = data["runtime"].fillna(runtime_mean).astype("int")

    

    data["release_date"].fillna("08/02/2006", inplace=True)

    data["release_year"] = data["release_date"].apply(get_year)

    data.loc[ data["release_year"] <= 1940, "release_year"] = 1 

    data.loc[(data["release_year"] <= 1960) & (data["release_year"] > 1940), "release_year"] = 2 

    data.loc[(data["release_year"] <= 1980) & (data["release_year"] > 1960), "release_year"] = 3 

    data.loc[(data["release_year"] <= 1997) & (data["release_year"] > 1980), "release_year"] = 4 

    data.loc[(data["release_year"] > 1997), "release_year"] = 5 

    

    data["release_season"]   = data["release_date"].apply(get_season)

    data["ReleaseWinter"]    = data["release_season"] == "Winter"

    data["ReleaseSpring"]    = data["release_season"] == "Spring" 

    data["ReleaseSummer"]    = data["release_season"] == "Summer" 

    data["ReleaseAutumn"]    = data["release_season"] == "Autumn" 

    data["ReleaseChristmas"] = data["release_season"] == "Christmas" 

    

    to_drop = ["release_season", "belongs_to_collection", "cast", "crew", "title", "tagline", "spoken_languages", 

               "production_companies", "Keywords", "poster_path", "status", "homepage", "release_date",

               "production_countries", "overview", "original_title", "original_language", "imdb_id", "genres"]

    

    data.drop(to_drop, axis=1, inplace=True)

    return data
data_prepared = prepare_data(data_original)

data_prepared.columns
data_prepared.info()
corr = data_prepared.corr()

corr.T["revenue"]
X_train = data_prepared.drop("revenue", axis=1) 

y_train = data_prepared["revenue"]
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor()

forest.fit(X_train, y_train) 

forest.score(X_train, y_train)
X_test = pd.read_csv('/kaggle/input/tmdb-box-office-prediction/test.csv', index_col="id")

X_test
X_test["release_date"].astype('str')
X_test.info()
X_test_prepared = prepare_data(X_test)
X_test_prepared
predictions = forest.predict(X_test_prepared)

predictions
submission = pd.DataFrame({

    "id": X_test_prepared.index, 

    "revenue": predictions

})

submission
submission.to_csv("movie_submission.csv", index=False)