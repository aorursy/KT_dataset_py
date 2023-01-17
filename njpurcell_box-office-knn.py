# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from collections import Counter

from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import KFold

from nltk.corpus import stopwords
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

print(set(train.columns).difference(set(test.columns)))
test["revenue"] = np.nan

all_movies = pd.concat([train, test])

all_movies.head()
all_movies.shape
def clean_belongs_to_collection(x):

    if x is np.nan:

        return ""

    x = x[1:-1]

    return eval(x)['name']



all_movies["collection"] = all_movies["belongs_to_collection"].apply(clean_belongs_to_collection)

print(all_movies["collection"].isnull().sum())

print(all_movies["collection"].head())
def get_genres(x):

    if x is np.nan:

        return ""

    genres = list()

    for genre_dict in eval(x):

        genres.append(genre_dict['name'])

    return ",".join(genres)



all_movies["genres_list"] = all_movies["genres"].apply(get_genres)

print(all_movies["genres_list"].isnull().sum())

all_movies["genres_list"].head()
genres = list()

for i in range(all_movies.shape[0]):

    genres += (all_movies.iloc[i]["genres_list"].split(","))

genres = Counter(genres)

genres.most_common(25)
for genre, count in genres.most_common(25)[:-2]:

    all_movies[genre] = all_movies["genres_list"].apply(lambda x: 1 if genre in x else 0)

all_movies = all_movies.drop(["genres_list"], axis=1)

all_movies.isnull().sum()
all_movies["has_webpage"] = all_movies["homepage"].apply(lambda x: 0 if x is np.nan else 1)

# I'll use imdb_id to scrape critic scores in a later kernel

all_movies = all_movies.drop(["belongs_to_collection", "genres",

                              "homepage", "imdb_id", "poster_path"], axis=1)
dummy_orig_langs = pd.get_dummies(all_movies["original_language"], prefix="original_lang")

all_movies = pd.concat([all_movies,dummy_orig_langs], axis=1)
stop_words = set(stopwords.words('english'))

text_cols = ["overview", "tagline"]

ovw_words = list()

tag_words = list()

for i in range(all_movies.shape[0]):

    try:

        ovw_words += (all_movies.iloc[i]["overview"].replace(",","").replace(".","").lower().split())

        tag_words += (all_movies.iloc[i]["tagline"].replace(",","").replace(".","").lower().split())

    except AttributeError as e:

        continue

ovw_words = Counter([w for w in ovw_words if len(w) > 4 and w not in stop_words])

tag_words = Counter([w for w in tag_words if len(w) > 4 and w not in stop_words])

print(ovw_words.most_common(10))

print(tag_words.most_common(10))
for word, _ in ovw_words.most_common(100):

    col = "overview_"+word

    all_movies[col] = all_movies["overview"].apply(lambda x: 0 if x is np.nan else 1 if word in x.lower() else 0)

for word, _ in tag_words.most_common(100):

    col = "tagline_"+word

    all_movies[col] = all_movies["tagline"].apply(lambda x: 0 if x is np.nan else 1 if word in x.lower() else 0)

all_movies.shape
ovw_high_var_list = list()

tag_high_var_list = list()

train_idx = train.shape[0]

train = all_movies.iloc[:train_idx]

ovw_cols = [col for col in list(train.columns) if "overview_" in col]

tag_cols = [col for col in list(train.columns) if "tagline_" in col]

for col in ovw_cols:

    ovw_high_var_list.append((col, train[train[col] == 1]["revenue"].var()))

for col in tag_cols:

    tag_high_var_list.append((col, train[train[col] == 1]["revenue"].var()))

ovw_high_var_list = sorted(ovw_high_var_list, key=lambda x: x[1], reverse=True)

tag_high_var_list = sorted(tag_high_var_list, key=lambda x: x[1], reverse=True)



# take the top half of variances in revenue

ovw_drop_cols = [x[0] for x in ovw_high_var_list[50:]]

tag_drop_cols = [x[0] for x in tag_high_var_list[50:]]

all_movies = all_movies.drop((ovw_drop_cols + tag_drop_cols), axis=1)

all_movies.shape
all_movies.isnull().sum()
all_movies["status"].value_counts()
all_movies.loc[all_movies['title'].isnull(), 'title'] = all_movies.loc[all_movies['title'].isnull(), 'original_title']

all_movies['status'].fillna("Released", inplace = True)



# fill runtime based on info found at https://www.imdb.com

all_movies.loc[all_movies['title']=='Happy Weekend', 'runtime'] = 81

all_movies.loc[all_movies['title']=='Miesten välisiä keskusteluja', 'runtime'] = 90

all_movies.loc[all_movies['title']=='Nunca en horas de clase', 'runtime'] = 100

all_movies.loc[all_movies['title']=='Pancho, el perro millonario', 'runtime'] = 91

all_movies.loc[all_movies['title']=='La caliente niña Julietta', 'runtime'] = 93

all_movies.loc[all_movies['title']=='Королёв', 'runtime'] = 130



# release date of Jails, Hospitals & Hip-Hop movie : May 2000

all_movies.loc[all_movies['release_date'].isnull(), 'release_date'] = '5/1/00'
#all_movies["release_day"] = all_movies["release_date"].apply(lambda x: int(x.split("/")[1])).astype(int)

all_movies["release_month"] = all_movies["release_date"].apply(lambda x: x.split("/")[0])

all_movies["release_month"].value_counts()
# this function was used above as get_genres

def get_list_of_values(x, key):

    if x is np.nan:

        return ""

    vals = list()

    for val in eval(x):

        vals.append(val[key])

    return ",".join(vals)



def find_most_common(col, n):

    values = list()

    for i in range(all_movies.shape[0]):

        values += all_movies.iloc[i][col].split(",")

    return Counter(values).most_common(n)



def one_hot_encode_most_common(new_col, list_col, cmn_lst):

    for name, cnt in cmn_lst:

        all_movies[new_col+"_"+name] = all_movies[list_col].apply(

            lambda x: 1 if name in x else 0)

    return None



# production companies

all_movies["companies_list"] = all_movies["production_companies"].apply(

    get_list_of_values, args=('name',))

most_cmn_comps = find_most_common("companies_list", 10)

one_hot_encode_most_common("production_companies", "companies_list", most_cmn_comps)



# production countries

all_movies["countries_list"] = all_movies["production_countries"].apply(

    get_list_of_values, args=('iso_3166_1',))

most_cmn_countries = find_most_common("countries_list", 25)

one_hot_encode_most_common("production_countries", "countries_list", most_cmn_countries)



# spoken languages

all_movies["spoken_lang_list"] = all_movies["spoken_languages"].apply(

    get_list_of_values, args=('iso_639_1',))

most_cmn_langs = find_most_common("spoken_lang_list", 25)

one_hot_encode_most_common("spoken_languages", "spoken_lang_list", most_cmn_langs)



# Keywords

all_movies["keywords_list"] = all_movies["Keywords"].apply(

    get_list_of_values, args=('name',))

most_cmn_kywds = find_most_common("keywords_list", 25)

one_hot_encode_most_common("Keywords", "keywords_list", most_cmn_kywds)



# cast

all_movies.loc[all_movies['cast'].isnull(), 'cast'] = "[{'gender':'','gender':'','gender':''}]"

all_movies['cast_gender_0'] = all_movies['cast'].apply(lambda x: np.nan if len(eval(x)) < 1 else eval(x)[0]['gender'])

all_movies['cast_gender_1'] = all_movies['cast'].apply(lambda x: np.nan if len(eval(x)) < 2 else eval(x)[1]['gender'])

all_movies['cast_gender_2'] = all_movies['cast'].apply(lambda x: np.nan if len(eval(x)) < 3 else eval(x)[2]['gender'])



all_movies.shape
all_movies = all_movies.drop(["release_date", "production_companies", "production_countries",

                             "spoken_languages", "Keywords", "cast", "crew", "overview", "tagline",

                             "companies_list", "countries_list", "spoken_lang_list","keywords_list"], axis=1)
all_movies.isnull().sum()
all_movies["cast_gender_0"].value_counts()
# there are more than two genders

dummy_genders_0 = pd.get_dummies(all_movies["cast_gender_0"], prefix="first_cast_gender")

all_movies = pd.concat([all_movies, dummy_genders_0], axis=1)

dummy_genders_1 = pd.get_dummies(all_movies["cast_gender_1"], prefix="scnd_cast_gender_1")

all_movies = pd.concat([all_movies, dummy_genders_1], axis=1)

dummy_genders_2 = pd.get_dummies(all_movies["cast_gender_2"], prefix="thrd_cast_gender_2")

all_movies = pd.concat([all_movies, dummy_genders_2], axis=1)

all_movies = all_movies.drop(["cast_gender_0", "cast_gender_1", "cast_gender_2"], axis=1)
all_movies[[col for col in all_movies.columns.tolist() if col != "revenue"]].isnull().sum().sum()
dummy_months = pd.get_dummies(all_movies["release_month"], prefix="month")

all_movies = pd.concat([all_movies, dummy_months], axis=1)
all_movies = all_movies.drop(['original_language','original_title','status','title',

                             'collection'], axis=1)
all_movies.dtypes
num_movies = all_movies.select_dtypes(include=['float64'])

num_movies = pd.concat([num_movies, all_movies[["budget"]]], axis=1)

num_movies.describe()
def normalize_col(df, col):

    df[col+"_norm"] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    return None



normalize_col(all_movies, "popularity")

normalize_col(all_movies, "runtime")

normalize_col(all_movies, "budget")
all_movies[["popularity_norm", "runtime_norm", "budget_norm"]].describe()
train = all_movies.iloc[:train_idx]

test = all_movies.iloc[train_idx:]
all_movies.columns.tolist()
month_pivot = train.pivot_table(index="release_month", values="revenue", aggfunc=np.mean)

month_pivot.plot.bar()
all_movies["release_month"].dtype
all_movies["summer"] = all_movies["release_month"].apply(lambda x: 1 if x in ['5','6','7'] else 0)

all_movies["winter"] = all_movies["release_month"].apply(lambda x: 1 if x in ['11', '12'] else 0)
train_idx = train.shape[0]

train = all_movies.iloc[:train_idx]

test = all_movies.iloc[train_idx:]
train.columns.tolist()
simple_features = ['popularity_norm', 'runtime_norm', 'budget_norm']

other_features = ['has_webpage']

overview_features = [col for col in train.columns.tolist() if "overview" in col]

tagline_features = [col for col in train.columns.tolist() if "tagline" in col]

company_features = [col for col in train.columns.tolist() if "production_companies" in col]

country_features = [col for col in train.columns.tolist() if "production_countries" in col]

spoken_lang_features = [col for col in train.columns.tolist() if "spoken_languages" in col]

keyword_features = [col for col in train.columns.tolist() if "Keywords_" in col]

cast_gender_features = [col for col in train.columns.tolist() if "cast_gender_" in col]

month_features = [col for col in train.columns.tolist() if "month_" in col]

season_features = ['summer', 'winter']

june = ['month_6']

all_features =  [col for col in train.columns.tolist() if col not in ["revenue", "id", "released_month",

                                                                     "budget", "popularity", "runtime"]]

genre_features = ['Drama',

 'Comedy',

 'Thriller',

 'Action',

 'Romance',

 'Adventure',

 'Crime',

 'Science Fiction',

 'Horror',

 'Family',

 'Fantasy',

 'Mystery',

 'Animation',

 'History',

 'Music',

 'War',

 'Documentary',

 'Western',

 'Foreign']
feature_sets = [simple_features,other_features,overview_features,tagline_features,

                company_features,country_features,spoken_lang_features,keyword_features,

                cast_gender_features,month_features,season_features,june,

                all_features,genre_features]

feature_set_strings = ["simple_features","other_features","overview_features","tagline_features",

                "company_features","country_features","spoken_lang_features","keyword_features",

                "cast_gender_features","month_features", "season_features", "june",

                       "all_features","genre_features"]



y = train["revenue"]

k_s = [3,5,7,9,11,13,15, 17, 19]

model_results = list()

for feature_set, set_string in zip(feature_sets, feature_set_strings):

    print(set_string)

    features = list(set((feature_set + simple_features)))

    X = train[features]

    kf = KFold(n_splits=5, random_state=319, shuffle=True)

    feature_set_errors = list()

    for k in k_s:

        i = 0

        print(str(k)+" neighbors")

        for train_idx, test_idx in kf.split(X):

            i +=1

            model = KNeighborsRegressor(n_neighbors=k)

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]

            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            error = np.sqrt(mean_squared_log_error(y_test, predictions))

            feature_set_errors.append(error)

            print("Fold "+str(i) + ": "+str(round(error,4)))

        print(set_string + " "+ str(k)+" neigbors mean: " + str(round(np.mean(feature_set_errors),4)))

        model_results.append([(set_string+"_"+str(k)), round(np.mean(feature_set_errors),4)])
model_results = sorted(model_results, key=lambda x: x[1])

model_results[:30]
all_features = simple_features + other_features + company_features + country_features + june

my_hunch = simple_features + company_features + june

feature_sets = [simple_features,other_features,company_features,country_features,

                june, all_features, my_hunch]

feature_set_strings = ['simple_features','other_features','company_features','country_features',

                       'june', 'all_features', 'my_hunch']



y = train["revenue"]

k_s = [k for k in range(1,8)]

model_results = list()

for feature_set, set_string in zip(feature_sets, feature_set_strings):

    print(set_string)

    features = list(set((feature_set + simple_features)))

    X = train[features]

    kf = KFold(n_splits=5, random_state=319, shuffle=True)

    feature_set_errors = list()

    for k in k_s:

        i = 0

        print(str(k)+" neighbors")

        for train_idx, test_idx in kf.split(X):

            i +=1

            model = KNeighborsRegressor(n_neighbors=k)

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]

            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            error = np.sqrt(mean_squared_log_error(y_test, predictions))

            feature_set_errors.append(error)

            print("Fold "+str(i) + ": "+str(round(error,4)))

        print(set_string + " "+ str(k)+" neigbors mean: " + str(round(np.mean(feature_set_errors),4)))

        model_results.append([(set_string+"_"+str(k)), round(np.mean(feature_set_errors),4)])
model_results = sorted(model_results, key=lambda x: x[1])

model_results[:25]
all_features = simple_features + other_features + company_features + country_features + june

my_hunch = simple_features + company_features + june

feature_sets = [simple_features,other_features,company_features,country_features,

                june, all_features, my_hunch]

feature_set_strings = ['simple_features','other_features','company_features','country_features',

                       'june', 'all_features', 'my_hunch']



y = train["revenue"]

k_s = [k for k in range(1,20)]

model_results = list()

for feature_set, set_string in zip(feature_sets, feature_set_strings):

    features = list(set((feature_set + simple_features)))

    X = train[features]

    kf = KFold(n_splits=5, random_state=319, shuffle=True)

    feature_set_errors = list()

    for k in k_s:

        for train_idx, test_idx in kf.split(X):

            model = KNeighborsRegressor(n_neighbors=k)

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]

            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            error = np.sqrt(mean_squared_log_error(y_test, predictions))

            feature_set_errors.append(error)

        model_results.append([(set_string+"_"+str(k)), round(np.mean(feature_set_errors),4)])

model_results = sorted(model_results, key=lambda x: x[1])

model_results[:25]
my_hunch = simple_features + company_features + june

feature_sets = [company_features, my_hunch]

feature_set_strings = ['company_features','my_hunch']



y = train["revenue"]

k_s = [k for k in range(1,30)]

model_results = list()

for feature_set, set_string in zip(feature_sets, feature_set_strings):

    features = list(set((feature_set + simple_features)))

    X = train[features]

    kf = KFold(n_splits=5, random_state=319, shuffle=True)

    feature_set_errors = list()

    for k in k_s:

        for train_idx, test_idx in kf.split(X):

            model = KNeighborsRegressor(n_neighbors=k)

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]

            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            error = np.sqrt(mean_squared_log_error(y_test, predictions))

            feature_set_errors.append(error)

        model_results.append([(set_string+"_"+str(k)), round(np.mean(feature_set_errors),4)])

model_results = sorted(model_results, key=lambda x: x[1])

model_results[:25]
features = (company_features + simple_features)

knn = KNeighborsRegressor(n_neighbors=17)

knn.fit(train[features], train['revenue'])

predictions17 = knn.predict(test[features])
submission_df = {"id": test['id'], "revenue": predictions17}

submission17 = pd.DataFrame(submission_df)

submission17.to_csv("knn_submission_17.csv", index=False)
features = (company_features + simple_features)

knn = KNeighborsRegressor(n_neighbors=3)

knn.fit(train[features], train['revenue'])

predictions3 = knn.predict(test[features])
submission_df = {"id": test['id'], "revenue": predictions3}

submission3 = pd.DataFrame(submission_df)

submission3.to_csv("knn_submission_3.csv", index=False)