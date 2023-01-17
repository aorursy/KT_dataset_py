import numpy as np

import pandas as pd

from IPython.display import display

from dataclasses import dataclass



RAND_SEED = 47

np.random.seed(RAND_SEED)
# http://www.in2013dollars.com/1860-dollars-in-2017?amount=1

INFLATION_CPI_HIST = {

    "1913": 9.9,

    "1914": 10,

    "1915": 10.1,

    "1916": 10.9,

    "1917": 12.8,

    "1918": 15,

    "1919": 17.3,

    "1920": 20,

    "1921": 17.9,

    "1922": 16.8,

    "1923": 17.1,

    "1924": 17.1,

    "1925": 17.5,

    "1926": 17.7,

    "1927": 17.4,

    "1928": 17.2,

    "1929": 17.2,

    "1930": 16.7,

    "1931": 15.2,

    "1932": 13.6,

    "1933": 12.9,

    "1934": 13.4,

    "1935": 13.7,

    "1936": 13.9,

    "1937": 14.4,

    "1938": 14.1,

    "1939": 13.9,

    "1940": 14,

    "1941": 14.7,

    "1942": 16.3,

    "1943": 17.3,

    "1944": 17.6,

    "1945": 18,

    "1946": 19.5,

    "1947": 22.3,

    "1948": 24,

    "1949": 23.8,

    "1950": 24.1,

    "1951": 26,

    "1952": 26.6,

    "1953": 26.8,

    "1954": 26.9,

    "1955": 26.8,

    "1956": 27.2,

    "1957": 28.1,

    "1958": 28.9,

    "1959": 29.2,

    "1960": 29.6,

    "1961": 29.9,

    "1962": 30.3,

    "1963": 30.6,

    "1964": 31,

    "1965": 31.5,

    "1966": 32.5,

    "1967": 33.4,

    "1968": 34.8,

    "1969": 36.7,

    "1970": 38.8,

    "1971": 40.5,

    "1972": 41.8,

    "1973": 44.4,

    "1974": 49.3,

    "1975": 53.8,

    "1976": 56.9,

    "1977": 60.6,

    "1978": 65.2,

    "1979": 72.6,

    "1980": 82.4,

    "1981": 90.9,

    "1982": 96.5,

    "1983": 99.6,

    "1984": 103.9,

    "1985": 107.6,

    "1986": 109.6,

    "1987": 113.6,

    "1988": 118.3,

    "1989": 124,

    "1990": 130.7,

    "1991": 136.2,

    "1992": 140.3,

    "1993": 144.5,

    "1994": 148.2,

    "1995": 152.4,

    "1996": 156.9,

    "1997": 160.5,

    "1998": 163,

    "1999": 166.6,

    "2000": 172.2,

    "2001": 177.1,

    "2002": 179.9,

    "2003": 184,

    "2004": 188.9,

    "2005": 195.3,

    "2006": 201.6,

    "2007": 207.3,

    "2008": 215.3,

    "2009": 214.5,

    "2010": 218.1,

    "2011": 224.9,

    "2012": 229.6,

    "2013": 233,

    "2014": 236.7,

    "2015": 237,

    "2016": 240,

    "2017": 245.1,

    "2018": 250.5

}
def load_data():

    def do_load(path):

        df = pd.read_csv(path)

        display(df.head())

        print(df.info())

        display(df.describe())

        return df

    print("Train data:")

    train_df = do_load("../input/train.csv")

    print("Test data:")

    test_df = do_load("../input/test.csv")

    return (train_df, test_df)



orig_train_df, orig_test_df = load_data()

orig_test_df["revenue"] = 0.0
def transform_data(df):

    import datetime as dt

    import ast

    print("Original size: ", len(df))

    

    # Fill empty values

    df["runtime"].fillna(df.runtime.median(), inplace=True)

    df["release_date"].fillna("01/01/00", inplace=True)

    

    # Add date columns from release date

    def parse_date(s):

        items = s.split("/")

        year = items[2]

        year = "20" + year if int(year) <= 19 else "19" + year

        return dt.datetime(int(year), int(items[0]), int(items[1]))

    df["release_date_obj"] = df.release_date.map(lambda i: parse_date(i))

    df["release_year"] = df.release_date_obj.dt.year

    df["release_quarter"] = (df.release_date_obj.dt.month - 1) // 3

    df["release_month"] = df.release_date_obj.dt.month

    df["release_week"] = df.release_date_obj.map(lambda i: i.isocalendar()[1])

    df["release_day"] = df.release_date_obj.dt.day

    df["release_weekday"] = df.release_date_obj.map(lambda i: i.weekday())

    

    # Add total movies releases per period

    releases_per_year = df.groupby(["release_year"]).release_year.count()

    release_per_quarter = df.groupby(["release_year", "release_quarter"]).release_quarter.count()

    release_per_month = df.groupby(["release_year", "release_month"]).release_month.count()

    release_per_week = df.groupby(["release_year", "release_week"]).release_week.count()

    df["releases_same_year"] = df.release_year.apply(lambda year: releases_per_year.loc[year])

    df["releases_same_quarter"] = df[["release_year", "release_quarter"]].apply(lambda i: release_per_quarter.loc[(i.release_year, i.release_quarter)], axis=1)

    df["releases_same_month"] = df[["release_year", "release_month"]].apply(lambda i: release_per_month.loc[(i.release_year, i.release_month)], axis=1)

    df["releases_same_week"] = df[["release_year", "release_week"]].apply(lambda i: release_per_week.loc[(i.release_year, i.release_week)], axis=1)

    

    # Adjust budget and revenue by inflation

    def adjust_by_inflation(year, amount):

        return INFLATION_CPI_HIST["2018"] / INFLATION_CPI_HIST[str(int(year))] * amount

    df["budget_adjusted"] = df[["release_year", "budget"]].apply(lambda i: adjust_by_inflation(i.release_year, i.budget), axis=1)

    df["revenue_adjusted"] = df[["release_year", "revenue"]].apply(lambda i: adjust_by_inflation(i.release_year, i.revenue), axis=1)

    df["budget_adjusted_log"] = df["budget_adjusted"].apply(np.log1p)

    

    # Json columns to list or dict

    def parse_array(s):

        try:

            return ast.literal_eval(s)

        except:

            return []

    for col in ["genres", "production_companies", "production_countries", "spoken_languages", "Keywords", "cast", "crew", "belongs_to_collection"]:

        df[col].fillna("['name': 'empty', 'job': 'empty']", inplace=True)

    df["genres"] = df.genres.map(lambda s: [i["name"] for i in parse_array(s)])

    df["production_companies"] = df.production_companies.map(lambda s: [i["name"] for i in parse_array(s)])

    df["production_countries"] = df.production_countries.map(lambda s: [i["name"] for i in parse_array(s)])

    df["spoken_languages"] = df.spoken_languages.map(lambda s: [i["name"] for i in parse_array(s)])

    df["keywords"] = df.Keywords.map(lambda s: [i["name"] for i in parse_array(s)])

    df["cast"] = df.cast.map(lambda s: parse_array(s))

    df["actors"] = df.cast.map(lambda c: [i["name"] for i in c])

    df["crew"] = df.crew.map(lambda s: [(i["job"], i["name"]) for i in parse_array(s)])

    df["collections"] = df.belongs_to_collection.map(lambda s: [i["name"] for i in parse_array(s)])

    

    # Add columns for relevant crew members

    def find_crew_job(row, job):

        return [i[1] for i in row if i[0] == job]

    df["crew_producer"] = df.crew.map(lambda i: find_crew_job(i, "Producer"))

    df["crew_exec_producer"] = df.crew.map(lambda i: find_crew_job(i, "Executive Producer"))

    df["crew_director"] = df.crew.map(lambda i: find_crew_job(i, "Director"))

    df["crew_screenplay"] = df.crew.map(lambda i: find_crew_job(i, "Screenplay"))

    df["crew_editor"] = df.crew.map(lambda i: find_crew_job(i, "Editor"))

    df["crew_casting"] = df.crew.map(lambda i: find_crew_job(i, "Casting"))

    df["crew_photography"] = df.crew.map(lambda i: find_crew_job(i, "Director of Photography"))

    df["crew_music"] = df.crew.map(lambda i: find_crew_job(i, "Original Music Composer"))

    df["crew_writer"] = df.crew.map(lambda i: find_crew_job(i, "Writer"))

    df["crew_art"] = df.crew.map(lambda i: find_crew_job(i, "Art Direction"))

    

    # Gender

    df["actors_gender_0"] = df.cast.map(lambda c: sum(1 for i in c if i["gender"] == 0))

    df["actors_gender_1"] = df.cast.map(lambda c: sum(1 for i in c if i["gender"] == 1))

    df["actors_gender_2"] = df.cast.map(lambda c: sum(1 for i in c if i["gender"] == 2))

    df["actors_first_gender"] = df.cast.map(lambda c: c[0]["gender"] if len(c) > 0 else 0)

    df["actors_second_gender"] = df.cast.map(lambda c: c[1]["gender"] if len(c) > 1 else 0)

    df["actors_third_gender"] = df.cast.map(lambda c: c[2]["gender"] if len(c) > 2 else 0)

    

    # Add columns for crew and actors size

    df["crew_size"] = df.crew.map(lambda i: len(i))

    df["actors_size"] = df.actors.map(lambda i: len(i))

    

    # Revenue to millions and log scale

    df["revenue_log"] = df["revenue"].apply(np.log1p)

    df["revenue"] = df["revenue"] / 1000000.0

    

    # Add column if it belongs to a collection

    df["has_collection"] = df.collections.map(lambda i: 1 if len(i) > 0 else 0)

    

    # Add column if it has a homepage

    df["has_homepage"] = 0

    df.loc[df.homepage.isnull() == False, "has_homepage"] = 1

    

    # Split text columns into tokens

    from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

    ignore_tokens = set(ENGLISH_STOP_WORDS)

    ignore_tokens.update(["", "-", "--", "äì", "&"])

    for col in ["title", "original_title", "tagline", "overview"]:

        df[col].fillna("", inplace=True)

    def clean_str(s):

        for i in [".", "\"", "\'", ",", ":"]:

            s = s.replace(i, "")

        return s

    def create_tokens(s):

        tokens = [clean_str(i.lower().strip()) for i in s.split(" ")]

        return [i for i in tokens if i not in ignore_tokens]

    df["title_tokens"] = df.title.map(lambda i: create_tokens(i))

    df["original_title_tokens"] = df.original_title.map(lambda i: create_tokens(i))

    df["tagline_tokens"] = df.tagline.map(lambda i: create_tokens(i))

    df["overview_tokens"] = df.overview.map(lambda i: create_tokens(i))

    

    # Drop columns

    df.drop(["belongs_to_collection", "homepage", "imdb_id", "original_title", "overview", "poster_path", "release_date", 

             "tagline", "cast", "Keywords", "status"], axis=1, inplace=True, errors="ignore")

    

    

    print("Altered size: ", len(df))

    display(df.head())

    return df





train_df = transform_data(orig_train_df.copy())

test_df = transform_data(orig_test_df.copy())
OTHER_CATEGORY = "other"

#CATEGORICAL_METHOD = "LARGEST_REVENUE"

CATEGORICAL_METHOD = "MOST_COMMON"



def get_categorical_best_values(df, col, n):

    return get_values_with_largest_revenue(df, col, n) if CATEGORICAL_METHOD == "LARGEST_REVENUE" else get_most_common_values(df, col, n)



def get_values_with_largest_revenue(df, col, n):

    data = []

    for _, row in df[[col, "revenue_adjusted"]].iterrows():

        row_values = row[col] if type(row[col]) == list else list([row[col]])

        for a in row_values:

            data.append([a, row["revenue_adjusted"]])

    actors_df = pd.DataFrame(data, columns=[col, "revenue"])

    actors_df = actors_df.groupby(col).sum().sort_values(by="revenue", ascending=False)

    values = set(i for i in actors_df.head(n).index.values)

    values.add(OTHER_CATEGORY)

    return values



def get_most_common_values(df, col, n):

    from collections import Counter

    values = df[col].values

    if type(values[0]) == list: 

        values = [i for items in values for i in items]

    most_common = set(word for word, _ in Counter(values).most_common(n))

    most_common.add(OTHER_CATEGORY)

    return most_common
for i in ["actors", "keywords", "production_companies", "crew_director", "crew_producer", "crew_music", "crew_screenplay", "collections", 

         "title_tokens", "overview_tokens", "tagline_tokens", "original_title_tokens", "original_language"]:

    print(i, ":")

    for v in get_categorical_best_values(train_df, i, 20):

        print("\t", v)
def explore_budget_revenue(df):

    df.plot(kind='scatter', x='budget_adjusted', y='revenue_adjusted', color='red', figsize=(10, 10), title="Budget vs Revenue (Inflation adj)")



    

explore_budget_revenue(train_df)
def explore_budget(df):

    import matplotlib.pyplot as plt

    df = df.copy()

    df["budget_log"] = df["budget"].apply(np.log1p)

    df["revenue_log"] = df["revenue"].apply(np.log1p)

    

    plt.figure(figsize=(15, 8))

    plt.subplot(2, 2, 1)

    plt.title("Revenue histogram")

    df["revenue"].hist(bins=100)

    

    plt.subplot(2, 2, 2)

    plt.title("Log Revenue histogram")

    df["revenue_log"].hist(bins=100)

    

    plt.subplot(2, 2, 3)

    plt.title("Budget histogram")

    df["budget"].hist(bins=100)

    

    plt.subplot(2, 2, 4)

    plt.title("Log Budget histogram")

    df["budget_log"].hist(bins=100)

    plt.show()

    

    df.plot(kind="scatter", x="budget", y="revenue", title="Budget vs Revenue", figsize=(10, 8))

    df.plot(kind="scatter", x="budget_log", y="revenue_log", title="Log Budget vs Log Revenue", figsize=(10, 8))

    

    

explore_budget(orig_train_df)
def explore_release_date(df):

    print("Release year")

    max_year = df.groupby("release_year").release_year.count().idxmax()

    print("First year: {}, Latest year: {}, Max year: {}".format(df.release_year.min(), df.release_year.max(), max_year))

    df["release_year"].hist(bins=100, figsize=(10,10))

    print("Years with most releases:")

    print(df.groupby("release_year").release_year.value_counts().nlargest(20))

    

explore_release_date(train_df)
def explore_genres(df):

    import matplotlib.pyplot as plt

    print("Genres")

    data = [i for values in df.genres.values for i in values]

    all_genres = set(data)

    print("All genres {}: {}".format(all_genres, len(all_genres)))

    data_dict = dict([(g, 0) for g in all_genres])

    for g in data:

        data_dict[g] += 1

        

    plt.figure(figsize=(20,10))   

    plt.hist(data, bins=len(all_genres))

    plt.show()

    

explore_genres(train_df)
def explore_popularity(df):

    df.plot(kind="scatter", x="popularity", y="revenue_adjusted", figsize=(10,10), title="Popularity vs Revenue")

    df[df.popularity < 20].plot(kind="scatter", x="popularity", y="revenue_adjusted", figsize=(10,10), title="Popularity vs Revenue (0-20)")

    

explore_popularity(train_df)
def explore_runtime(df):

    df.plot(kind="scatter", x="runtime", y="revenue_adjusted", figsize=(10, 10), title="Runtime vs Revenue")

    

explore_runtime(train_df)
def explore_actors(df):

    data = []

    for _, row in df[["actors", "revenue_adjusted"]].iterrows():

        rev = row["revenue_adjusted"] / 1000000.0

        for a in row["actors"]:

            data.append([a, rev])

    actors_df = pd.DataFrame(data, columns=["actor", "revenue"])

    actors_df = actors_df.groupby("actor").sum().sort_values(by="revenue", ascending=False)

    display(actors_df.head(30))

    

    

explore_actors(train_df)

def explore_crew(df):

    import ast

    def parse_array(s):

        try:

            return ast.literal_eval(s)

        except:

            return []

    df = df[["crew", "revenue"]].copy()

    df["crew"].fillna("['job': 'empty']", inplace=True)

    df["crew"] = df.crew.map(lambda s: [i for i in parse_array(s)])

    crew_dict = {}

    for crew in df["crew"].values:

        for i in crew:

            crew_dict[i["job"]] = crew_dict.get(i["job"], 0) + 1

    return crew_dict





crew_dict = explore_crew(orig_train_df)

sorted(crew_dict.items(), key=lambda i: i[1], reverse=True)
def explore_movie_crew(df, name):

    import ast

    return ast.literal_eval(df[df.title.str.contains(name, case=False)]["crew"].values[0])



#[(i["job"], i["name"]) for i in explore_movie_crew(orig_train_df, "Jurassic")]

#[(i["job"], i["name"]) for i in explore_movie_crew(orig_train_df, "Back to the future")]

[(i["job"], i["name"]) for i in explore_movie_crew(orig_train_df, "Titanic")]
@dataclass

class Feature:

    col: str

    ftype: str

    cat_size: int = 500

        

TRAIN_SET_SIZE = 0.8

LABEL_COL = "revenue_log"

        

FEATURES = [

    Feature("genres", "categorical"),

    Feature("actors_first_gender", "categorical"),

    Feature("actors_second_gender", "categorical"),

    Feature("actors_third_gender", "categorical"),

    Feature("original_language", "categorical_constrained", 10),

    Feature("production_countries", "categorical_constrained", 10),

    Feature("spoken_languages", "categorical_constrained", 10),

    Feature("production_companies", "categorical_constrained", 10),

    Feature("actors", "categorical_constrained", 30),

    Feature("crew_producer", "categorical_constrained", 20),

    Feature("crew_exec_producer", "categorical_constrained", 20),

    Feature("crew_director", "categorical_constrained", 20),

    Feature("crew_screenplay", "categorical_constrained", 20),

    Feature("crew_editor", "categorical_constrained", 20),

    Feature("crew_casting", "categorical_constrained", 20),

    Feature("crew_photography", "categorical_constrained", 20),

    Feature("crew_music", "categorical_constrained", 20),

    Feature("crew_writer", "categorical_constrained", 20),

    Feature("crew_art", "categorical_constrained", 20),    

    Feature("keywords", "categorical_constrained", 30),

    Feature("title_tokens", "categorical_constrained", 30),    

    #Feature("original_title_tokens", "categorical_constrained", 30),    

    #Feature("tagline_tokens", "categorical_constrained", 30),    

    #Feature("overview_tokens", "categorical_constrained", 30),    

    Feature("budget_adjusted_log", "numerical"),

    #Feature("release_year", "categorical_constrained", 10),

    Feature("release_quarter", "categorical"),

    Feature("release_month", "categorical"),

    Feature("release_week", "categorical_constrained", 10),

    Feature("release_weekday", "categorical"),

    Feature("releases_same_year", "numerical"),

    Feature("releases_same_quarter", "numerical"),

    Feature("releases_same_month", "numerical"),

    Feature("releases_same_week", "numerical"),

    Feature("runtime", "numerical"),

    Feature("has_collection", "numerical"),

    Feature("has_homepage", "numerical"),

    Feature("crew_size", "numerical"),

    Feature("actors_size", "numerical"),

    Feature("actors_gender_0", "numerical"),

    Feature("actors_gender_1", "numerical"),

    Feature("actors_gender_2", "numerical"),

    Feature("popularity", "numerical")

]

FEATURES_COL = [f.col for f in FEATURES]



def create_samples(df, categories_dict, trainSetProportion, log=True):

    from sklearn.preprocessing import MinMaxScaler

    from sklearn.model_selection import train_test_split

    

    # Convert categorical columns to one hot encoding

    for f in FEATURES:

        if f.ftype in ["categorical", "categorical_constrained"]:

            df[f.col] = df[f.col].apply(lambda i: labels_to_one_hot(i, f, categories_dict))

    

    # Normalize numeric columns

    scaler = MinMaxScaler()

    numeric_cols = [f.col for f in FEATURES if f.ftype == "numerical"]

    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    

    # Create samples

    tot_samples = len(df)

    tot_features = sum(1 if f.ftype == "numerical" else len(categories_dict[f.col]) for f in FEATURES)

    X = np.zeros((tot_samples, tot_features), dtype="float32")

    Y = np.zeros((tot_samples, 1), dtype="float32")

    for i in range(tot_samples):

        fidx = 0

        for f in FEATURES:

            col_value = df[f.col].iloc[i]

            col_len = 1 if f.ftype == "numerical" else len(col_value)

            X[i, fidx : fidx + col_len] = col_value

            fidx += col_len

        Y[i] = df[LABEL_COL].iloc[i]

        

    if log:

        print("All X: ", X.shape)

        print("All Y: ", Y.shape)

    

    # Split in train-test set

    if trainSetProportion < 1.0:

        X_train, X_test, Y_train, Y_test, df_train, df_test = train_test_split(X, Y, df, test_size=1-trainSetProportion, random_state=RAND_SEED)

        if log:

            print("Train set:", X_train.shape, Y_train.shape, X_train.dtype)

            print("Test set:", X_test.shape, Y_test.shape)

    else:

        X_train = X

        X_test = X

        Y_train = Y

        Y_test = Y

        df_train = df

        df_test = df

    return (X_train, X_test, Y_train, Y_test, df_train, df_test, categories_dict)



def create_categories_dict(df):

    categories_dict = {}

    for f in (f for f in FEATURES if f.ftype in ["categorical"]):

        categories_dict[f.col] = labels_to_dict(df[f.col])

    for f in (f for f in FEATURES if f.ftype == "categorical_constrained"):

        categories_dict[f.col] = dict((k, i)for i, k in enumerate(get_most_common_values(df, f.col, f.cat_size)))

        

    print("Categories:")

    for categ, categ_map in categories_dict.items():

        print("\t{}: {}".format(categ, len(categ_map)))

    return categories_dict

    

def labels_to_dict(df_col):

    s = set()

    for items in df_col.values:

        items = items if type(items) == list else list([items])

        s.update(items)

    s.add(OTHER_CATEGORY)

    return dict((item,idx) for idx,item in enumerate(s))

    

def labels_to_one_hot(values, feature, categories_dict):

    values = values if type(values) == list else list([values])

    x = np.zeros(len(categories_dict[feature.col]), dtype="float32")

    other_idx = categories_dict[feature.col][OTHER_CATEGORY]

    for value in values:

        x[categories_dict[feature.col].get(value, other_idx)] = 1

    return x

    

print("Creating samples...")

categories_dict = create_categories_dict(train_df.copy().append(test_df.copy()))

samples = create_samples(train_df.copy(), categories_dict, TRAIN_SET_SIZE)

all_samples = create_samples(train_df.copy(), categories_dict, 1.0, log=False)
def train_model(samples, epochs, batch_size, log_interval, load_pretrained_model, log=True):

    import keras

    from keras import backend as K

    import os

    import time



    X_train, X_test, Y_train, Y_test, _, _, _ = samples

    features = X_train.shape[1]

    

    # Create Keras model

    if log: print("Network architecture:")

    model = keras.models.Sequential([

        keras.layers.Dense(64, input_shape=(features,), activation="relu"),

        keras.layers.Dropout(0.2),

        keras.layers.Dense(64, activation="relu"),

        keras.layers.Dropout(0.2),

        keras.layers.Dense(1)

    ])

    '''

    model = keras.models.Sequential([

        keras.layers.Dense(64, input_shape=(features,), use_bias=False),

        keras.layers.BatchNormalization(),

        keras.layers.Activation('relu'),

        keras.layers.Dropout(0.5),

        keras.layers.Dense(64, use_bias=False),

        keras.layers.BatchNormalization(),

        keras.layers.Activation('relu'),

        keras.layers.Dropout(0.5),

        keras.layers.Dense(1)

    ])

    '''

    if log: model.summary()

    

    # Compile model

    model.compile(optimizer='adam', loss='mse')

    

    # Callbacks

    def print_progress(epoch, logs):

        if log and epoch % log_interval == 0:

            loss = logs["loss"]

            val_loss = logs["val_loss"]

            rmse = np.sqrt(loss)

            val_rmse = np.sqrt(val_loss)

            print("\tEpoch {}, Loss: {}, Val Loss: {}, Val RMSE: {}".format(epoch, round(loss,2), round(val_loss,2), round(val_rmse,2)))

        

    model_path = "trained_model.hdf5"

    fit_callbacks = [

        keras.callbacks.ModelCheckpoint(model_path, monitor="loss", save_weights_only=True),

        keras.callbacks.LambdaCallback(on_epoch_end = lambda epoch, logs: print_progress(epoch, logs))

    ]

    

    # Load pre-trained model

    if load_pretrained_model and os.path.isfile(model_path):

        print("Loading pre-trained model: ", model_path)

        model.load_weights(model_path)

    

    # Train model

    if epochs > 0:

        if log: print("Training (epochs: {}, batch: {})...".format(epochs, batch_size))

        start_time = time.time()

        history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=0, callbacks=fit_callbacks, 

                            validation_data=(X_test, Y_test))

        elapsed_time_min = (time.time() - start_time) / 60.0

        model.save_weights(model_path)

        if log: print("Done ({} mins)".format(round(elapsed_time_min, 2)))



        # Test accuracy on train and test set

        if log: print("Predicting...")

        score_train = model.evaluate(X_train, Y_train, verbose=0)

        score_test = model.evaluate(X_test, Y_test, verbose=0)

        stats = {

            "training_time_min": elapsed_time_min,

            "train_loss": score_train,

            "test_loss": score_test,

            "train_rmse": np.sqrt(score_train),

            "test_rmse": np.sqrt(score_test)

        }

        if log:

            print("Train set - Loss: {}, RMSE: {}".format(round(stats["train_loss"],2), round(stats["train_rmse"],2)))

            print("Test set - Loss: {}, RMSE: {}".format(round(stats["test_loss"],2), round(stats["test_rmse"],2)))



        # Plot training

        if log: plot_training_progress(history)

    else:

        history = {}

        stats = {}

        print("Skipping training, using pre-loaded model")

    

    return (model, history, stats, model_path)

    

def plot_training_progress(history):

    from matplotlib import pyplot as plt

    def plot(title, x_axis_label, y_axis_label, y_label, y_data, y_val_label, y_val_data):

        x = np.arange(len(y_data))

        fig, ax = plt.subplots(1,1, figsize=(20,8))

        plt.scatter(x, y_data, c="red", label=y_label)

        plt.plot(x, y_data, color="red")

        plt.scatter(x, y_val_data, c="blue", label=y_val_label)

        plt.plot(x, y_val_data, color="blue")

        plt.title(title)

        plt.xlabel(x_axis_label)

        plt.ylabel(y_axis_label)

        plt.legend()

        plt.show()

    

    y_loss = np.array(history.history["loss"])

    y_val_loss = np.array(history.history["val_loss"])

    plot("Training progress - Loss", "Epochs", "Loss", "Train loss", y_loss, "Val loss", y_val_loss)



    

    

train_result = train_model(samples, epochs=25, batch_size=64, log_interval=1, load_pretrained_model=False, log=True)
print("Training runs stats:")

pd.DataFrame([

    ["Base - 20 epochs, 32 batch, 0.2 dropoup, 2x64 lagers", 1.79, 2.6],

    ["64 epoch", 1.95, 2.51],

    ["RmsProp", 2.17, 2.65],

    ["Small features", 2.26, 2.47],

    ["Small features, 30 epochs", 2.2, 2.46],

    ["Small features, 25 epochs, 80% train", 2.26, 2.45],

    ["Most common features", 1.9, 2.44],

    ["Smaller features", 2.0, 2.4],

    ["No title tokens", 2.04, 2.42],

], columns=["Config", "Train RMSE", "Test RMSE"])
def train_with_kfold(samples, categories_dict, epochs, batch_size, splits):

    from sklearn.model_selection import KFold

    X, _, Y, _, df, _, _ = samples

    print("Training K folds: ", splits)

    kfold = KFold(n_splits=splits, random_state=RAND_SEED, shuffle=True)

    i = 1

    results = []

    for train_index, test_index in kfold.split(X):

        print("Fold {}), Train: {}, Test: {}, epochs: {}, batch: {}".format(i, len(train_index), len(test_index), epochs, batch_size))

        fold_samples = (X[train_index], X[test_index], Y[train_index], Y[test_index], df.iloc[train_index], df.iloc[test_index], categories_dict)

        train_result = train_model(fold_samples, epochs=epochs, batch_size=batch_size, log_interval=100000, load_pretrained_model=False, log=False)

        results.append(train_result)

        model, history, stats, model_path = train_result

        print("\tTrain set - Loss: {}, RMSE: {}".format(round(stats["train_loss"],2), round(stats["train_rmse"],2)))

        print("\tTest set - Loss: {}, RMSE: {}".format(round(stats["test_loss"],2), round(stats["test_rmse"],2)))

        i += 1

        

    all_rmse = np.array([i[2]["test_rmse"] for i in results])

    print("RMSE avg: {}, min: {}, max: {}".format(round(all_rmse.mean(),2), round(all_rmse.min(),2), round(all_rmse.max(),2)))

    plot_training_progress(results[all_rmse.argmin()][1])

    return results

    

    

#kfold_results = train_with_kfold(all_samples, categories_dict, epochs=15, batch_size=32, splits=10)   
def generate_submission(model, df, categories_dict, using_log_label=False):

    X, _, Y, _, _, _, _ = create_samples(df.copy(), categories_dict, 1.0)

    Y_pred = model.predict(X).reshape((len(Y)))

    if using_log_label:

        Y_pred = np.expm1(Y_pred)

    else:

         Y_pred = Y_pred * 1000000.0

    submission = pd.DataFrame({"id": df["id"].values, "revenue": Y_pred})

    submission.to_csv('submission.csv', index=False)

    return submission

    

submission = generate_submission(train_result[0], test_df, categories_dict, using_log_label=True)

submission.head(30)
def generate_submission_as_average(models, df, categories_dict):

    X, _, Y, _, _, _, _ = create_samples(df.copy(), categories_dict, 1.0)

    tot_samples = Y.shape[0]

    tot_models = len(models)

    all_predictions = np.zeros((tot_models, tot_samples))

    for i, model in enumerate(models): 

        Y_pred = model.predict(X).reshape((tot_samples)) * 1000000.0

        all_predictions[i] = Y_pred

    avg_pred = all_predictions.mean(axis=0)

    submission = pd.DataFrame({"id": df["id"].values, "revenue": avg_pred})

    submission.to_csv('submission.csv', index=False)

    return submission

    

#submission = generate_submission_as_average([i[0] for i in kfold_results], test_df, categories_dict)

#submission.head(30)
sum(i for i in range(10))