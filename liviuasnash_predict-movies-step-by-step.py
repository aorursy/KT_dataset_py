# Import

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import os

import collections

import time

from sklearn.preprocessing import LabelEncoder

print(os.listdir("../input"))
# Load files

train = pd.read_csv("../input/tmdb-box-office-prediction/train.csv")

test = pd.read_csv("../input/tmdb-box-office-prediction/test.csv")

imdb = pd.read_csv("../input/imdb-web-scraping-4/imdb.csv")

train.head()
imdb.head()
# Combine train and test to allow feature engineering on the combined data. Store the ids and the revenue for future use

train_rows = train.shape[0]

y_train = np.log1p(train["revenue"]).values

train_ID = train['id']

test_ID = test['id']



all_data = pd.concat([train, test], sort=False)

all_data.drop(['revenue'], axis=1, inplace=True)

all_data.drop(['id'], axis=1, inplace=True)



# Combine the result with the imdb data

all_data = pd.merge(all_data, imdb, how="left", on="imdb_id")

all_data.head()
# Several columns (e.g. genres) are lists of values - split them to dictionaries for easier processing

import ast

for c in ['belongs_to_collection', 'genres', 'production_companies', 'production_countries', 'spoken_languages', 

          'Keywords', 'cast', 'crew']:

    all_data[c] = all_data[c].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x))
# The EDA and feature engineering for the dict columns seems similar. Create a few basic common utilities

def print_few_values(col_name):

    print("Sample values for", col_name)

    all_data[col_name].head(5).apply(lambda x: print(x))

    

def dictionary_sizes(col_name):

    return (all_data[col_name].apply(lambda x: len(x)).value_counts())



def print_dictionary_sizes(col_name):

    print("\n===================================================")

    print("Distribution of sizes for", col_name)

    print(dictionary_sizes(col_name))

    

# returns a list of tuples of names for a given row of a column

def dict_name_list(d, name="name"):

    return ([i[name] for i in d] if d != {} else [])



# returns a list of tuples of the (id,name) pairs for a given row of a column

def dict_id_name_list(d, name="name"):

    return ([(i["id"],i[name]) for i in d] if d != {} else [])



# returns a list of names for a given column

def col_name_list(col_name, name="name"):

    # Get a list of lists of names

    name_list_list = list(all_data[col_name].apply(lambda x: dict_name_list(x, name)).values)

    # Merge into 1 list

    return ([i for name_list in name_list_list for i in name_list])



# returns a list of tuples of the (id,name) pairs for a given column

def col_id_name_list(col_name, name="name"):

    # Get a list of lists of (id,name) tuples

    tuple_list_list = list(all_data[col_name].apply(lambda x: dict_id_name_list(x, name)).values)

    # Merge into 1 list

    return ([i for tuple_list in tuple_list_list for i in tuple_list])



def get_names_counter(col_name, name="name"):

    name_list = col_name_list(col_name, name)

    return (collections.Counter(name_list))

    

def print_top_names(col_name, name="name"):

    print("\n===================================================")

    print("Top {0}s for {1}".format(name, col_name))

    c = get_names_counter(col_name, name)

    print(c.most_common(20))

    

def EDA_dict(col_name):

    print_few_values(col_name)

    print_dictionary_sizes(col_name)

    print_top_names(col_name)

    

def add_dict_size_column(col_name):

    all_data[col_name + "_size"] = all_data[col_name].apply(lambda x: len(x) if x != {} else 0)



def add_dict_id_column(col_name):

    c = col_name + "_id"

    all_data[c] = all_data[col_name].apply(lambda x: x[0]["id"] if x != {} else 0)

    all_data[c] = all_data[c].astype("category")



# for each of the top values in the dictionary, add an column indicating if the row belongs to it

def add_dict_indicator_columns(col_name):

    c = get_names_counter(col_name)

    top_names = [x[0] for x in c.most_common(20)]

    for name in top_names:

        all_data[col_name + "_" + name] = all_data[col_name].apply(lambda x: name in dict_name_list(x))

        

def drop_column(col_name):

    all_data.drop([col_name], axis=1, inplace=True)

    

def feature_engineer_dict(col_name):

    add_dict_size_column(col_name)

    max_size = dictionary_sizes(col_name).index.max()

    if max_size == 1:

        add_dict_id_column(col_name)

    else:

        add_dict_indicator_columns(col_name)

    drop_column(col_name)

    

def encode_column(col_name):

    lbl = LabelEncoder()

    lbl.fit(list(all_data[col_name].values)) 

    all_data[col_name] = lbl.transform(list(all_data[col_name].values))
col_name = "belongs_to_collection"

# EDA

EDA_dict(col_name)
# Feature engineering

feature_engineer_dict(col_name)
col_name="genres"

EDA_dict(col_name)
# Feature engineering

feature_engineer_dict(col_name)
col_name="production_companies"

EDA_dict(col_name)
# Feature engineering

feature_engineer_dict(col_name)
col_name="production_countries"

EDA_dict(col_name)
# Feature engineering

feature_engineer_dict(col_name)
col_name="spoken_languages"

EDA_dict(col_name)
# Feature engineering

feature_engineer_dict(col_name)
# Check for nulls

all_data.loc[all_data["release_date"].isnull()]

# There is 1 movie w/o a release date. Looking it up in imdb by imdb_id = tt0210130, it was release in march 2000

all_data.loc[all_data["release_date"].isnull(), "release_date"] = "05/01/2000"

# Parse the string to a date

all_data["release_date"] = pd.to_datetime(all_data["release_date"])

# Create columns for each part of the date

all_data["release_date_weekday"] = all_data["release_date"].dt.weekday.astype(int)

all_data["release_date_month"] = all_data["release_date"].dt.month.astype(int)

all_data["release_date_year"] = all_data["release_date"].dt.year.astype(int)

# The year is formatted as yy as opposed to yyyy, and therefore the century is sometimes incorrect.

all_data["release_date_year"] = np.where(all_data["release_date_year"]>2019, all_data["release_date_year"]-100, all_data["release_date_year"])

# Compare with imdb

all_data["imdb_year"]=all_data["imdb_year"].astype(int)

# Compare the values

print(all_data[abs(all_data["release_date_year"]-all_data["imdb_year"])>2][["imdb_id","release_date_year","imdb_year"]])

# There are 4 movies from the 20th century that appear as 21st century due to the issue above. Fix them.

all_data.loc[abs(all_data["release_date_year"]-all_data["imdb_year"])>2, "release_date_year"] = all_data["imdb_year"]

drop_column("release_date")

drop_column("imdb_year")
all_data.loc[all_data["runtime"].isnull(), "imdb_id"]

all_data.loc[all_data["runtime"]==0, "imdb_id"]

# A few movies don't have runtime. Get their imdb_id to look them up on imdb.

all_data.loc[all_data["runtime"]==0, "runtime"] = all_data["imdb_runtime"]

all_data.loc[all_data["runtime"].isnull(), "runtime"] = all_data["imdb_runtime"]

# Most of them have values in imdb. Found one more online.Filling the remaining ones with the column average 

all_data["runtime"].fillna(all_data["runtime"].mean(), inplace=True)

drop_column("imdb_runtime")
encode_column("original_language")
# Fix Gender

# Both cast and crew have lots of entries with gender == 0 (i.e. unknown). 

# Use the rest of the entries to decide if a first name is male or female, and categorize the unknown ones by that



# Create a DataFrame of all the names in the cast and crew columns and how many times they appear as each gender

def get_names():

    tuple_list_list = list(all_data["cast"].apply(lambda x: [(cm["name"].split(" ")[0],cm["gender"]) for cm in x]).values)

    tuple_list_list.extend(list(all_data["crew"].apply(lambda x: [(cm["name"].split(" ")[0],cm["gender"]) for cm in x]).values))

    tuple_list = [i for tuple_list in tuple_list_list for i in tuple_list]

    names = set()

    c = [None] * 3

    for gender in range (0, 3):

        t = [i[0] for i in tuple_list if i[1] == gender]

        names.update(set(t))

        c[gender] = collections.Counter(t)

    names_df = pd.DataFrame(data = list(names), columns=["name"])

    names_df["female_count"] = names_df["name"].apply(lambda x: c[1][x])

    names_df["male_count"] = names_df["name"].apply(lambda x: c[2][x])

    names_df["unknown_count"] = names_df["name"].apply(lambda x: c[0][x])

    names_df["total_count"] = names_df.apply(lambda x: x["female_count"]+x["male_count"]+x["unknown_count"], axis=1)

    return (names_df)



names_df = get_names()



# Classify each name with a gender (or leave as 0 if there is not enough data to decide or the name is unisex)

def ClassifyName(row):

    fcount = row["female_count"]

    mcount = row["male_count"]

    cl = "TBD"

    gender = 0

    if (fcount + mcount < 5):

        cl = "too few names"

    elif (fcount == 0):

        gender = 2

        cl = "high confidence"

    elif (mcount == 0):

        gender = 1

        cl = "high confidence"

    else:  # both are > 0

        # If a name is 90+% male or female, even if unisex, we'll bet on the majority

        if (mcount / float(fcount) < 0.1):

            gender = 1

            cl = "unisex - pick one"

        elif (fcount / float(mcount) < 0.1):

            gender = 2

            cl = "unisex - pick one"

        else: #unisex, no sex more than 90% - leave as undefined

            cl = "unisex - TBD"

    return [gender, cl]



names_df = names_df.merge(names_df.apply(lambda x: ClassifyName(x), axis=1, result_type="expand"), left_index=True, right_index=True)

names_df.rename(columns={0:"gender", 1:"classification"}, inplace=True)

# pd.pivot_table(names_df, index=["classification"], values="total_count", aggfunc=[np.sum, "count"])



# Create a dictionary that maps each name that as male or female (to not add names with undefined gender)

names_to_gender = dict()

def update_names_to_gender(row):

    if (row["gender"] > 0):

        names_to_gender[row["name"]] = row["gender"]

j = names_df.apply(lambda x: update_names_to_gender(x), axis=1)

# names_to_gender



# Method to fix the entries with gender == 0 (in cast or crew)

def fix_unknown_gender_row(row):

    for cm in row:

        if cm["gender"] == 0:

            name = cm["name"]

            first_name = name.split(" ")[0]

            if (first_name in names_to_gender):

                cm["gender"] = names_to_gender[first_name]

    

def fix_unknown_gender_col(col_name):

    all_data[col_name].apply(lambda x: fix_unknown_gender_row(x))
col_name = "crew"

EDA_dict(col_name)

# The crew column is different than others as it has much more properties. Let's look at jobs

print_top_names(col_name, name="job")
fix_unknown_gender_col(col_name)



def get_most_common_gender(row):

    genders = [mc["gender"] for mc in row]

    if (genders == []):

        return -1

    return (collections.Counter(genders).most_common(1)[0][0])

        

# Split by job

top_jobs = [x[0] for x in get_names_counter(col_name, "job").most_common(10)]

for job in top_jobs:

    c = "crew_" + job

    print("-------------\n",job,"\n-------------")

    all_data[c] = all_data[col_name].apply(lambda x: [crew_member for crew_member in x if crew_member["job"]==job])

    all_data[c + "_common_gender"] = all_data[c].apply(lambda x: get_most_common_gender(x))

    EDA_dict(c)

    feature_engineer_dict(c)

drop_column(col_name)
col_name = "cast"

# EDA_dict(col_name)
fix_unknown_gender_col(col_name)



# Split by gender

genders = [x[0] for x in get_names_counter(col_name, "gender").most_common(2)]

for gender in genders:

    c = "cast_gender_" + str(gender)

    all_data[c] = all_data[col_name].apply(lambda x: [cast_member for cast_member in x if cast_member["gender"]==gender])

    #print("-------------\n",gender,"\n-------------")

    EDA_dict(c)

    feature_engineer_dict(c)

drop_column(col_name)
# Add an indicator for lack of homepage

all_data["no_homepage"] = all_data["homepage"].isnull()

print(all_data["no_homepage"].value_counts())

# Create a list of all homepages that appear more than once

multiple_homepages = all_data["homepage"].value_counts()

multiple_homepages = multiple_homepages[multiple_homepages>1].index.tolist()

print("There are ", len(multiple_homepages), "homepages which appear more than once")

# All the unique homepages are not interesting, replace them with null

all_data.loc[all_data["homepage"].isin(multiple_homepages) == False, "homepage"] = "" 

# Encode the strings

encode_column("homepage")
# all_data["budget"].value_counts().sort_index()

# all_data[(all_data.budget>0) & (all_data.budget<1000)][["budget","release_date_year","title"]]

all_data["no_budget"] =(all_data["budget"] == 0)

# all_data[(all_data.budget>0) & (all_data.budget<1000), "budget"] = all_data["budget"] * 1000000

# all_data.head()

all_data["log_budget"]=np.log1p(all_data["budget"])
# Ratings have the following format: Rated <rating> for <reason>

# Remove the word "Rating at the beginning"

all_data["mpaa_rating"] = all_data["mpaa_rating"].apply(lambda s: s.split(" ", 1)[1] if isinstance(s, str) else s)

# Set the reason to be the sentence after ""<rating> for"

all_data["mpaa_rating_reason"] = all_data["mpaa_rating"].apply(lambda s: s.split(" ", 2)[2] if isinstance(s, str) else s)

# Set the rating to be the first word

all_data["mpaa_rating"] = all_data["mpaa_rating"].apply(lambda s: s.split(" ", 1)[0] if isinstance(s, str) else s)

all_data["mpaa_rating"].value_counts()
encode_column("mpaa_rating")
def SplitReason(r):

    s = r.strip().replace("  ", " ").replace(", and for ", ", ").replace(", and ", ", ")

    rl = s.rsplit(" and ", 1)

    rl2 = rl[0].split(", ")

    if len(rl)>1:

        rl2.append(rl[1])

    return (rl2)



# Split the reasons, create an indicator column for the top reasons

col_name = "mpaa_rating_reason"

all_data[col_name] = all_data[col_name].apply(lambda s: SplitReason(s) if isinstance(s, str) else [])

reason_list_list = list(all_data[col_name].values)

top_reasons = [reason[0] for reason in collections.Counter([i for reason_list in reason_list_list for i in reason_list]).most_common(100)]

for reason in top_reasons:

    all_data[col_name + "_" + reason] = all_data[col_name].apply(lambda x: reason in x)

all_data.head()

drop_column(col_name)
# Drop unneeded columns



# Drop all columns which haven't been handled yet. We'll gradually add them back later on

all_data.drop(['imdb_id'], axis=1, inplace=True)

all_data.drop(['original_title'], axis=1, inplace=True)

all_data.drop(['overview'], axis=1, inplace=True)

all_data.drop(['poster_path'], axis=1, inplace=True)

all_data.drop(['status'], axis=1, inplace=True)

all_data.drop(['tagline'], axis=1, inplace=True)

all_data.drop(['title'], axis=1, inplace=True)

all_data.drop(['Keywords'], axis=1, inplace=True)



drop_column("imdb_release")

drop_column("imdb_budget")

drop_column("imdb_country")
# Split back to train and test

train = all_data[:train_rows]

test = all_data[train_rows:]

train.describe()
# Create a model

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from catboost import CatBoostRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import xgboost as xgb

import lightgbm as lgb



#Validation function

n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)

    rmse = np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv=kf))

    return(rmse)



def eval_model(model, name):

    start_time = time.time()

    score = rmsle_cv(model)

    print("{} score: {:.4f} ({:.4f}),     execution time: {:.1f}".format(name, score.mean(), score.std(), time.time()-start_time))
mod_lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.005, random_state=1))

eval_model(mod_lasso, "lasso")

mod_enet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

eval_model(mod_enet, "enet")

mod_cat = CatBoostRegressor(iterations=10000, learning_rate=0.01,

                            depth=5, eval_metric='RMSE',

                            colsample_bylevel=0.7, random_seed = 17, silent=True,

                            bagging_temperature = 0.2, early_stopping_rounds=200)

eval_model(mod_cat, "cat")

mod_gboost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=5, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state=5)

eval_model(mod_gboost, "gboost")

mod_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state=7, nthread=-1)

eval_model(mod_xgb, "xgb")

mod_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=8,

                              learning_rate=0.05, n_estimators=650,

                              max_bin=58, bagging_fraction=0.80,

                              bagging_freq=5, feature_fraction=0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf=7, min_sum_hessian_in_leaf=11)

eval_model(mod_lgb, "lgb")
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, base_models, meta_model, n_folds=5):

        self.base_models = base_models

        self.meta_model = meta_model

        self.n_folds = n_folds

   

    # We again fit the data on clones of the original models

    def fit(self, X, y):

        self.base_models_ = [list() for x in self.base_models]

        self.meta_model_ = clone(self.meta_model)

        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        

        # Train cloned base models then create out-of-fold predictions

        # that are needed to train the cloned meta-model

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))

        for i, model in enumerate(self.base_models):

            for train_index, holdout_index in kfold.split(X, y):

                instance = clone(model)

                self.base_models_[i].append(instance)

                instance.fit(X[train_index], y[train_index])

                y_pred = instance.predict(X[holdout_index])

                out_of_fold_predictions[holdout_index, i] = y_pred

                

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature

        self.meta_model_.fit(out_of_fold_predictions, y)

        return self

   

    #Do the predictions of all base models on the test data and use the averaged predictions as 

    #meta-features for the final prediction which is done by the meta-model

    def predict(self, X):

        meta_features = np.column_stack([

            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)

            for base_models in self.base_models_ ])

        return self.meta_model_.predict(meta_features)

    

mod_stacked = StackingAveragedModels(base_models = (mod_cat, mod_xgb, mod_gboost, mod_lgb), meta_model = mod_lasso)

eval_model(mod_stacked, "stacked")
def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))



def predict(model):

    model.fit(train.values, y_train)

    train_pred = model.predict(train.values)

    pred = np.expm1(model.predict(test.values))

    print(rmsle(y_train, train_pred))

    return (pred)
# Predict

# Start with a simple model

prediction = predict(mod_lasso)

prediction = predict(mod_enet)

prediction = predict(mod_xgb)

prediction = predict(mod_gboost)

prediction = predict(mod_lgb)

prediction = predict(mod_stacked)
# Submit

submission = pd.DataFrame()

submission['id'] = test_ID

submission['revenue'] = prediction

submission.to_csv('submission.csv', index=False)
# Log                     Estimate Public score

# baseline:               2.6454   2.68084    v2

# belong_to_collection:   2.6437   2.67052    v5

# genres:                 2.5476   2.59730    v6

# production_companies:   2.4396   2.49865    v7

# production_countries:   2.4340   2.47048    v8

# spoken_languages:       2.4316   2.46744    v9

# release_date:           2.4274   2.46433    v10

# runtime:                2.4034   2.42784    v11

# original_language       2.4050   2.42811    v12 - no improvement

# crew                    2.3649   2.38779    v13

# cast                    2.3714   2.40004    v14 - no improvement. Maybe it means that we are overfitting with so many columns.

# stacked model           2.0624   2.02996    v15

# Fix years (century)     2.0632   2.03133    v16 - didn't help

# homepage                2.0623   2.02096    v17

# added no_budget         2.0551   2.02131    v19 - didn't help

# added log(budget)       2.0417   1.99055    v20

# filled runtime from imdb2.0424   1.99148    v21 - didn't help

# fixed century of 4 items2.0448   1.99040    v22

# add mpaa_rating         2.0333   1.97524    v23

# add mpaa_rating_reason  2.0473   1.97890    v24

# add cat model, rem lasso2.1055   2.01712    v25

# bring back lasso        2.0402   1.97147    v26

# manual model param tune 2.0367   1.96204    v27