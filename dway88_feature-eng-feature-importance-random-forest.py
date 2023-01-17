import numpy as np

import pandas as pd



%load_ext autoreload

%autoreload 2



%matplotlib inline
!pip install fastai==0.7.0 --quiet
from fastai.imports import *

from fastai.structured import *



from pandas_summary import DataFrameSummary

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.preprocessing import MultiLabelBinarizer



from IPython.display import display



from sklearn import metrics



import ast
PATH = "../input/"

!ls {PATH}
df_raw = pd.read_csv(f'{PATH}train.csv', 

                     low_memory=False, 

                     parse_dates=["release_date"])
def display_all(df):

    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 

        display(df)
# look at the data

display_all(df_raw.tail().T)
# describe all the data

display_all(df_raw.describe(include='all').T)
df_raw.dtypes
df_raw.revenue = np.log1p(df_raw.revenue)
# transform release_date to multiple new columns containing information from the date

add_datepart(df_raw, 'release_date')

df_raw.dtypes
# we'll use it for one-hot encoding

mlb = MultiLabelBinarizer()
# look at a random movie genres

df_raw.iloc[1300]['genres']
def convertStringToList(strVal):

    if type(strVal) is not str:

        return  []

    else:

        return ast.literal_eval(strVal)
def formatDictColumnAndExtractNames(strVal):

    listOfItems = convertStringToList(strVal)

    return list(map(lambda x: x['name'], listOfItems))
def extractGenres(df):

    df['genres'] = df['genres'].apply(formatDictColumnAndExtractNames)



    return df.join(pd.DataFrame(mlb.fit_transform(df.pop('genres')),

                          columns=list(map(lambda x: 'genre_'+x,mlb.classes_)),

                          index=df.index))
df_raw = extractGenres(df_raw)
def extractCommonProdCompanies(df):

    df['production_companies'] = df['production_companies'].apply(formatDictColumnAndExtractNames)



    companiesCount = df['production_companies'].apply(pd.Series).stack().value_counts()

    companiesToKeep = companiesCount[companiesCount > 30].keys()

    print("We'll keep the companies that appear more than 30 times:")

    print(companiesToKeep)



    df['production_companies'] = df['production_companies'].apply(lambda x: list(filter(lambda i: i in companiesToKeep, x)))



    return df.join(pd.DataFrame(mlb.fit_transform(df.pop('production_companies')),

                          columns=list(map(lambda x: 'prod_company_'+x,mlb.classes_)),

                          index=df.index))
df_raw = extractCommonProdCompanies(df_raw)
def extractCommonProdCountries(df):

    df['production_countries'] = df['production_countries'].apply(formatDictColumnAndExtractNames)



    countriesCount = df['production_countries'].apply(pd.Series).stack().value_counts()

    countriesToKeep = countriesCount[countriesCount > 10].keys()

    print("We'll keep the countries that appear more than 10 times:")

    print(countriesToKeep)



    df['production_countries'] = df['production_countries'].apply(lambda x: list(filter(lambda i: i in countriesToKeep, x)))

    return df.join(pd.DataFrame(mlb.fit_transform(df.pop('production_countries')),

                          columns=list(map(lambda x: 'prod_country_'+x,mlb.classes_)),

                          index=df.index))
df_raw = extractCommonProdCountries(df_raw)
def extractCommonSpokenLanguages(df):

    df['spoken_languages'] = df['spoken_languages'].apply(formatDictColumnAndExtractNames)



    languageCount = df['spoken_languages'].apply(pd.Series).stack().value_counts()

    languagesToKeep = languageCount[languageCount > 10].keys()

    print("We'll keep the languages that appear more than 10 times:")

    print(languagesToKeep)



    df['spoken_languages'] = df['spoken_languages'].apply(lambda x: list(filter(lambda i: i in languagesToKeep, x)))



    return df.join(pd.DataFrame(mlb.fit_transform(df.pop('spoken_languages')),

                          columns=list(map(lambda x: 'spoken_language_'+x,mlb.classes_)),

                          index=df.index))
df_raw = extractCommonSpokenLanguages(df_raw)
def extractCommonKeywords(df):

    df['Keywords'] = df['Keywords'].apply(formatDictColumnAndExtractNames)



    keywordCount = df['Keywords'].apply(pd.Series).stack().value_counts()

    keywordsToKeep = keywordCount[keywordCount >= 30].keys()

    print("We'll keep the keywords that appear more than 30 times:")

    print(keywordsToKeep)



    df['Keywords'] = df['Keywords'].apply(lambda x: list(filter(lambda i: i in keywordsToKeep, x)))



    return df.join(pd.DataFrame(mlb.fit_transform(df.pop('Keywords')),

                          columns=list(map(lambda x: 'keyword_'+x,mlb.classes_)),

                          index=df.index))
df_raw = extractCommonKeywords(df_raw)
# have a look at a cast cell

df_raw.iloc[3]['cast']
def addCastLengthColumn(df):

    castNames = df['cast'].apply(formatDictColumnAndExtractNames)

    df['cast_len'] = castNames.apply(lambda x: len(x))

    return df
df_raw = addCastLengthColumn(df_raw)
df_raw.drop(['cast'], axis=1, inplace=True)
# have a look at a crew cell

df_raw.iloc[113]['crew']
def formatDictColumnAndExtractJobName(strVal, job):

    listOfItems = convertStringToList(strVal)

    

    jobItem = (list(filter(lambda lst: lst['job'] == job, listOfItems)) or [None])[0]

    if type(jobItem) is dict:

        return jobItem['name']

    else:

        return None
def addCrewJobsColumns(df):

    df['director'] = df['crew'].apply(formatDictColumnAndExtractJobName, args=('Director',))

    df['screenplay'] = df['crew'].apply(formatDictColumnAndExtractJobName, args=('Screenplay',))

    df['director_of_photography'] = df['crew'].apply(formatDictColumnAndExtractJobName, args=('Director of Photography',))

    df['original_music_composer'] = df['crew'].apply(formatDictColumnAndExtractJobName, args=('Original Music Composer',))

    df['art_director'] = df['crew'].apply(formatDictColumnAndExtractJobName, args=('Art Direction',))

    

    return df
df_raw = addCrewJobsColumns(df_raw)
def formatDictColumnAndExtractDirectorGender(strVal):

    listOfItems = convertStringToList(strVal)



    directorItem = (list(filter(lambda lst: lst['job'] == 'Director', listOfItems)) or [None])[0]

    if type(directorItem) is dict:

        return directorItem['gender']

    else:

        return None
def addDirectorGenderColumn(df):

    df['director_gender'] = df['crew'].apply(formatDictColumnAndExtractDirectorGender)

    return df
df_raw = addDirectorGenderColumn(df_raw)
def addCrewLenghtColumn(df):

    df['crew'] = df['crew'].apply(formatDictColumnAndExtractNames)

    df['crew_len'] = df['crew'].apply(lambda x: len(x))

    return df
df_raw = addCrewLenghtColumn(df_raw)
# drop crew column

df_raw.drop(['crew'], axis=1, inplace=True)
df_raw['has_homepage'] = df_raw['homepage'].apply(lambda x: isinstance(x, str))
# look at a random cell

df_raw.iloc[8]['belongs_to_collection']
df_raw['belongs_to_collection'] = df_raw['belongs_to_collection'].apply(lambda x: isinstance(x, str))
def extractTaglineInfo(df):

    df['has_tagline'] = df['tagline'].apply(lambda x: isinstance(x, str))

    df['tagline_len'] = df['tagline'].apply(lambda x: len(x) if isinstance(x, str) else 0)

    return df
df_raw = extractTaglineInfo(df_raw)
# check the result

df_raw[['tagline', 'has_tagline', 'tagline_len']].head(8)
def extractOverviewInfo(df):

    df['has_overview'] = df['overview'].apply(lambda x: isinstance(x, str))

    df['overview_len'] = df['overview'].apply(lambda x: len(x) if isinstance(x, str) else 0)

    return df
df_raw = extractOverviewInfo(df_raw)
# check the result

df_raw[['overview', 'has_overview', 'overview_len']].head(8)
# we noticed quite a lot of movies with budget 0...

df_raw['has_budget'] = df_raw['budget'].apply(lambda x: x > 0)
toRemove = ['imdb_id', 'id', 'poster_path', 'overview', 'homepage', 'tagline', 'original_title', 'status']

df_raw.drop(toRemove, axis=1, inplace=True)
train_cats(df_raw)
df_trn, y_trn, nas = proc_df(df_raw, 'revenue')
m = RandomForestRegressor(n_jobs=-1)

m.fit(df_trn, y_trn)

m.score(df_trn,y_trn)
def split_vals(a,n): return a[:n], a[n:]

n_valid = 600 # 20%

n_trn = len(df_trn)-n_valid

X_train, X_valid = split_vals(df_trn, n_trn)

y_train, y_valid = split_vals(y_trn, n_trn)

raw_train, raw_valid = split_vals(df_raw, n_trn)
def rmse(x,y): return math.sqrt(((x-y)**2).mean())



# print_score function depending on the evaluation metric: rmse

def print_score(m):

    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),

                m.score(X_train, y_train), m.score(X_valid, y_valid)]

    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)

    print(res)
#rf with hyper parameters

m = RandomForestRegressor(n_estimators=40, min_samples_leaf=10, max_features=0.5, n_jobs=-1, oob_score=True)

m.fit(X_train, y_train)

print_score(m)
fi = rf_feat_importance(m, df_trn)

fi
fi.plot('cols', 'imp', figsize=(10,6), legend=False);
# function to plot feature importance

def plot_fi(fi): return fi.plot('cols', 'imp', 'barh', figsize=(12,12), legend=False)
# plot the top 50 features

plot_fi(fi[:50]);
# we'll keep the top ~30 features

to_keep = fi[fi.imp>0.002].cols; len(to_keep)
df_keep = df_trn[to_keep].copy()

X_train, X_valid = split_vals(df_keep, n_trn)
# new model with only the top ~30 features

m = RandomForestRegressor(n_estimators=40, min_samples_leaf=10, max_features=0.5,

                          n_jobs=-1, oob_score=True)

m.fit(X_train, y_train)

print_score(m)
# plot the new feature importance

fi = rf_feat_importance(m, df_keep)

plot_fi(fi);
from scipy.cluster import hierarchy as hc
# dendrogram plot

corr = np.round(scipy.stats.spearmanr(df_keep).correlation, 4)

corr_condensed = hc.distance.squareform(1-corr)

z = hc.linkage(corr_condensed, method='average')

fig = plt.figure(figsize=(16,10))

dendrogram = hc.dendrogram(z, labels=df_keep.columns, orientation='left', leaf_font_size=16)

plt.show()
# function to get the OOB score for a given datraframe (with the same hyperparameters as before)

def get_oob(df):

    m = RandomForestRegressor(n_estimators=40, min_samples_leaf=10, max_features=0.5, n_jobs=-1, oob_score=True)

    x, _ = split_vals(df, n_trn)

    m.fit(x, y_train)

    return m.oob_score_
# check the current OOB score

get_oob(df_keep)
for c in ('release_Month', 'release_Dayofyear', 'release_Week', 'release_Year', 'release_Elapsed'):

    print(c, get_oob(df_keep.drop(c, axis=1)))
to_drop = ['release_Month', 'release_Elapsed']

get_oob(df_keep.drop(to_drop, axis=1))
df_keep.drop(to_drop, axis=1, inplace=True)

X_train, X_valid = split_vals(df_keep, n_trn)

y_train, y_valid = split_vals(y_trn, n_trn)
m = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, max_features=0.5,

                          n_jobs=-1, oob_score=True)

m.fit(X_train, y_train)

print_score(m)
df_test = pd.read_csv(f'{PATH}test.csv', 

                     low_memory=False, 

                     parse_dates=["release_date"])
add_datepart(df_test, 'release_date')



df_test = extractGenres(df_test)

df_test = extractCommonProdCompanies(df_test)

df_test = extractCommonProdCountries(df_test)

df_test = extractCommonSpokenLanguages(df_test)

df_test = extractCommonKeywords(df_test)

df_test = addCastLengthColumn(df_test)

df_test.drop(['cast'], axis=1, inplace=True)

df_test = addCrewJobsColumns(df_test)

df_test = addDirectorGenderColumn(df_test)

df_test = addCrewLenghtColumn(df_test)

df_test.drop(['crew'], axis=1, inplace=True)

df_test['has_homepage'] = df_test['homepage'].apply(lambda x: isinstance(x, str))

df_test['belongs_to_collection'] = df_test['belongs_to_collection'].apply(lambda x: isinstance(x, str))

df_test = extractTaglineInfo(df_test)

df_test = extractOverviewInfo(df_test)

df_test['has_budget'] = df_test['budget'].apply(lambda x: x > 0)

df_test.drop(toRemove, axis=1, inplace=True)
# apply the same categories

apply_cats(df_test, df_raw)
# process the test dataframe

df_test,_,nas = proc_df(df_test,na_dict = nas)
# keep the most important features & some redundant

df_test_keep = df_test[to_keep].copy()

df_test_keep.drop(to_drop, axis=1, inplace=True)
# predict!

predictions = m.predict(df_test_keep)
# copy of the initial test set, to get the ids

df_test_raw = pd.read_csv(f'{PATH}test.csv', low_memory = False)
submission = pd.DataFrame({'id': df_test_raw['id'], 'revenue': np.expm1(predictions)})
submission
submission.to_csv('tmdb_predictions_kaggle.csv', index=False)