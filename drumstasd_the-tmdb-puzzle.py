import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker



from collections import Counter

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor

from sklearn.model_selection import cross_val_score, train_test_split



from xgboost.sklearn import XGBRegressor

from nltk.sentiment.vader import SentimentIntensityAnalyzer



from skopt.space import Real, Integer

from skopt.utils import use_named_args

from skopt import gp_minimize, BayesSearchCV



import shap

import eli5

from eli5.sklearn import PermutationImportance
df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
df.head()
# Aa function for encoding NaN's to zeros and other values to ones of a selected feature

def binary_encode(feature):

    encoder = lambda x: 0 if pd.isna(x) else 1

    df[feature] = df[feature].apply(encoder)

    test_df[feature] = test_df[feature].apply(encoder)
# Convert homepage and belongs_to_collection to binary values

binary_encode('homepage')

binary_encode('belongs_to_collection')
# Create a binary feature showing if original language is english

english_identifier = lambda lang: 1 if lang == 'en' else 0



df['english_original'] = df['original_language'].apply(english_identifier)

test_df['english_original'] = test_df['original_language'].apply(english_identifier)
# A set of functions that will be used for counting the most frequent companies, cast and genres that appear in movies

extract_names = lambda companies: [company['name'] for company in eval(companies)] if pd.notna(companies) else []

top_x = lambda top_count, list_of_items: [item[0] for item in Counter([i for j in list_of_items for i in j]).most_common(top_count)]



def count_top(x, top):

    counter = 0

    for name in x:

        if name in top:

            counter += 1

    return counter
# Count number of top 20 production_companies in a movie

top_10_companies = top_x(20, df['production_companies'].apply(extract_names).values)



df['num_top_companies'] = df['production_companies'].apply(extract_names).apply(count_top, args=(top_10_companies,))

test_df['num_top_companies'] = test_df['production_companies'].apply(extract_names).apply(count_top, args=(top_10_companies,))
# Count number of top 5 genres in a movie

top_5_genres = top_x(5, df['genres'].apply(extract_names).values)



df['num_top_genres'] = df['genres'].apply(extract_names).apply(count_top, args=(top_5_genres,))

test_df['num_top_genres'] = test_df['genres'].apply(extract_names).apply(count_top, args=(top_5_genres,))
# Count number of top 50 cast in a movie

top_50_cast = top_x(50, df['cast'].apply(extract_names).values)



df['num_top_cast'] = df['cast'].apply(extract_names).apply(count_top, args=(top_50_cast,))

test_df['num_top_cast'] = test_df['cast'].apply(extract_names).apply(count_top, args=(top_50_cast,))
# Count number of items in a list and make the result a feature

def count_num(feature):

    name = 'num_' + feature

    counter = lambda x: np.nan if pd.isna(x) else len(eval(x))

    df[name] = df[feature].apply(counter)

    test_df[name] = test_df[feature].apply(counter)
# Count number of genres, spoken_languages, production_companies, production_countries, cast, crew, Keywords

# and create new features from them

for feature in ['genres', 'spoken_languages', 'production_companies', 'production_countries', 'cast', 'crew', 'Keywords']:

    count_num(feature)
# Some processed features had NaN values, fill them with median

for feature in ['num_genres', 'num_spoken_languages', 'num_production_companies', 'num_production_countries', 'num_cast', 'num_crew', 'num_Keywords']:

    df[feature] = df[feature].fillna(df[feature].median())

    test_df[feature] = test_df[feature].fillna(test_df[feature].median())
# Noticed that 'runtime' has two nulls and 12 zero values, let's fill them with median

df.loc[df['runtime'] == 0, 'runtime'] = df['runtime'].median()

test_df.loc[test_df['runtime'] == 0, 'runtime'] = test_df['runtime'].median()



df['runtime'] = df['runtime'].fillna(df['runtime'].median())

test_df['runtime'] = test_df['runtime'].fillna(test_df['runtime'].median())
# Fix null release_date in test dataset

test_df.loc[test_df['release_date'].isnull() == True, 'release_date'] = '12/8/01'
# Since years in release dates are represented as 19 and 20, this will change them to 19xx and 20xx respectively

def fix_year(date):

    year = date.split('/')[2]

    if int(year) < 19:

        return date[:-2] + "20" + year

    else:

        return date[:-2] + "19" + year
df['release_date'] = df['release_date'].apply(lambda x: fix_year(x))

test_df['release_date'] = test_df['release_date'].apply(lambda x: fix_year(x))



df['release_date'] = pd.to_datetime(df['release_date'])

test_df['release_date'] = pd.to_datetime(test_df['release_date'])
# Make separate features from month, day, weekday and year of release dates

df['year'] = df['release_date'].apply(lambda x: x.year)

test_df['year'] = test_df['release_date'].apply(lambda x: x.year)



df['month'] = df['release_date'].apply(lambda x: x.month)

test_df['month'] = test_df['release_date'].apply(lambda x: x.month)



df['day'] = df['release_date'].apply(lambda x: x.day)

test_df['day'] = test_df['release_date'].apply(lambda x: x.day)
# Create features for positivity, neurality and negativity of the movie overview using VADER sentiment analyser

analyser = SentimentIntensityAnalyzer()



df['overview_positivity'] = df['overview'].apply(lambda sentence: analyser.polarity_scores(str(sentence))['pos'])

test_df['overview_positivity'] = test_df['overview'].apply(lambda sentence: analyser.polarity_scores(str(sentence))['pos'])



df['overview_neutrality'] = df['overview'].apply(lambda sentence: analyser.polarity_scores(str(sentence))['neu'])

test_df['overview_neutrality'] = test_df['overview'].apply(lambda sentence: analyser.polarity_scores(str(sentence))['neu'])



df['overview_negativity'] = df['overview'].apply(lambda sentence: analyser.polarity_scores(str(sentence))['neg'])

test_df['overview_negativity'] = test_df['overview'].apply(lambda sentence: analyser.polarity_scores(str(sentence))['neg'])
# Drop columns that are not gonna be used, but save test id

test_id = test_df['id']



columns = ['id', 'imdb_id', 'poster_path', 'genres', 'original_language', 'original_title', 'overview', 'production_companies', 

           'production_countries', 'release_date','spoken_languages', 'status', 'tagline', 'title', 'Keywords', 'cast', 'crew']



df.drop(columns, axis=1, inplace=True)

test_df.drop(columns, axis=1, inplace=True)
df.head()
# Revenue for movies belonging to collection vs movies that don't belong to a collection

sns.catplot(x='belongs_to_collection', y='revenue', data=df)
# Revenue for movies with a homepage vs movies without one

sns.catplot(x='homepage', y='revenue', data=df)
sns.catplot(x='english_original', y='revenue', data=df)
plot = sns.catplot(x='num_Keywords', y='revenue', data=df)

plot.set_xticklabels(step=3, rotation=30)
plot = sns.catplot(x='num_crew', y='revenue', data=df)

plot.set_xticklabels(step=15, rotation=30)
plot = sns.catplot(x='num_cast', y='revenue', data=df)

plot.set_xticklabels(step=10, rotation=30)
plot = sns.catplot(x='num_production_companies', y='revenue', data=df)

plot.set_xticklabels(step=2, rotation=30)
sns.catplot(x='num_production_countries', y='revenue', data=df)
sns.catplot(x='num_spoken_languages', y='revenue', data=df)
sns.catplot(x='num_genres', y='revenue', data=df)
plot = sns.catplot(x='year', y='revenue', data=df)

plot.set_xticklabels(step=10, rotation=30)
sns.catplot(x='month', y='revenue', data=df)
plot = sns.catplot(x='day', y='revenue', data=df)

plot.set_xticklabels(step=2, rotation=30)
sns.heatmap(df.corr(), xticklabels=list(df), yticklabels=list(df))
plt.scatter(df['runtime'], df['revenue'])
plt.scatter(df['popularity'], df['revenue'])
plt.scatter(df['budget'], df['revenue'])
# Create separate variables for models to use as features and labels when fitting

labels = df['revenue']

df.drop('revenue', axis=1, inplace=True)
def score_model(model):

    scores = cross_val_score(model, df, labels, scoring="neg_mean_squared_error", cv=10)

    return np.sqrt(-scores).mean().round()
# Print mean error using cross validation

print('XGB: ', score_model(XGBRegressor()))

print('Linear: ', score_model(LinearRegression()))

print('Decision Tree: ', score_model(DecisionTreeRegressor()))

print('SVR auto: ', score_model(SVR(gamma='auto')))

print('Random Forest: ', score_model(RandomForestRegressor(n_estimators=10)))

print('Gradient Boosting: ', score_model(GradientBoostingRegressor()))

print('Ada Boost: ', score_model(AdaBoostRegressor()))
features_train, features_test, labels_train, labels_test = train_test_split(df, labels, random_state=42)
def print_scores(model):

    print("validation score: %s" % np.sqrt(-model.best_score_))

    print("test score: %s" % np.sqrt(-model.score(features_test, labels_test)))
params = {

    'learning_rate': (10**-5, 10**0),

    'n_estimators': (100, 1000),

    'max_features': (1, len(list(df))),

    'min_samples_split': (2, 100),

    'subsample': (0.1, 1)

}



gradient = BayesSearchCV(GradientBoostingRegressor(), params, n_iter=32, cv=5, scoring="neg_mean_squared_error")



gradient.fit(features_train, labels_train)



print('Gradient Boost Scores')

print_scores(gradient)
params = {

    'learning_rate': (10**-5, 10**0),

    'n_estimators': (100, 1000),

    'max_features': (1, len(list(df))),

    'min_samples_split': (2, 100),

    'subsample': (0.1, 1)

}



xgb = BayesSearchCV(XGBRegressor(), params, n_iter=32, cv=5, scoring="neg_mean_squared_error")



xgb.fit(features_train, labels_train)
print('XGB Scores')

print_scores(xgb)
params = {

    'n_estimators': (10, 1000),

    'max_features': ['auto', 'sqrt', 'log2'],

    'min_samples_split': (2, 100),

    'min_samples_leaf': (1, 10),

    'min_impurity_decrease': (0.0, 1.0)

}



forest = BayesSearchCV(RandomForestRegressor(), params, n_iter=32, cv=5, scoring="neg_mean_squared_error")



forest.fit(features_train, labels_train)



print('Random Forest Scores')

print_scores(forest)

print("Best params:")

print(forest.best_params_)
reg = RandomForestRegressor(**forest.best_params_, random_state=42)

reg.fit(df, labels)
perm = PermutationImportance(reg, random_state=42).fit(features_test, labels_test)

eli5.show_weights(perm, feature_names = features_test.columns.tolist())
data_for_prediction = test_df.iloc[1998]



explainer = shap.TreeExplainer(reg)

shap_values = explainer.shap_values(data_for_prediction)



shap.initjs()

shap.force_plot(explainer.expected_value, shap_values, data_for_prediction)
pred = reg.predict(test_df)

pd.DataFrame({'id': test_id, 'revenue': pred}).to_csv('submissions.csv', index=False)