import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import LabelEncoder

from collections import Counter

from sklearn.model_selection import train_test_split

import lightgbm as lgb
train = pd.read_csv('../input/tmdb-box-office-prediction/train.csv')

test = pd.read_csv('../input/tmdb-box-office-prediction/test.csv')

sample_submission = pd.read_csv('../input/tmdb-box-office-prediction/sample_submission.csv')
train.head()
test.head()
print("Number of rows(train): "+str(len(train)))

print("Number of rows(test): "+str(len(test)))
train.info()
train['has_collection'] = train['belongs_to_collection'].apply(lambda x: 1 if str(x) != 'nan' else 0)

train['collection_id'] = train['belongs_to_collection'].apply(lambda x: eval(x)[0]['id'] if str(x) != 'nan' else 0)



test['has_collection'] = test['belongs_to_collection'].apply(lambda x: 1 if str(x) != 'nan' else 0)

test['collection_id'] = test['belongs_to_collection'].apply(lambda x: eval(x)[0]['id'] if str(x) != 'nan' else 0)
train = train.drop(['belongs_to_collection'], axis=1)

test = test.drop(['belongs_to_collection'], axis=1)
list_of_genres = list(train['genres'].apply(lambda x: [i['name'] for i in eval(x)] if str(x) != 'nan' else []).values)
Counter([i for j in list_of_genres for i in j]).most_common(15)
top_genres = [m[0] for m in Counter([i for j in list_of_genres for i in j]).most_common(15)]



train['all_genres'] = train['genres'].apply(lambda x: ' '.join(sorted([i['name'] for i in eval(x)])) if (isinstance(x,int) or isinstance(x,str)) == True else '')

for gen in top_genres:

    train['genre_' + gen] = train['all_genres'].apply(lambda x: 1 if gen in x else 0)

    

test['all_genres'] = test['genres'].apply(lambda x: ' '.join(sorted([i['name'] for i in eval(x)])) if (isinstance(x,int) or isinstance(x,str)) == True else '')

for gen in top_genres:

    test['genre_' + gen] = test['all_genres'].apply(lambda x: 1 if gen in x else 0)

    

train = train.drop(['genres'], axis=1)

test = test.drop(['genres'], axis=1)
lang_encoder = LabelEncoder()

train['all_genres'] = lang_encoder.fit_transform(train['all_genres'])

test['all_genres'] = lang_encoder.fit_transform(test['all_genres'])
train['original_language'].unique()
lang_encoder = LabelEncoder()

train['original_language'] = lang_encoder.fit_transform(train['original_language'])

test['original_language'] = lang_encoder.fit_transform(test['original_language'])
prod_companies = list(train['production_companies'].apply(lambda x: [i['name'] for i in eval(x)] if str(x) != 'nan' else '').values)
train['prod_companies_count'] = train['production_companies'].apply(lambda x: len([i for i in eval(x)]) if str(x) != 'nan' else 0)

test['prod_companies_count'] = test['production_companies'].apply(lambda x: len([i for i in eval(x)]) if str(x) != 'nan' else 0)
pop_production = Counter([i for j in prod_companies for i in j])
train['production_score'] = train['production_companies'].apply(lambda x: np.tanh(max([pop_production[i['name']] for i in eval(x)])) if str(x) != 'nan' else 0)

test['production_score'] = test['production_companies'].apply(lambda x: np.tanh(max([pop_production[i['name']] for i in eval(x)])) if str(x) != 'nan' else 0)
train = train.drop(['production_companies'], axis=1)

test = test.drop(['production_companies'], axis=1)
train['production_countries'] = train['production_countries'].apply(lambda x: [i['name'] for i in eval(x)][0] if str(x) != 'nan' else '')

test['production_countries'] = test['production_countries'].apply(lambda x: [i['name'] for i in eval(x)][0] if str(x) != 'nan' else '')
prod_country_encoder = LabelEncoder()

train['production_countries'] = prod_country_encoder.fit_transform(train['production_countries'])

test['production_countries'] = prod_country_encoder.fit_transform(test['production_countries'])
train['production_countries'].head()
train['release_date'] = train['release_date'].apply(lambda x: pd.to_datetime(x))

test['release_date'] = test['release_date'].apply(lambda x: pd.to_datetime(x))
train['year'] = train['release_date'].apply(lambda x: x.year)

train['month'] = train['release_date'].apply(lambda x: x.month)

train['day_of_week'] = train['release_date'].apply(lambda x: x.weekday())



test['year'] = test['release_date'].apply(lambda x: x.year)

test['month'] = test['release_date'].apply(lambda x: x.month)

test['day_of_week'] = test['release_date'].apply(lambda x: x.weekday())
train = train.drop(['release_date'], axis=1)

test = test.drop(['release_date'], axis=1)
_ = sns.lineplot(x=train['year'], y=train['revenue'], color='r')
train['year'] = train['year'].apply(lambda x: x-100 if x>2020 else x)

test['year'] = test['year'].apply(lambda x: x-100 if x>2020 else x)
_ = sns.lineplot(x=train['year'], y=train['revenue'], color='g')
avg_runtime_train = train['runtime'].mean()

train['runtime'] = train['runtime'].apply(lambda x: x if str(x) != 'nan' else avg_runtime_train)



avg_runtime_test = test['runtime'].mean()

test['runtime'] = test['runtime'].apply(lambda x: x if str(x) != 'nan' else avg_runtime_test)
_ = sns.distplot(train['runtime'])
train[train['budget']==0].head() 
train['spoken_languages_count'] = train['spoken_languages'].apply(lambda x: len(eval(x)) if str(x) != 'nan' else 0)

test['spoken_languages_count'] = test['spoken_languages'].apply(lambda x: len(eval(x)) if str(x) != 'nan' else 0)
_ = sns.countplot(x=train['spoken_languages_count'])
train = train.drop(['spoken_languages'], axis=1)

test = test.drop(['spoken_languages'], axis=1)
train['Keywords']
list_of_keywords = list(train['Keywords'].apply(lambda x: [i['name'] for i in eval(x)] if str(x) != 'nan' else []).values)
train['num_Keywords'] = train['Keywords'].apply(lambda x: len(eval(x)) if str(x) != 'nan' else 0)

train['all_Keywords'] = train['Keywords'].apply(lambda x: ' '.join(sorted([i['name'] for i in eval(x)])) if str(x) != 'nan' else '')



top_keywords = [m[0] for m in Counter([i for j in list_of_keywords for i in j]).most_common(30)]



for g in top_keywords:

    train['keyword_'+g] = train['all_Keywords'].apply(lambda x: 1 if g in x else 0)

    

    

    

test['num_Keywords'] = test['Keywords'].apply(lambda x: len(eval(x)) if str(x) != 'nan' else 0)

test['all_Keywords'] = test['Keywords'].apply(lambda x: ' '.join(sorted([i['name'] for i in eval(x)])) if str(x) != 'nan' else '')



top_keywords = [m[0] for m in Counter([i for j in list_of_keywords for i in j]).most_common(30)]



for g in top_keywords:

    test['keyword_'+g] = test['all_Keywords'].apply(lambda x: 1 if g in x else 0)
keywords_encoder = LabelEncoder()

train['all_Keywords'] = keywords_encoder.fit_transform(train['all_Keywords'])

test['all_Keywords'] = keywords_encoder.fit_transform(test['all_Keywords'])
train = train.drop(['Keywords'], axis=1)

test = test.drop(['Keywords'], axis=1)
train['cast'] = train['cast'].apply(lambda x: eval(x) if str(x) != 'nan' else [])

test['cast'] = test['cast'].apply(lambda x: eval(x) if str(x) != 'nan' else [])
train['gender_0_count'] = train['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))

train['gender_1_count'] = train['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))

train['gender_2_count'] = train['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))



test['gender_0_count'] = test['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))

test['gender_1_count'] = test['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))

test['gender_2_count'] = test['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))
list_of_cast_members = list(train['cast'].apply(lambda x: [i['name'] for i in x] if str(x) != 'nan' else []).values)
top_cast_members = [m[0] for m in Counter([i for j in list_of_cast_members for i in j]).most_common(50)]
for g in top_cast_members:

    train['cast_member_'+g] = train['cast'].apply(lambda x: 1 if g in str(x) else 0)

    

for g in top_cast_members:

    test['cast_member_'+g] = test['cast'].apply(lambda x: 1 if g in str(x) else 0)
train = train.drop(['cast'], axis=1)

test = test.drop(['cast'], axis=1)
train['crew'] = train['crew'].apply(lambda x: eval(x) if str(x) != 'nan' else [])

test['crew'] = test['crew'].apply(lambda x: eval(x) if str(x) != 'nan' else [])
list_of_crew_members = list(train['crew'].apply(lambda x: [i['name'] for i in x] if str(x) != 'nan' else []).values)
top_crew_members = [m[0] for m in Counter([i for j in list_of_crew_members for i in j]).most_common(50)]
for g in top_crew_members:

    train['crew_member_'+g] = train['crew'].apply(lambda x: 1 if g in str(x) else 0)

    

for g in top_crew_members:

    test['crew_member_'+g] = test['crew'].apply(lambda x: 1 if g in str(x) else 0)
train = train.drop(['crew'], axis=1)

test = test.drop(['crew'], axis=1)
train['has_homepage'] = train['homepage'].apply(lambda x: 1 if str(x) != 'nan' else 0)

test['has_homepage'] = test['homepage'].apply(lambda x: 1 if str(x) != 'nan' else 0)
train = train.drop(['homepage'], axis=1)

test = test.drop(['homepage'], axis=1)
train = train.drop(['poster_path'], axis=1)

test = test.drop(['poster_path'], axis=1)
train = train.drop(['status'], axis=1)

test = test.drop(['status'], axis=1)
for col in ['title', 'tagline', 'overview', 'original_title']:

    train['len_' + col] = train[col].fillna('').apply(lambda x: len(str(x)))

    train['words_' + col] = train[col].fillna('').apply(lambda x: len(str(x.split(' '))))

    

    test['len_' + col] = test[col].fillna('').apply(lambda x: len(str(x)))

    test['words_' + col] = test[col].fillna('').apply(lambda x: len(str(x.split(' '))))
train = train.drop(["imdb_id", "original_title", "overview", "tagline", "title"], axis=1)

test = test.drop(["imdb_id", "original_title", "overview", "tagline", "title"], axis=1)
X = train.drop(['id', 'revenue'], axis=1)

Y = np.log1p(train['revenue'])

X_test = test.drop(['id'], axis=1)
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.1)
params = {'num_leaves': 30,

         'min_data_in_leaf': 20,

         'objective': 'regression',

         'max_depth': 5,

         'learning_rate': 0.01,

         "boosting": "gbdt",

         "feature_fraction": 0.9,

         "bagging_freq": 1,

         "bagging_fraction": 0.9,

         "bagging_seed": 11,

         "metric": 'rmse',

         "lambda_l1": 0.2,

         "verbosity": -1}

model = lgb.LGBMRegressor(**params, n_estimators = 20000, nthread = 4, n_jobs = -1)

model.fit(X_train, Y_train, 

        eval_set=[(X_train, Y_train), (X_valid, Y_valid)], eval_metric='rmse',

        verbose=1000, early_stopping_rounds=200)
y_pred_valid = model.predict(X_valid)

y_pred = model.predict(X_test, num_iteration=model.best_iteration_)
sample_submission['revenue'] = np.expm1(y_pred)

sample_submission.to_csv("submission.csv", index=False)