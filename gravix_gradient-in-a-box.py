import pandas as pd

pd.set_option('display.max_columns', 100)

pd.set_option('display.max_rows', 100)



import numpy as np



import os



import matplotlib.pyplot as plt

import seaborn as sns



import ast



import re



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder



import xgboost as xgb



from collections import Counter

print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.isna().sum()
test.isna().sum()
dict_columns = ['belongs_to_collection', 'genres', 'production_companies',

                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']



def text_to_dict(df):

    for column in dict_columns:

        df[column] = df[column].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x) )

    return df

        

train = text_to_dict(train)

test = text_to_dict(test)



def build_category_list(x, field, feature):

    regex = re.compile('[^0-9a-zA-Z_]')

    category_list = ""

    

    for d in x:

        new_category = regex.sub('', d[field].lower().replace(" ","_"))

        category_list += " " + new_category

    return category_list.strip()





target_fields = {'belongs_to_collection': 'name', 'genres': 'name',

                 'production_countries': 'iso_3166_1', 'production_companies': 'name',

                 'spoken_languages': 'iso_639_1', 'Keywords': 'name', 'cast':'name',

                 'crew':'name'

                }



train['crew_copy'] = train['crew']

test['crew_copy'] = test['crew']



train['cast_copy'] = train['cast']

test['cast_copy'] = test['cast']





for k,v in target_fields.items():

    print(k)

    train[k] = train[k].apply(lambda x: build_category_list(x, v, k))

    test[k] = test[k].apply(lambda x: build_category_list(x, v, k)) 

    

thresholds = {'belongs_to_collection': 0, 'genres': 0,

                 'production_countries': 10, 'production_companies': 10,

                 'spoken_languages': 10, 'Keywords': 10, 'cast': 10, 'crew': 10

                }



def streamline(x, kept):

    streamlined = ""

    for w in x.split(" "):

        if w in kept:

            streamlined = streamlined + " " + w

    return streamlined.strip()



for k,v in thresholds.items():

    print(k)

    c = Counter(" ".join(train[k]).split(" "))

    print("Initial:", len(c))

    kept = [w for w,nb in c.items() if nb > v]

    print("Kept:", len(kept))

    print("")

    train[k] = train[k].apply(lambda x: streamline(x, kept))

    test[k] = test[k].apply(lambda x: streamline(x, kept))
def build_category_list_with_roles(x, v, rv):

    regex = re.compile('[^0-9a-zA-Z_]')

    category_list = ""

    for d in x:

        if d[v['role_field']] != rv:

            pass

        else:

            if category_list == "":

                new_category = regex.sub('', d[v['field']].lower().replace(" ","_"))

                category_list += " " + new_category

    return category_list.strip()  

    

target_fields = {'cast_copy':{'field':'name', 'role_field':'order', 'role_values':[0,1,2,3,4,5]}, 

                 'crew_copy':{'field': 'name', 'role_field': 'job',

                         'role_values':['Director', 'Producer',

                                        'Executive Producer', 'Writer', 'First Assistant Director',

                                        'Associate Producer', 'Director of Photography'

                                       ]

                        }

                }





additional_label_encoding_columns = []



for k,v in target_fields.items():

    print(k)

    for rv in v['role_values']:

        striped_rv = str(rv).lower().replace(' ','_')

        additional_label_encoding_columns.append(k + '_' + striped_rv)

        train[k + '_' + striped_rv] = train[k].apply(lambda x: build_category_list_with_roles(x, v, rv))

        test[k + '_' + striped_rv] = test[k].apply(lambda x: build_category_list_with_roles(x, v, rv))

    

fillna_columns = {'release_date':'mode',

                  'status':'mode',

                  'belongs_to_collection': 'none',

                  'runtime': 'mode'}



for k,v in fillna_columns.items():

    if v == 'mode':

        fill = train[k].mode()[0]

    else:

        fill = v

    print(k, ': ', fill)

    train[k] = train[k].fillna(value = fill)

    test[k] = test[k].fillna(value = fill)

def extract_nb_within_collection(r):

    regex = re.compile('[^0-9a-zA-Z_]')

    original_title = regex.sub('', r['original_title'].lower().replace(" ","_"))

    

    if r['is_part_of_collection'] == 0:

        return 0

    else:

        if (r['belongs_to_collection'] == original_title + '_collection') or (r['belongs_to_collection'] == original_title):

            return 1

        else:

            regex = re.compile('[^0-9]')

            probable_number = regex.sub('', r['original_title'])

            if probable_number == '' or int(probable_number) > 5:

                return 0

            else:

                return probable_number



def feature_addition(df):

    

    df['release_year'] = df.release_date.apply(lambda x: x[-2:]).astype('int')

    df['release_month'] = df.release_date.apply(lambda x: x.split('/')[0]).astype('int')

    df['release_quarter'] = df.release_month % 4 + 1

    

    df['budget'] = df.budget / 1000000

    

    df['nb_spoken_languages'] = df.spoken_languages.apply(lambda r: len(r.split(' ')))

    df['nb_words_overview'] = df.overview.apply(lambda x: len(str(x).split(' ')) )

    df['nb_production_companies'] = df.production_companies.apply(lambda x: len(x.split(' ')) )

    df['nb_production_countries'] = df.production_countries.apply(lambda x: len(x.split(' ')) )

    df['nb_cast'] = df.cast.apply(lambda x: len(x.split(' ')) )

    df['nb_crew'] = df.crew.apply(lambda x: len(x.split(' ')) )

    df['nb_keywords'] = df.Keywords.apply(lambda x: len(x.split(' ')) )

    df['nb_words_title'] = df.title.apply(lambda x: len(str(x).split(' ')) )

    df['nb_words_tagline'] = df.tagline.apply(lambda x: len(str(x).split(' ')) )

    

    df['nb_words_original_title'] = df.original_title.apply(lambda x: len(x.split(' ')) )

    

    df['has_original_title'] = (df.title == df.original_title).astype('int')



    df['has_homepage'] = 1 - df.homepage.isna().astype('int')

    df['homepage_base'] = df.homepage.apply(lambda x: str(x).split('//')[-1].split('/')[0].split('www.')[-1].split('.')[0])

    df['homepage_extension'] = df.homepage.apply(lambda x: str(x).split('//')[-1].split('/')[0].split('www.')[-1].split('.')[-1]).fillna(value = '')



    df['is_part_of_collection'] = 1 - (df.belongs_to_collection == '').astype('int')

    df['nb_within_collection'] =  df.apply(lambda r: extract_nb_within_collection(r), axis = 1).astype('int')

    

    return df

                                                

train = feature_addition(train)

test = feature_addition(test)
columns_to_categorize = ['belongs_to_collection', 'status', 'original_language', 'homepage_base', 'homepage_extension']

columns_to_categorize += additional_label_encoding_columns



for c in columns_to_categorize:

    print(c)

    le = LabelEncoder()

    le.fit_transform(train[c])

    test[c] = test[c].map(lambda s: 'unknown' if s not in le.classes_ else s)

    le.classes_ = np.append(le.classes_, 'unknown')

    train[c] = le.transform(train[c])

    test[c] = le.transform(test[c])

submission = pd.DataFrame(test['id'])



removed_columns = ['id', 'homepage', 'imdb_id', 'original_title', 'spoken_languages',

                   'overview', 'poster_path', 'tagline', 'title',

                  'release_date', 'crew_copy', 'cast_copy']





train.drop(removed_columns, axis = 1, inplace = True)

test.drop(removed_columns, axis = 1, inplace = True)
features_to_vectorize = ['genres', 'production_countries', 'production_companies', 'Keywords', 'cast', 'crew']





for f in features_to_vectorize:

    print(f)

    vectorizer = TfidfVectorizer(use_idf = False)

    vectorized_features = vectorizer.fit_transform(train[f])

    vectorized_features_names = [f + '_' + v for v in vectorizer.get_feature_names()]



    vectorized_features_sparse = pd.SparseDataFrame([ pd.SparseSeries(vectorized_features[i].toarray().ravel()) 

                              for i in np.arange(vectorized_features.shape[0]) ], columns = vectorized_features_names)



    train = pd.concat([train, vectorized_features_sparse], axis = 1)

    

    vectorized_features = vectorizer.transform(test[f])

    vectorized_features_sparse = pd.SparseDataFrame([ pd.SparseSeries(vectorized_features[i].toarray().ravel()) 

                              for i in np.arange(vectorized_features.shape[0]) ], columns = vectorized_features_names)



    test = pd.concat([test, vectorized_features_sparse], axis = 1)

    

    train.drop(f, inplace = True, axis = 1)

    test.drop(f, inplace = True, axis = 1)

    
train['revenue'] = np.log1p(train['revenue'] )
target_column = 'revenue'



train_set, validate_set = train_test_split(train, test_size = 0.2, random_state = 1)



x_train = train_set.drop([target_column], axis = 1).copy()

y_train = train_set[target_column].copy()



x_validate = validate_set.drop([target_column], axis = 1).copy()

y_validate = validate_set[target_column].copy()



x_total = train.drop([target_column], axis = 1).copy()

y_total = train[target_column].copy()



x_test = test.copy()
import lightgbm as lgb



params_lgb = {'drop_rate': [0.09777484320779173], 'feature_fraction': [0.6087324102659581],

              'lambda_l1': [0.03915143495854047], 'lambda_l2': [26.68081917087524],

              'learning_rate': [0.013231541159028165],

              'max_drop': [67.0], 'min_data_in_leaf': [1.0],

              'num_leaves': [32.0], 'num_trees': [1370.0]}



params_lgb = {k:v[0] for k,v in params_lgb.items()}





lg = lgb.LGBMRegressor(

                        objective = 'regression',

                        metric = 'rmse',

                        early_stopping_round = 50,

                        drop_rate = params_lgb['drop_rate'],

                        feature_fraction = params_lgb['feature_fraction'],

                        lambda_l1 = params_lgb['lambda_l1'],

                        lambda_l2 = params_lgb['lambda_l2'],

                        learning_rate = params_lgb['learning_rate'],

                        max_drop = int(params_lgb['max_drop']),

                        min_data_in_leaf = int(params_lgb['min_data_in_leaf']),

                        num_leaves = int(params_lgb['num_leaves']),

                        num_trees = int(params_lgb['num_trees']))



lg.fit(x_train, y_train.values, eval_set=[(x_train, y_train), (x_validate, y_validate)])
feature_importance = pd.DataFrame(lg.feature_importances_, columns = ['importance'])

feature_importance['feature'] = train.columns[:-1]

feature_importance.sort_values(by='importance', inplace = True, ascending = False)

feature_importance.reset_index(drop = True, inplace = True)

feature_importance
fig, ax = plt.subplots(figsize=(10, 15))

sns.barplot(y = 'feature', x = 'importance', data = feature_importance[0:20])
lg = lgb.LGBMRegressor(

                        objective = 'regression',

                        metric = 'rmse',

                        early_stopping_round = 50,

                        drop_rate = params_lgb['drop_rate'],

                        feature_fraction = params_lgb['feature_fraction'],

                        lambda_l1 = params_lgb['lambda_l1'],

                        lambda_l2 = params_lgb['lambda_l2'],

                        learning_rate = params_lgb['learning_rate'],

                        max_drop = int(params_lgb['max_drop']),

                        min_data_in_leaf = int(params_lgb['min_data_in_leaf']),

                        num_leaves = int(params_lgb['num_leaves']),

                        num_trees = 592)



lg.fit(x_total, y_total.values, eval_set=[(x_train, y_train), (x_validate, y_validate)])
y_test_p = pd.Series(lg.predict(x_test))

submission['revenue'] = np.expm1(y_test_p)

submission.to_csv("submission.csv", index = False)