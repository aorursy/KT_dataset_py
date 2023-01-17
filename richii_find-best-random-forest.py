import numpy as np

import pandas as pd

import datetime



from multiprocessing import  Pool

from math import sqrt

from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso

from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

from nltk.stem.snowball import SnowballStemmer



stemmer = SnowballStemmer('english')
def str_stemmer(s):

	return " ".join([stemmer.stem(word) for word in s.lower().split()])



def str_common_word(str1, str2):

	return sum(int(str2.find(word)>=0) for word in str1.split())
def parallelize_dataframe(df, func, n_cores=4):

    df_split = np.array_split(df, n_cores)

    pool = Pool(n_cores)

    df = pd.concat(pool.map(func, df_split))

    pool.close()

    pool.join()

    return df
def stemming(df_orig):

    df = df_orig.copy()

    df['search_term'] = df['search_term'].map(lambda x:str_stemmer(x))

    df['product_title'] = df['product_title'].map(lambda x:str_stemmer(x))

    df['product_description'] = df['product_description'].map(lambda x:str_stemmer(x))

    return df
# return dataframe with features from attributes

def add_features_from_attributes(df_orig, df_attr, names):

    df = df_orig.copy()

    for name in names:

        att = df_attr[df_attr['name']==name]

        att = att.drop(['name'], axis=1)

        df = df.merge(att, on='product_uid', how='left').rename(columns={'value': name})

        df[name] = df[name].astype(str).map(lambda x:str_stemmer(x))

        df['search_term_in_'+name] = df['search_term']+"\t"+df[name]

        df['search_term_in_'+name] = df['search_term_in_'+name].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))    

    df = df.drop(names,axis=1)

    return df
def count_search_term(df_orig):

    df = df_orig.copy()

    df['len_of_query'] = df['search_term'].map(lambda x:len(x.split())).astype(np.int64)

    df['product_info'] = df['search_term']+"\t"+df['product_title']+"\t"+df['product_description']

    df['word_in_title'] = df['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))

    df['word_in_description'] = df['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))

    return df.drop(['search_term','product_title','product_description','product_info'],axis=1)
# split dataframe in train and test

def split_dataframe(df):

    return train_test_split(df, test_size=0.2, random_state=0)
# return data splitted in X and y

def prepare_data(train, test):

    y_train = train['relevance'].values

    X_train = train.drop(['id','relevance', 'product_uid'],axis=1).values

    

    y_test = test['relevance'].values

    X_test = test.drop(['id','relevance', 'product_uid'],axis=1).values

    return X_train, X_test, y_train, y_test
str_to_array_num = lambda x: [int(s) for s in x.split() if s.isdigit()]



def count_numbers(t):

    p, s = t

    c = 0

    for n in p:

        if n in s:

            c = c +1

    return c



def use_numbers(df_orig, att_orig):

    attr = att_orig.copy()

    df = df_orig.copy()

    # extract array of numbers in search term column

    df['nums'] = df['search_term'].map(str_to_array_num)

    

    # remove special character

    attr['value'] = attr['value'].astype(str).str.replace('°', '') 

    attr['value'] = attr['value'].astype(str).str.replace('⁰', '')

    

    # extract array of numbers in value column

    attr['value'] = attr['value'].map(str_to_array_num)

    

    # calculate the length of the array (usefull to filter useless attributes)

    attr['length'] = attr['value'].map(lambda x: len(x))

    # remove the attributes without numbers, group attributes on product_uid and create a array of numbers (contained in the attributes)

    attr = attr[attr['length']!=0].groupby('product_uid')['value'].apply(lambda x: list(set([item for sublist in x for item in sublist])))

    

    # merge dataframe of products with attributes

    df = df.merge(attr, on='product_uid', how='left')

    # check if value colmun is a valid array

    df['value'] = df['value'].apply(lambda d: d if isinstance(d, list) else [])

    # create a list of tuple where the first value is the array of numbers in the query and the second is 

    # the array of numbers in the attributes

    df['nums_in_value'] = list(zip(df['nums'],df['value']))

    # check how many numbers in the query are in the attributes

    df['nums_in_value'] = df['nums_in_value'].map(count_numbers)    

    return df.drop(['nums','value'],axis=1)
def tfidf(doc, to_analyze):

    #instantiate CountVectorizer()

    cv=CountVectorizer()

    # this steps generates word counts for the words in your docs

    word_count_vector=cv.fit_transform(doc)

    # compute the IDF values

    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)

    tfidf_transformer.fit(word_count_vector)

    # count matrix

    count_vector=cv.transform(to_analyze)

    # tf-idf scores

    tf_idf_vector=tfidf_transformer.transform(count_vector)

    feature_names = cv.get_feature_names()

    # create dataframe from sparse matrix

    # from_m = pd.DataFrame.sparse.from_spmatrix(tf_idf_vector)

    # from_m.columns = feature_names

    # return from_m.loc[:, (from_m != 0).any(axis=0)]

    result = []

    for line in tf_idf_vector:

        result.append(line.sum())

    return result



def create_df_with_tfidf(df_orig):

    df = df_orig.copy()

    df['tfidf_description'] = tfidf(df['product_description'], df['search_term'])

    df['tfidf_title'] = tfidf(df['product_title'], df['search_term'])

    return df
folder = '/kaggle/input/home-depot/'

df_train = pd.read_csv(folder + 'train.csv', encoding="ISO-8859-1")

df_test = pd.read_csv(folder + 'test.csv', encoding="ISO-8859-1")

df_attr = pd.read_csv(folder + 'attributes.csv')

df_pro_desc = pd.read_csv(folder + 'product_descriptions.csv')



num_train = df_train.shape[0]
# create dataframe

df_all = pd.merge(df_train, df_pro_desc, how='left', on='product_uid')
%%time

df_stem = parallelize_dataframe(df_all, stemming)
from sklearn.model_selection import RandomizedSearchCV



max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)



# Create the random grid

random_grid = {

    'n_estimators': [int(x) for x in np.linspace(start = 10, stop = 2000, num = 10)], # Number of trees in random forest

    'max_features': ['auto', 'sqrt'], # Number of features to consider at every split

    'max_depth': max_depth, # Maximum number of levels in tree

    'min_samples_split': [2, 5, 10], # Minimum number of samples required to split a node

    'min_samples_leaf': [1, 2, 4], # Minimum number of samples required at each leaf node

    'bootstrap': [True, False] # Method of selecting samples for training each tree

}
%%time

df_tfidf = create_df_with_tfidf(df_stem)

r, e = split_dataframe(parallelize_dataframe(df_tfidf, count_search_term))

X_train, X_test, y_train, y_test = prepare_data(r, e)
%%time

# Use the random grid to search for best hyperparameters

# First create the base model to tune

rf = RandomForestRegressor()

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 50, cv = 10, verbose=2, scoring='neg_mean_squared_error', random_state=42, n_jobs = -1)

# Fit the random search model

rf_random.fit(X_train, y_train)
%%time

rf_random.best_params_
def evaluate(model, test_features, test_labels):

    predictions = model.predict(test_features)

    errors = abs(predictions - test_labels)

    mape = 100 * np.mean(errors / test_labels)

    accuracy = 100 - mape

    mean = sqrt(mean_squared_error(predictions, test_labels))

    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))

    print('Accuracy = {:0.2f}%.'.format(accuracy))

    print('RMSE = {:0.4f}.'.format(mean))

    return mean

base_model = RandomForestRegressor(n_estimators= 500, min_samples_split= 5, min_samples_leaf= 10, max_features='log2', max_depth= 20, n_jobs=-1)

base_model.fit(X_train, y_train)

print('Base Model Performance')

base_accuracy = evaluate(base_model, X_test, y_test)



best_random = rf_random.best_estimator_

print('\nRandom Model Performance')

random_accuracy = evaluate(best_random, X_test, y_test)

print('Improvement of {:0.4f}.'.format( base_accuracy - random_accuracy ))

from IPython.display import Audio, display

display(Audio(url='https://sound.peal.io/ps/audios/000/000/537/original/woo_vu_luvub_dub_dub.wav', autoplay=True))