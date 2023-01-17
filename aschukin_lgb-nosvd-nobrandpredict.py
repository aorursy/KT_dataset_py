import os
import re
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction import DictVectorizer
import lightgbm as lgb
#from sklearn.metrics import make_scorer, accuracy_score, mean_squared_log_error, mean_squared_error
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import SGDClassifier, SGDRegressor
import pickle

import gensim
import gensim.corpora as corpora
import nltk
#from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
#nltk.download('wordnet')
#nltk.download('stopwords')
lemma = WordNetLemmatizer()
def clean(text):
    # Remove new line characters
    text = re.sub('\s+', ' ', text)
    # Remove distracting single quotes
    text = re.sub("\'", "", text)
    return text


def doc_to_words(doc, lemma):
    words = [w for w in gensim.utils.simple_preprocess(str(doc), deacc=True)]
    words = [lemma.lemmatize(w) for w in words]
    
    return words
def get_dict_with_split(text_arr):
    clean_text_arr = [clean(text) for text in text_arr]
    words = [doc_to_words(text, lemma) for text in clean_text_arr]
    words_dict = [{w: 1 for w in ww} for ww in words]
    return words_dict
def get_dict_without_split(text_arr):
    words_dict = [{w: 1} for w in text_arr]
    return words_dict
def df_drop_price(df):
    try:
        df = df.drop(['price'], axis=1)
    except KeyError:
        pass
    return df
def df_fillna(df):
    df['category_name'] = df['category_name'].fillna('NAN')
    df['brand_name'] = df['brand_name'].fillna('NAN')
    df['item_description'] = df['item_description'].fillna('NAN')
    return df
def df_append_cats(df):
    df['category_level1'] = df['category_name']
    df['category_level2'] = df['category_name']
    df['category_level3'] = df['category_name']
    return df
def split_cats(arr):
    for i in range(len(arr)):
        try:
            s = arr[i,3].split('/')
            arr[i,7] = s[0]
            arr[i,8] = s[1]
            arr[i,9] = s[2]
        except Exception:
            arr[i,7] = 'NAN'
            arr[i,8] = 'NAN'
            arr[i,9] = 'NAN'
    return arr
def fill_na_descriptions(arr):
    arr[np.where(arr[:,6]=='No description yet')[0],6] = arr[np.where(arr[:,6]=='No description yet')[0],1]
    arr[np.where(arr[:,6]=='NAN')[0],6] = arr[np.where(arr[:,6]=='NAN')[0],1]
    return arr
def get_X_from_df(df):
    df = df_drop_price(df)
    df = df_fillna(df)
    df = df_append_cats(df)
    
    X = np.array(df)
    X = split_cats(X)
    X = fill_na_descriptions(X)
    return X
INPUT_PATH = r'../input'
df_train = pd.read_table(os.path.join(INPUT_PATH, 'train.tsv'), engine='c')
#df_train = pd.read_csv('Mercari/train.tsv', nrows=1000, sep='\t')
#price = df_train['price']
logprice = np.log1p(df_train['price'])
X_train = get_X_from_df(df_train)
print('names')
words_dict = get_dict_with_split(X_train[:,1])

print('dv_names')
dv_names = DictVectorizer(sparse=True)
dv_names.fit(words_dict)
names_vec_matrix = dv_names.transform(words_dict)
pickle.dump(dv_names, open('../dv_names.dv', 'wb'))
dv_names = 0
print('category1')
words_dict = get_dict_with_split(X_train[:,7])

print('dv_cat1')
dv_cat1 = DictVectorizer(sparse=True)
dv_cat1.fit(words_dict)
cat1d_vec_matrix = dv_cat1.transform(words_dict)
pickle.dump(dv_cat1, open('../dv_cat1.dv', 'wb'))
dv_cat1 = 0
print('category2')
words_dict = get_dict_with_split(X_train[:,8])

print('dv_cat2')
dv_cat2 = DictVectorizer(sparse=True)
dv_cat2.fit(words_dict)
cat2d_vec_matrix = dv_cat2.transform(words_dict)
pickle.dump(dv_cat2, open('../dv_cat2.dv', 'wb'))
dv_cat2 = 0
print('category3')
words_dict = get_dict_with_split(X_train[:,9])

print('dv_cat3')
dv_cat3 = DictVectorizer(sparse=True)
dv_cat3.fit(words_dict)
cat3d_vec_matrix = dv_cat3.transform(words_dict)
pickle.dump(dv_cat3, open('../dv_cat3.dv', 'wb'))
dv_cat3 = 0
print('description')
words_dict = get_dict_with_split(X_train[:,6])

print('dv_des')
dv_des = DictVectorizer(sparse=True)
dv_des.fit(words_dict)
des_vec_matrix = dv_des.transform(words_dict)
pickle.dump(dv_des, open('../dv_des.dv', 'wb'))
dv_des = 0
condition = X_train[:,2]
shipping = X_train[:,5]

print('MinMaxScaler')
mms = MinMaxScaler()

mms.fit(condition.reshape(-1,1))
condition_arr_n = mms.transform(condition.reshape(-1,1))
shipping_arr_n = shipping.reshape(-1,1).astype('float64')
pickle.dump(mms, open('../mms.mms', 'wb'))
mms = 0
print('brands')
words_dict = get_dict_without_split(X_train[:,4])

print('dv_brand')
dv_brand = DictVectorizer(sparse=True)
dv_brand.fit(words_dict)
brand_vec_matrix = dv_brand.transform(words_dict)
pickle.dump(dv_brand, open('../dv_brand.dv', 'wb'))
dv_brand = 0
print('X_modified')
X_modified = hstack((names_vec_matrix,cat1d_vec_matrix,cat2d_vec_matrix,cat3d_vec_matrix,shipping_arr_n,condition_arr_n,brand_vec_matrix,des_vec_matrix))
X_modified.shape
names_vec_matrix = 0
cat1d_vec_matrix = 0
cat2d_vec_matrix = 0
cat3d_vec_matrix = 0
des_vec_matrix = 0
shipping_arr_n = 0
condition_arr_n = 0
words_dict = 0
t1 = time.time()
print('LGBMRegressor')

lgb_r = lgb.LGBMRegressor(subsample_for_bin = 100000, reg_lambda = 0.0, reg_alpha = 0.0, num_leaves = 51, 
                          n_estimators = 1000, min_split_gain = 0.0, min_child_samples = 20, max_depth = -1, 
                          learning_rate = 0.1, importance_type = 'split', class_weight = None, 
                          boosting_type = 'gbdt')
lgb_r.fit(X_modified, logprice)
t2 = time.time()
print(t2-t1)
pickle.dump(lgb_r, open('../lgb_r.lgb', 'wb'))
lgb_r = 0
df_test = pd.read_table(os.path.join(INPUT_PATH, 'test_stg2.tsv'), engine='c')
#df_test = pd.read_csv('Mercari/test.tsv', nrows=1000, sep='\t')
X_test = get_X_from_df(df_test)
dv_names = pickle.load(open('../dv_names.dv', 'rb'))
print('names')
words_dict = get_dict_with_split(X_test[:,1])

print('dv_names')
names_vec_matrix = dv_names.transform(words_dict)
dv_names = 0
dv_cat1 = pickle.load(open('../dv_cat1.dv', 'rb'))
print('category1')
words_dict = get_dict_with_split(X_test[:,7])

print('dv_cat1')
cat1d_vec_matrix = dv_cat1.transform(words_dict)
dv_cat1 = 0
dv_cat2 = pickle.load(open('../dv_cat2.dv', 'rb'))
print('category2')
words_dict = get_dict_with_split(X_test[:,8])

print('dv_cat2')
cat2d_vec_matrix = dv_cat2.transform(words_dict)
dv_cat2 = 0
dv_cat3 = pickle.load(open('../dv_cat3.dv', 'rb'))
print('category3')
words_dict = get_dict_with_split(X_test[:,9])

print('dv_cat3')
cat3d_vec_matrix = dv_cat3.transform(words_dict)
dv_cat3 = 0
dv_des = pickle.load(open('../dv_des.dv', 'rb'))
print('description')
words_dict = get_dict_with_split(X_test[:,6])

print('dv_des')
des_vec_matrix = dv_des.transform(words_dict)
dv_des = 0
mms = pickle.load(open('../mms.mms', 'rb'))
condition = X_test[:,2]
shipping = X_test[:,5]

print('MinMaxScaler')

condition_arr_n = mms.transform(condition.reshape(-1,1))
shipping_arr_n = shipping.reshape(-1,1).astype('float64')
mms = 0
dv_brand = pickle.load(open('../dv_brand.dv', 'rb'))
print('brands')
words_dict = get_dict_without_split(X_test[:,4])

print('dv_brand')
brand_vec_matrix = dv_brand.transform(words_dict)
print('X_test_modified')
X_test_modified = hstack((names_vec_matrix,cat1d_vec_matrix,cat2d_vec_matrix,cat3d_vec_matrix,shipping_arr_n,condition_arr_n,brand_vec_matrix,des_vec_matrix))
names_vec_matrix = 0
cat1d_vec_matrix = 0
cat2d_vec_matrix = 0
cat3d_vec_matrix = 0
des_vec_matrix = 0
shipping_arr_n = 0
condition_arr_n = 0
words_dict = 0
X_test_modified.shape
lgb_r = pickle.load(open('../lgb_r.lgb', 'rb'))
print('y_pred')
y_pred = lgb_r.predict(X_test_modified)
print('y_pred_e')
y_pred_e = np.array([np.power(np.e,y)-1 for y in y_pred])
for i in np.where(y_pred_e < 0)[0]:
    y_pred_e[i] = np.mean(price)
submission = df_test[['test_id']]
submission['price'] = y_pred_e
submission.to_csv('lgbm_submission.csv', index = False)
