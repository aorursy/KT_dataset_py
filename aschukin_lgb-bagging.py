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
price = df_train['price']
mean_price = np.mean(price)
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
print('brands')
words_dict = get_dict_without_split(X_train[:,4])

print('dv_brand')
dv_brand = DictVectorizer(sparse=True)
dv_brand.fit(words_dict)
brand_vec_matrix = dv_brand.transform(words_dict)
pickle.dump(dv_brand, open('../dv_brand.dv', 'wb'))
dv_brand = 0
condition = X_train[:,2]
shipping = X_train[:,5]

print('MinMaxScaler')
mms = MinMaxScaler()

mms.fit(condition.reshape(-1,1))
condition_arr_n = mms.transform(condition.reshape(-1,1))
shipping_arr_n = shipping.reshape(-1,1).astype('float64')
pickle.dump(mms, open('../mms.mms', 'wb'))
mms = 0

lgb_names = lgb.LGBMRegressor(subsample_for_bin = 100000, reg_lambda = 0.0, reg_alpha = 0.0, num_leaves = 51, 
                              n_estimators = 1000, min_split_gain = 0.0, min_child_samples = 20, max_depth = -1, 
                              learning_rate = 0.1, importance_type = 'split', class_weight = None, 
                              boosting_type = 'gbdt')
lgb_cat1 = lgb.LGBMRegressor(subsample_for_bin = 100000, reg_lambda = 0.0, reg_alpha = 0.0, num_leaves = 51, 
                              n_estimators = 1000, min_split_gain = 0.0, min_child_samples = 20, max_depth = -1, 
                              learning_rate = 0.1, importance_type = 'split', class_weight = None, 
                              boosting_type = 'gbdt')
lgb_cat2 = lgb.LGBMRegressor(subsample_for_bin = 100000, reg_lambda = 0.0, reg_alpha = 0.0, num_leaves = 51, 
                              n_estimators = 1000, min_split_gain = 0.0, min_child_samples = 20, max_depth = -1, 
                              learning_rate = 0.1, importance_type = 'split', class_weight = None, 
                              boosting_type = 'gbdt')
lgb_cat3 = lgb.LGBMRegressor(subsample_for_bin = 100000, reg_lambda = 0.0, reg_alpha = 0.0, num_leaves = 51, 
                              n_estimators = 1000, min_split_gain = 0.0, min_child_samples = 20, max_depth = -1, 
                              learning_rate = 0.1, importance_type = 'split', class_weight = None, 
                              boosting_type = 'gbdt')
lgb_des = lgb.LGBMRegressor(subsample_for_bin = 100000, reg_lambda = 0.0, reg_alpha = 0.0, num_leaves = 51, 
                              n_estimators = 1000, min_split_gain = 0.0, min_child_samples = 20, max_depth = -1, 
                              learning_rate = 0.1, importance_type = 'split', class_weight = None, 
                              boosting_type = 'gbdt')
lgb_brand = lgb.LGBMRegressor(subsample_for_bin = 100000, reg_lambda = 0.0, reg_alpha = 0.0, num_leaves = 51, 
                              n_estimators = 1000, min_split_gain = 0.0, min_child_samples = 20, max_depth = -1, 
                              learning_rate = 0.1, importance_type = 'split', class_weight = None, 
                              boosting_type = 'gbdt')
lgb_condition = lgb.LGBMRegressor(subsample_for_bin = 100000, reg_lambda = 0.0, reg_alpha = 0.0, num_leaves = 51, 
                              n_estimators = 1000, min_split_gain = 0.0, min_child_samples = 20, max_depth = -1, 
                              learning_rate = 0.1, importance_type = 'split', class_weight = None, 
                              boosting_type = 'gbdt')
lgb_shipping = lgb.LGBMRegressor(subsample_for_bin = 100000, reg_lambda = 0.0, reg_alpha = 0.0, num_leaves = 51, 
                              n_estimators = 1000, min_split_gain = 0.0, min_child_samples = 20, max_depth = -1, 
                              learning_rate = 0.1, importance_type = 'split', class_weight = None, 
                              boosting_type = 'gbdt')
t1 = time.time()

lgb_names.fit(names_vec_matrix, logprice)

t2 = time.time()
print(t2-t1)
pickle.dump(lgb_names, open('../lgb_names.lgb', 'wb'))
lgb_names = 0
t1 = time.time()

lgb_cat1.fit(cat1d_vec_matrix, logprice)

t2 = time.time()
print(t2-t1)
pickle.dump(lgb_cat1, open('../lgb_cat1.lgb', 'wb'))
lgb_cat1 = 0
t1 = time.time()

lgb_cat2.fit(cat2d_vec_matrix, logprice)

t2 = time.time()
print(t2-t1)
pickle.dump(lgb_cat2, open('../lgb_cat2.lgb', 'wb'))
lgb_cat2 = 0
t1 = time.time()

lgb_cat3.fit(cat3d_vec_matrix, logprice)

t2 = time.time()
print(t2-t1)
pickle.dump(lgb_cat3, open('../lgb_cat3.lgb', 'wb'))
lgb_cat3 = 0
t1 = time.time()

lgb_des.fit(des_vec_matrix, logprice)

t2 = time.time()
print(t2-t1)
pickle.dump(lgb_des, open('../lgb_des.lgb', 'wb'))
lgb_des = 0
t1 = time.time()

lgb_brand.fit(brand_vec_matrix, logprice)

t2 = time.time()
print(t2-t1)
pickle.dump(lgb_brand, open('../lgb_brand.lgb', 'wb'))
lgb_brand = 0
t1 = time.time()

lgb_condition.fit(condition_arr_n, logprice)

t2 = time.time()
print(t2-t1)
pickle.dump(lgb_condition, open('../lgb_condition.lgb', 'wb'))
lgb_condition = 0
t1 = time.time()

lgb_shipping.fit(shipping_arr_n, logprice)

t2 = time.time()
print(t2-t1)
pickle.dump(lgb_shipping, open('../lgb_shipping.lgb', 'wb'))
lgb_shipping = 0
names_vec_matrix = 0
cat1d_vec_matrix = 0
cat2d_vec_matrix = 0
cat3d_vec_matrix = 0
des_vec_matrix = 0
shipping_arr_n = 0
condition_arr_n = 0
words_dict = 0

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
dv_brand = pickle.load(open('../dv_brand.dv', 'rb'))
print('brands')
words_dict = get_dict_without_split(X_test[:,4])

print('dv_brand')
brand_vec_matrix = dv_brand.transform(words_dict)
dv_brand = 0
mms = pickle.load(open('../mms.mms', 'rb'))
condition = X_test[:,2]
shipping = X_test[:,5]

print('MinMaxScaler')

condition_arr_n = mms.transform(condition.reshape(-1,1))
shipping_arr_n = shipping.reshape(-1,1).astype('float64')
mms = 0

def fill_bad_predict(y_pred):
    for i in np.where(y_pred < 0)[0]:
        y_pred[i] = mean_price 
    return y_pred
lgb_names = pickle.load(open('../lgb_names.lgb', 'rb'))
y_names_pred = lgb_names.predict(names_vec_matrix)
y_names_pred_e = np.array([np.power(np.e,y)-1 for y in y_names_pred])
y_names_pred_e = fill_bad_predict(y_names_pred_e)
lgb_names = 0
lgb_cat1 = pickle.load(open('../lgb_cat1.lgb', 'rb'))
y_cat1_pred = lgb_cat1.predict(cat1d_vec_matrix)
y_cat1_pred_e = np.array([np.power(np.e,y)-1 for y in y_cat1_pred])
y_cat1_pred_e = fill_bad_predict(y_cat1_pred_e)
lgb_cat1 = 0
lgb_cat2 = pickle.load(open('../lgb_cat2.lgb', 'rb'))
y_cat2_pred = lgb_cat2.predict(cat2d_vec_matrix)
y_cat2_pred_e = np.array([np.power(np.e,y)-1 for y in y_cat2_pred])
y_cat2_pred_e = fill_bad_predict(y_cat2_pred_e)
lgb_cat2 = 0
lgb_cat3 = pickle.load(open('../lgb_cat3.lgb', 'rb'))
y_cat3_pred = lgb_cat3.predict(cat3d_vec_matrix)
y_cat3_pred_e = np.array([np.power(np.e,y)-1 for y in y_cat3_pred])
y_cat3_pred_e = fill_bad_predict(y_cat3_pred_e)
lgb_cat3 = 0
lgb_des = pickle.load(open('../lgb_des.lgb', 'rb'))
y_des_pred = lgb_des.predict(des_vec_matrix)
y_des_pred_e = np.array([np.power(np.e,y)-1 for y in y_des_pred])
y_des_pred_e = fill_bad_predict(y_des_pred_e)
lgb_des = 0
lgb_brand = pickle.load(open('../lgb_brand.lgb', 'rb'))
y_brand_pred = lgb_brand.predict(brand_vec_matrix)
y_brand_pred_e = np.array([np.power(np.e,y)-1 for y in y_brand_pred])
y_brand_pred_e = fill_bad_predict(y_brand_pred_e)
lgb_brand = 0
lgb_condition = pickle.load(open('../lgb_condition.lgb', 'rb'))
y_condition_pred = lgb_condition.predict(condition_arr_n)
y_condition_pred_e = np.array([np.power(np.e,y)-1 for y in y_condition_pred])
y_condition_pred_e = fill_bad_predict(y_condition_pred_e)
lgb_condition = 0
lgb_shipping = pickle.load(open('../lgb_shipping.lgb', 'rb'))
y_shipping_pred = lgb_shipping.predict(shipping_arr_n)
y_shipping_pred_e = np.array([np.power(np.e,y)-1 for y in y_shipping_pred])
y_shipping_pred_e = fill_bad_predict(y_shipping_pred_e)
lgb_shipping = 0
names_vec_matrix = 0
cat1d_vec_matrix = 0
cat2d_vec_matrix = 0
cat3d_vec_matrix = 0
des_vec_matrix = 0
brand_vec_matrix = 0
shipping_arr_n = 0
condition_arr_n = 0
words_dict = 0
#estimator_coeffs = [0.18558060332780146, 0.09968598557916138, 0.10991467469607039, 0.12458372580652287, 0.15931743162296658, 0.12362674362116682, 0.09599324155170731, 0.10129759379460329]
estimator_coeffs = [1/8 for x in range(8)]
predictions = [y_names_pred_e,y_cat1_pred_e,y_cat2_pred_e,y_cat3_pred_e,y_des_pred_e,y_brand_pred_e,y_condition_pred_e,y_shipping_pred_e]
y_pred_e = np.zeros(y_names_pred_e.shape[0])
for i in range(len(predictions)):
    y_pred_e += estimator_coeffs[i] * predictions[i]
submission = df_test[['test_id']]
submission['price'] = y_pred_e
submission.to_csv('lgbm_submission.csv', index = False)
