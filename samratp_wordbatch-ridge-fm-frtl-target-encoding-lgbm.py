# Based on this wonderful notebook by Peter - https://www.kaggle.com/peterhurford/lgb-and-fm-18th-place-0-40604
import time
start_time = time.time()

SUBMIT_MODE = True
#SUBMIT_MODE = False

import pandas as pd
import numpy as np
import time
import gc
import string
import re
import random
random.seed(2018)

from nltk.corpus import stopwords

from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection.univariate_selection import SelectKBest, f_regression
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import wordbatch
from wordbatch.extractors import WordBag
from wordbatch.models import FM_FTRL

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.naive_bayes import MultinomialNB
import lightgbm as lgb

# Viz
import seaborn as sns
import matplotlib.pyplot as plt
def rmse(predicted, actual):
    return np.sqrt(((predicted - actual) ** 2).mean())
class TargetEncoder:
    # Adapted from https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features
    def __repr__(self):
        return 'TargetEncoder'

    def __init__(self, cols, smoothing=1, min_samples_leaf=1, noise_level=0, keep_original=False):
        self.cols = cols
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.noise_level = noise_level
        self.keep_original = keep_original

    @staticmethod
    def add_noise(series, noise_level):
        return series * (1 + noise_level * np.random.randn(len(series)))

    def encode(self, train, test, target):
        for col in self.cols:
            if self.keep_original:
                train[col + '_te'], test[col + '_te'] = self.encode_column(train[col], test[col], target)
            else:
                train[col], test[col] = self.encode_column(train[col], test[col], target)
        return train, test

    def encode_column(self, trn_series, tst_series, target):
        temp = pd.concat([trn_series, target], axis=1)
        # Compute target mean
        averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
        # Compute smoothing
        smoothing = 1 / (1 + np.exp(-(averages["count"] - self.min_samples_leaf) / self.smoothing))
        # Apply average function to all target data
        prior = target.mean()
        # The bigger the count the less full_avg is taken into account
        averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
        averages.drop(['mean', 'count'], axis=1, inplace=True)
        # Apply averages to trn and tst series
        ft_trn_series = pd.merge(
            trn_series.to_frame(trn_series.name),
            averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
            on=trn_series.name,
            how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
        # pd.merge does not keep the index so restore it
        ft_trn_series.index = trn_series.index
        ft_tst_series = pd.merge(
            tst_series.to_frame(tst_series.name),
            averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
            on=tst_series.name,
            how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
        # pd.merge does not keep the index so restore it
        ft_tst_series.index = tst_series.index
        return self.add_noise(ft_trn_series, self.noise_level), self.add_noise(ft_tst_series, self.noise_level)
def to_number(x):
    try:
        if not x.isdigit():
            return 0
        x = int(x)
        if x > 100:
            return 100
        else:
            return x
    except:
        return 0

def sum_numbers(desc):
    if not isinstance(desc, str):
        return 0
    try:
        return sum([to_number(s) for s in desc.split()])
    except:
        return 0
# Define helpers for text normalization
stopwords_en = {x: 1 for x in stopwords.words('english')}
stopwords = {x: 1 for x in stopwords.words('russian')}
non_alphanums = re.compile(u'[^A-Za-z0-9]+')
non_alphanumpunct = re.compile(u'[^A-Za-z0-9\.?!,; \(\)\[\]\'\"\$]+')
RE_PUNCTUATION = '|'.join([re.escape(x) for x in string.punctuation])

def normalize_text(text):
    return u" ".join(
        [x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split(" ")] \
         if len(x) > 1 and x not in stopwords])

def clean_name(x):
    if len(x):
        x = non_alphanums.sub(' ', x).split()
        if len(x):
            return x[0].lower()
    return ''

    
print('[{}] Finished defining stuff'.format(time.time() - start_time))
train = pd.read_csv('../input/train.csv', index_col = "item_id", parse_dates = ["activation_date"])
test = pd.read_csv('../input/test.csv', index_col = "item_id", parse_dates = ["activation_date"])
print('[{}] Finished load data'.format(time.time() - start_time))
train['is_train'] = 1
test['is_train'] = 0
print('[{}] Compiled train / test'.format(time.time() - start_time))
print('Train shape: ', train.shape)
print('Test shape: ', test.shape)

y = train.deal_probability.copy()
nrow_train = train.shape[0]

merge = pd.concat([train, test])
submission = pd.DataFrame(test.index)
print('[{}] Compiled merge'.format(time.time() - start_time))
print('Merge shape: ', merge.shape)

del train
del test
gc.collect()
print('[{}] Garbage collection'.format(time.time() - start_time))
print("Feature Engineering - Part 1")
merge["price"] = np.log(merge["price"]+0.001)
merge["price"].fillna(-999,inplace=True)
merge["image_top_1"].fillna(-999,inplace=True)

print("\nCreate Time Variables")
merge["activation_weekday"] = merge['activation_date'].dt.weekday
merge["Weekd_of_Year"] = merge['activation_date'].dt.week
merge["Day_of_Month"] = merge['activation_date'].dt.day

print(merge.head(5))
gc.collect()
# Create Validation Index and Remove Dead Variables
training_index = merge.loc[merge.activation_date<=pd.to_datetime('2017-04-07')].index
validation_index = merge.loc[merge.activation_date>=pd.to_datetime('2017-04-08')].index
merge.drop(["activation_date","image"],axis=1,inplace=True)

merge['param_1_copy'] = merge['param_1']

#Drop user_id
merge.drop(["user_id"], axis=1,inplace=True)
# Meta Text Features
print("\nText Features")
textfeats = ["description", "title", "param_1_copy"]

for cols in textfeats:
    merge[cols] = merge[cols].astype(str) 
    merge[cols] = merge[cols].astype(str).fillna('missing') # FILL NA
    merge[cols] = merge[cols].str.lower() # Lowercase all text, so that capitalized words dont get treated differently
    merge[cols + '_num_stopwords'] = merge[cols].apply(lambda x: len([w for w in x.split() if w in stopwords])) # Count number of Stopwords
    merge[cols + '_num_stopwords_en'] = merge[cols].apply(lambda x: len([w for w in x.split() if w in stopwords_en])) # Count number of Stopwords
    merge[cols + '_num_punctuations'] = merge[cols].apply(lambda comment: (comment.count(RE_PUNCTUATION))) # Count number of Punctuations
    merge[cols + '_num_alphabets'] = merge[cols].apply(lambda comment: (comment.count(r'[a-zA-Z]'))) # Count number of Alphabets
    merge[cols + '_num_alphanumeric'] = merge[cols].apply(lambda comment: (comment.count(r'[A-Za-z0-9]'))) # Count number of AlphaNumeric
    merge[cols + '_num_digits'] = merge[cols].apply(lambda comment: (comment.count('[0-9]'))) # Count number of Digits
    merge[cols + '_num_letters'] = merge[cols].apply(lambda comment: len(comment)) # Count number of Letters
    merge[cols + '_num_words'] = merge[cols].apply(lambda comment: len(comment.split())) # Count number of Words
    merge[cols + '_num_unique_words'] = merge[cols].apply(lambda comment: len(set(w for w in comment.split())))
    merge[cols + '_words_vs_unique'] = merge[cols+'_num_unique_words'] / merge[cols+'_num_words'] # Count Unique Words
    merge[cols + '_letters_per_word'] = merge[cols+'_num_letters'] / merge[cols+'_num_words'] # Letters per Word
    merge[cols + '_punctuations_by_letters'] = merge[cols+'_num_punctuations'] / merge[cols+'_num_letters'] # Punctuations by Letters
    merge[cols + '_punctuations_by_words'] = merge[cols+'_num_punctuations'] / merge[cols+'_num_words'] # Punctuations by Words
    merge[cols + '_digits_by_letters'] = merge[cols+'_num_digits'] / merge[cols+'_num_letters'] # Digits by Letters
    merge[cols + '_alphanumeric_by_letters'] = merge[cols+'_num_alphanumeric'] / merge[cols+'_num_letters'] # AlphaNumeric by Letters
    merge[cols + '_alphabets_by_letters'] = merge[cols+'_num_alphabets'] / merge[cols+'_num_letters'] # Alphabets by Letters
    merge[cols + '_stopwords_by_letters'] = merge[cols+'_num_stopwords'] / merge[cols+'_num_letters'] # Stopwords by Letters
    merge[cols + '_stopwords_by_words'] = merge[cols+'_num_stopwords'] / merge[cols+'_num_words'] # Stopwords by Letters
    merge[cols + '_stopwords_by_letters_en'] = merge[cols+'_num_stopwords_en'] / merge[cols+'_num_letters'] # Stopwords by Letters
    merge[cols + '_stopwords_by_words_en'] = merge[cols+'_num_stopwords_en'] / merge[cols+'_num_words'] # Stopwords by Letters    
    merge[cols + '_mean'] = merge[cols].apply(lambda x: 0 if len(x) == 0 else float(len(x.split())) / len(x)) * 10 # Mean
    merge[cols + '_num_sum'] = merge[cols].apply(sum_numbers) 

# Extra Feature Engineering
merge['title_desc_len_ratio'] = merge['title_num_letters']/(merge['description_num_letters']+1)
merge['title_param1_len_ratio'] = merge['title_num_letters']/(merge['param_1_copy_num_letters']+1)
merge['param_1_copy_desc_len_ratio'] = merge['param_1_copy_num_letters']/(merge['description_num_letters']+1)

gc.collect()
cols = set(merge.columns.values)
cat_cols = {"region","city","parent_category_name","category_name","user_type","image_top_1"}
basic_cols = {"region","city","parent_category_name","category_name","user_type","image_top_1",
               "description","title","param_1_copy","param_1","param_2","param_3", "price", "item_seq_number"}
df_test = merge.loc[merge['is_train'] == 0]
df_train = merge.loc[merge['is_train'] == 1]
del merge
gc.collect()
df_test = df_test.drop(['is_train'], axis=1)
df_train = df_train.drop(['is_train'], axis=1)

print(df_train.shape)
print(y.shape)

if SUBMIT_MODE:
    y_train = y
    del y
    gc.collect()
else:
    df_train, df_test, y_train, y_test = train_test_split(df_train, y, test_size=0.2, random_state=144)

print('[{}] Splitting completed.'.format(time.time() - start_time))
wb = wordbatch.WordBatch(None, extractor=(WordBag, {"hash_ngrams": 2,
                                                              "hash_ngrams_weights": [1.5, 1.0],
                                                              "hash_size": 2 ** 29,
                                                              "norm": None,
                                                              "tf": 'binary',
                                                              "idf": None,
                                                              }), procs=8)
wb.dictionary_freeze = True
X_name_train = wb.fit_transform(df_train['title'])
X_name_test = wb.transform(df_test['title'])
del(wb)
mask = np.where(X_name_train.getnnz(axis=0) > 2)[0]
X_name_train = X_name_train[:, mask]
X_name_test = X_name_test[:, mask]
print('[{}] Vectorize `title` completed.'.format(time.time() - start_time))
gc.collect()
X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(X_name_train, y_train,
                                                              test_size = 0.5,
                                                              shuffle = False)
print('[{}] Finished splitting'.format(time.time() - start_time))

# Ridge adapted from https://www.kaggle.com/object/more-effective-ridge-script?scriptVersionId=1851819
model = Ridge(solver="sag", fit_intercept=True, random_state=42, alpha=30)
model.fit(X_train_1, y_train_1)
print('[{}] Finished to train name ridge (1)'.format(time.time() - start_time))
name_ridge_preds1 = model.predict(X_train_2)
name_ridge_preds1f = model.predict(X_name_test)
print('[{}] Finished to predict name ridge (1)'.format(time.time() - start_time))
model = Ridge(solver="sag", fit_intercept=True, random_state=42, alpha=30)
model.fit(X_train_2, y_train_2)
print('[{}] Finished to train name ridge (2)'.format(time.time() - start_time))
name_ridge_preds2 = model.predict(X_train_1)
name_ridge_preds2f = model.predict(X_name_test)
print('[{}] Finished to predict name ridge (2)'.format(time.time() - start_time))
name_ridge_preds_oof = np.concatenate((name_ridge_preds2, name_ridge_preds1), axis=0)
name_ridge_preds_test = (name_ridge_preds1f + name_ridge_preds2f) / 2.0
print('RMSLE OOF: {}'.format(rmse(name_ridge_preds_oof, y_train)))
if not SUBMIT_MODE:
    print('RMSLE TEST: {}'.format(rmse(name_ridge_preds_test, y_test)))
gc.collect()
wb = wordbatch.WordBatch(None, extractor=(WordBag, {"hash_ngrams": 2,
                                                              "hash_ngrams_weights": [1.0, 1.0],
                                                              "hash_size": 2 ** 28,
                                                              "norm": "l2",
                                                              "tf": 1.0,
                                                              "idf": None}), procs=8)
wb.dictionary_freeze = True
X_description_train = wb.fit_transform(df_train['description'])
X_description_test = wb.transform(df_test['description'])
del(wb)
mask = np.where(X_description_train.getnnz(axis=0) > 3)[0]
X_description_train = X_description_train[:, mask]
X_description_test = X_description_test[:, mask]
print('[{}] Vectorize `description` completed.'.format(time.time() - start_time))
gc.collect()
X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(X_description_train, y_train,
                                                              test_size = 0.5,
                                                              shuffle = False)
print('[{}] Finished splitting'.format(time.time() - start_time))

# Ridge adapted from https://www.kaggle.com/object/more-effective-ridge-script?scriptVersionId=1851819
model = Ridge(solver="sag", fit_intercept=True, random_state=205, alpha=3.3)
model.fit(X_train_1, y_train_1)
print('[{}] Finished to train desc ridge (1)'.format(time.time() - start_time))
desc_ridge_preds1 = model.predict(X_train_2)
desc_ridge_preds1f = model.predict(X_description_test)
print('[{}] Finished to predict desc ridge (1)'.format(time.time() - start_time))
model = Ridge(solver="sag", fit_intercept=True, random_state=205, alpha=3.3)
model.fit(X_train_2, y_train_2)
print('[{}] Finished to train desc ridge (2)'.format(time.time() - start_time))
desc_ridge_preds2 = model.predict(X_train_1)
desc_ridge_preds2f = model.predict(X_description_test)
print('[{}] Finished to predict desc ridge (2)'.format(time.time() - start_time))
desc_ridge_preds_oof = np.concatenate((desc_ridge_preds2, desc_ridge_preds1), axis=0)
desc_ridge_preds_test = (desc_ridge_preds1f + desc_ridge_preds2f) / 2.0
print('RMSLE OOF: {}'.format(rmse(desc_ridge_preds_oof, y_train)))
if not SUBMIT_MODE:
    print('RMSLE TEST: {}'.format(rmse(desc_ridge_preds_test, y_test)))
gc.collect()
wb = wordbatch.WordBatch(None, extractor=(WordBag, {"hash_ngrams": 2,
                                                              "hash_ngrams_weights": [1.0, 1.0],
                                                              "hash_size": 2 ** 28,
                                                              "norm": "l2",
                                                              "tf": 1.0,
                                                              "idf": None}), procs=8)
wb.dictionary_freeze = True
X_param1_train = wb.fit_transform(df_train['param_1_copy'])
X_param1_test = wb.transform(df_test['param_1_copy'])
del(wb)
mask = np.where(X_param1_train.getnnz(axis=0) > 3)[0]
X_param1_train = X_param1_train[:, mask]
X_param1_test = X_param1_test[:, mask]
print('[{}] Vectorize `param_1_copy` completed.'.format(time.time() - start_time))
gc.collect()
X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(X_param1_train, y_train,
                                                              test_size = 0.5,
                                                              shuffle = False)
print('[{}] Finished splitting'.format(time.time() - start_time))

# Ridge adapted from https://www.kaggle.com/object/more-effective-ridge-script?scriptVersionId=1851819
model = Ridge(solver="sag", fit_intercept=True, random_state=205, alpha=3.3)
model.fit(X_train_1, y_train_1)
print('[{}] Finished to train param1 ridge (1)'.format(time.time() - start_time))
param1_ridge_preds1 = model.predict(X_train_2)
param1_ridge_preds1f = model.predict(X_param1_test)
print('[{}] Finished to predict param1 ridge (1)'.format(time.time() - start_time))
model = Ridge(solver="sag", fit_intercept=True, random_state=205, alpha=3.3)
model.fit(X_train_2, y_train_2)
print('[{}] Finished to train param1 ridge (2)'.format(time.time() - start_time))
param1_ridge_preds2 = model.predict(X_train_1)
param1_ridge_preds2f = model.predict(X_param1_test)
print('[{}] Finished to predict param1 ridge (2)'.format(time.time() - start_time))
param1_ridge_preds_oof = np.concatenate((param1_ridge_preds2, param1_ridge_preds1), axis=0)
param1_ridge_preds_test = (param1_ridge_preds1f + param1_ridge_preds2f) / 2.0
print('RMSLE OOF: {}'.format(rmse(param1_ridge_preds_oof, y_train)))
if not SUBMIT_MODE:
    print('RMSLE TEST: {}'.format(rmse(param1_ridge_preds_test, y_test)))
gc.collect()
del X_train_1
del X_train_2
del y_train_1
del y_train_2
del name_ridge_preds1
del name_ridge_preds1f
del name_ridge_preds2
del name_ridge_preds2f
del desc_ridge_preds1
del desc_ridge_preds1f
del desc_ridge_preds2
del desc_ridge_preds2f
del param1_ridge_preds1
del param1_ridge_preds1f
del param1_ridge_preds2
del param1_ridge_preds2f
gc.collect()
print('[{}] Finished garbage collection'.format(time.time() - start_time))
lb = LabelBinarizer(sparse_output=True)
X_parent_category_train = lb.fit_transform(df_train['parent_category_name'])
X_parent_category_test = lb.transform(df_test['parent_category_name'])
print('[{}] Finished label binarize `parent_category_name`'.format(time.time() - start_time))
X_category_train = lb.fit_transform(df_train['category_name'])
X_category_test = lb.transform(df_test['category_name'])
print('[{}] Finished label binarize `category_name`'.format(time.time() - start_time))
X_region_train = lb.fit_transform(df_train['region'])
X_region_test = lb.transform(df_test['region'])
print('[{}] Finished label binarize `region`'.format(time.time() - start_time))
X_city_train = lb.fit_transform(df_train['city'])
X_city_test = lb.transform(df_test['city'])
print('[{}] Finished label binarize `city`'.format(time.time() - start_time))
X_imagetop1_train = lb.fit_transform(df_train['image_top_1'])
X_imagetop1_test = lb.transform(df_test['image_top_1'])
print('[{}] Finished label binarize `image_top_1`'.format(time.time() - start_time))
X_user_type_train = lb.fit_transform(df_train['user_type'])
X_user_type_test = lb.transform(df_test['user_type'])
print('[{}] Finished label binarize `user_type`'.format(time.time() - start_time))
num_features = list(cols - (basic_cols))
num_features.remove('is_train')

scaler = StandardScaler()

train_num_features = scaler.fit_transform(df_train[num_features].drop('deal_probability', axis=1))
test_num_features = scaler.fit_transform(df_test[num_features].drop('deal_probability', axis=1))

train_num_features = csr_matrix(train_num_features)
test_num_features = csr_matrix(test_num_features)
sparse_merge_train = hstack((X_description_train, X_param1_train, X_name_train, X_region_train, X_city_train, X_imagetop1_train, X_user_type_train)).tocsr()
sparse_merge_train = hstack([sparse_merge_train, train_num_features])
print('[{}] Create sparse merge train completed'.format(time.time() - start_time))

sparse_merge_test = hstack((X_description_test, X_param1_test, X_name_test, X_region_test, X_city_test, X_imagetop1_test, X_user_type_test)).tocsr()
sparse_merge_test = hstack([sparse_merge_test, test_num_features])
print('[{}] Create sparse merge test completed'.format(time.time() - start_time))
gc.collect()
print("\n FM_FTRL Starting...........")
if SUBMIT_MODE:
    iters = 3
else:
    iters = 1
    rounds = 3

model = FM_FTRL(alpha=0.035, beta=0.001, L1=0.00001, L2=0.15, D=sparse_merge_train.shape[1],
                alpha_fm=0.05, L2_fm=0.0, init_fm=0.01,
                D_fm=100, e_noise=0, iters=iters, inv_link="identity", threads=4)
if SUBMIT_MODE:
    model.fit(sparse_merge_train, y_train)
    print('[{}] Train FM completed'.format(time.time() - start_time))
    predsFM = model.predict(sparse_merge_test)
    print('[{}] Predict FM completed'.format(time.time() - start_time))
else:
    for i in range(rounds):
        model.fit(sparse_merge_train, y_train)
        predsFM = model.predict(sparse_merge_test)
        print('[{}] Iteration {}/{} -- RMSLE: {}'.format(time.time() - start_time, i + 1, rounds, rmse(predsFM, y_test)))

del model
gc.collect()
if not SUBMIT_MODE:
    print("FM_FTRL dev RMSLE:", rmse(predsFM, y_test))
del X_description_train, lb, X_name_train, X_param1_train, X_region_train, X_city_train, X_imagetop1_train, X_user_type_train
del X_description_test, X_name_test, X_param1_test, X_region_test, X_city_test, X_imagetop1_test, X_user_type_test
gc.collect()
fselect = SelectKBest(f_regression, k=48000)
train_features = fselect.fit_transform(sparse_merge_train, y_train)
test_features = fselect.transform(sparse_merge_test)
print('[{}] Select best completed'.format(time.time() - start_time))


del sparse_merge_train
del sparse_merge_test
gc.collect()
print('[{}] Garbage collection'.format(time.time() - start_time))
tv = TfidfVectorizer(max_features=250000,
                     ngram_range=(1, 3),
                     stop_words=None)
X_name_train = tv.fit_transform(df_train['title'])
print('[{}] Finished TFIDF vectorize `title` (1/2)'.format(time.time() - start_time))
X_name_test = tv.transform(df_test['title'])
print('[{}] Finished TFIDF vectorize `title` (2/2)'.format(time.time() - start_time))
tv = TfidfVectorizer(max_features=100000,
                     ngram_range=(1, 2),
                     stop_words=None)
X_description_train = tv.fit_transform(df_train['description'])
print('[{}] Finished TFIDF vectorize `description` (1/2)'.format(time.time() - start_time))
X_description_test = tv.transform(df_test['description'])
print('[{}] Finished TFIDF vectorize `description` (2/2)'.format(time.time() - start_time))
tv = TfidfVectorizer(max_features=50000,
                     ngram_range=(1, 2),
                     stop_words=None)
X_param1_train = tv.fit_transform(df_train['param_1_copy'])
print('[{}] Finished TFIDF vectorize `param_1_copy` (1/2)'.format(time.time() - start_time))
X_param1_test = tv.transform(df_test['param_1_copy'])
print('[{}] Finished TFIDF vectorize `param_1_copy` (2/2)'.format(time.time() - start_time))
sparse_merge_train = hstack((X_description_train, X_param1_train, X_name_train)).tocsr()
del X_description_train, X_param1_train, X_name_train
gc.collect()
print('[{}] Create sparse merge train completed'.format(time.time() - start_time))

sparse_merge_test = hstack((X_description_test, X_param1_test, X_name_test)).tocsr()
X_description_test, X_param1_test, X_name_test
gc.collect()
print('[{}] Create sparse merge test completed'.format(time.time() - start_time))
X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(sparse_merge_train, y_train,
                                                              test_size = 0.5,
                                                              shuffle = False)
print('[{}] Finished splitting'.format(time.time() - start_time))

model = Ridge(solver="sag", fit_intercept=True, random_state=205, alpha=3.3)
model.fit(X_train_1, y_train_1)
print('[{}] Finished to train ridge (1)'.format(time.time() - start_time))
ridge_preds1 = model.predict(X_train_2)
ridge_preds1f = model.predict(sparse_merge_test)
print('[{}] Finished to predict ridge (1)'.format(time.time() - start_time))
model = Ridge(solver="sag", fit_intercept=True, random_state=205, alpha=3.3)
model.fit(X_train_2, y_train_2)
print('[{}] Finished to train ridge (2)'.format(time.time() - start_time))
ridge_preds2 = model.predict(X_train_1)
ridge_preds2f = model.predict(sparse_merge_test)
print('[{}] Finished to predict ridge (2)'.format(time.time() - start_time))
ridge_preds_oof = np.concatenate((ridge_preds2, ridge_preds1), axis=0)
ridge_preds_test = (ridge_preds1f + ridge_preds2f) / 2.0
print('RMSLE OOF: {}'.format(rmse(ridge_preds_oof, y_train)))
if not SUBMIT_MODE:
    print('RMSLE TEST: {}'.format(rmse(ridge_preds_test, y_test)))
del ridge_preds1
del ridge_preds1f
del ridge_preds2
del ridge_preds2f
#del mnb_preds1
#del mnb_preds1f
#del mnb_preds2
#del mnb_preds2f
del X_train_1
del X_train_2
del y_train_1
del y_train_2
del sparse_merge_train
del sparse_merge_test
del model
gc.collect()
print('[{}] Finished garbage collection'.format(time.time() - start_time))
df_train['ridge'] = ridge_preds_oof
df_train['name_ridge'] = name_ridge_preds_oof
df_train['desc_ridge'] = desc_ridge_preds_oof
df_train['param1_ridge'] = param1_ridge_preds_oof
#df_train['mnb'] = mnb_preds_oof
df_test['ridge'] = ridge_preds_test
df_test['name_ridge'] = name_ridge_preds_test
df_test['desc_ridge'] = desc_ridge_preds_test
df_test['param1_ridge'] = param1_ridge_preds_test
#df_test['mnb'] = mnb_preds_test
print('[{}] Finished adding submodels'.format(time.time() - start_time))
f_cats = ["region","city","parent_category_name","category_name","user_type","image_top_1"]
target_encode = TargetEncoder(min_samples_leaf=100, smoothing=10, noise_level=0.01,
                              keep_original=True, cols=f_cats)
df_train, df_test = target_encode.encode(df_train, df_test, y_train)
print('[{}] Finished target encoding'.format(time.time() - start_time))
df_train.drop(f_cats, axis=1, inplace=True)
df_test.drop(f_cats, axis=1, inplace=True)
#del mnb_preds_oof
#del mnb_preds_test
del ridge_preds_oof
del ridge_preds_test
gc.collect()
print('[{}] Finished garbage collection'.format(time.time() - start_time))
cols = ['region_te', 'city_te', 'parent_category_name_te', 'category_name_te',
        'user_type_te', 'image_top_1_te', 'desc_ridge', 'name_ridge', 'ridge']
train_dummies = csr_matrix(df_train[cols].values)
print('[{}] Finished dummyizing model 1/5'.format(time.time() - start_time))
test_dummies = csr_matrix(df_test[cols].values)
print('[{}] Finished dummyizing model 2/5'.format(time.time() - start_time))
del df_train
del df_test
gc.collect()
print('[{}] Finished dummyizing model 3/5'.format(time.time() - start_time))
train_features = hstack((train_features, train_dummies)).tocsr()
print('[{}] Finished dummyizing model 4/5'.format(time.time() - start_time))
test_features = hstack((test_features, test_dummies)).tocsr()
print('[{}] Finished dummyizing model 5/5'.format(time.time() - start_time))
d_train = lgb.Dataset(train_features, label=y_train)
del train_features
gc.collect()
if SUBMIT_MODE:
    watchlist = [d_train]
else:
    d_valid = lgb.Dataset(test_features, label=y_test)
    watchlist = [d_train, d_valid]

params = {
    'learning_rate': 0.02,
    'application': 'regression',
    'max_depth': 13,
    'num_leaves': 400,
    'verbosity': -1,
    'metric': 'RMSE',
    'data_random_seed': 1,
    'bagging_fraction': 0.8,
    'feature_fraction': 0.6,
    'nthread': 4,
    'lambda_l1': 10,
    'lambda_l2': 10
}
print('[{}] Finished compiling LGB'.format(time.time() - start_time))
modelL = lgb.train(params,
                  train_set=d_train,
                  num_boost_round=1350,
                  valid_sets=watchlist,
                  verbose_eval=50)

predsL = modelL.predict(test_features)

if not SUBMIT_MODE:
    print("LGB RMSLE:", rmse(predsL, y_test))
del d_train
del modelL
if not SUBMIT_MODE:
    del d_valid
gc.collect()
preds_final = predsFM * 0.30 + predsL * 0.70
if not SUBMIT_MODE:
    print('Final RMSE: ', rmse(preds_final, y_test))

if SUBMIT_MODE:
    submission['deal_probability'] = preds_final
    submission['deal_probability'] = submission['deal_probability'].clip(0.0, 1.0) # Between 0 and 1
    submission.to_csv('lgb_and_fm_separate_train_test.csv', index=False)
    print('[{}] Writing submission done'.format(time.time() - start_time))

print(submission.head(5))