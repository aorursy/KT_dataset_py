import warnings
warnings.filterwarnings('ignore')

import os
import shutil
import datetime
import gc
from tqdm import tqdm

import pandas as pd
import numpy as np
from numpy import median

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')

from sklearn.manifold import TSNE
from sklearn import preprocessing

from collections import Counter

import string
import re
from nltk.corpus import stopwords

import scipy
from scipy import hstack


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import mean_squared_error as mse
from math import sqrt
from sklearn.linear_model import Ridge

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import log_loss

from sklearn.model_selection import RandomizedSearchCV 
from scipy.stats import randint as sp_randint
from scipy.stats import uniform
from sklearn.feature_selection.univariate_selection import SelectKBest, f_regression
import tracemalloc
import time
tracemalloc.start()

start_time = time.time()
snapshot1 = tracemalloc.take_snapshot()
# https://www.kaggle.com/peterhurford/lgb-and-fm-18th-place-0-40604
def split_cat(text):
    try:
        return text.split("/")
    except:
        return ("No Label", "No Label", "No Label")
train = pd.read_csv('train.tsv', sep='\t', 
                      dtype={'item_condition_id': 'category', 'shipping': 'category'}, 
                      converters={'category_name': split_cat})
test = pd.read_csv('test.tsv', sep='\t', 
                     dtype={'item_condition_id': 'category', 'shipping': 'category'}, 
                     converters={'category_name': split_cat})
print('Shape of train data: ', train.shape)
print('Shape of test data: ', test.shape)
train.head(5)
train.isnull().any()
test.isnull().any()
# Split category_name by '/' into subcategories and replace nulls with 'missing'
train['gencat_name'] = train['category_name'].str.get(0).replace('', 'missing').astype('category')
train['subcat1_name'] = train['category_name'].str.get(1).fillna('missing').astype('category')
train['subcat2_name'] = train['category_name'].str.get(2).fillna('missing').astype('category')
train.drop('category_name', axis=1, inplace=True)
# Split category_name by '/' into subcategories and replace nulls with 'missing'
test['gencat_name'] = test['category_name'].str.get(0).replace('', 'missing').astype('category')
test['subcat1_name'] = test['category_name'].str.get(1).fillna('missing').astype('category')
test['subcat2_name'] = test['category_name'].str.get(2).fillna('missing').astype('category')
test.drop('category_name', axis=1, inplace=True)
train['item_description'].fillna('missing', inplace=True)
train['brand_name'] = train['brand_name'].fillna('missing').astype('category')
test['item_description'].fillna('missing', inplace=True)
test['brand_name'] = test['brand_name'].fillna('missing').astype('category')
train[train.duplicated()]
train.isnull().any()
print('Removed {} rows' .format(len(train[train.price<=0])))
train = train[train.price > 0].reset_index(drop=True)
train.name.describe()
train.item_condition_id.describe()
condition_count = Counter(list(train.item_condition_id))
x, y = zip(*condition_count.most_common())
plt.figure(figsize=[8,6])
plt.bar(x, y, )
for i, val in enumerate(y):
           plt.annotate(val, (x[i], y[i]), color='b')
plt.xlabel('item condition')
plt.ylabel('count')
plt.grid(False, axis='x')
plt.show()
train.brand_name.describe()
brand_count = Counter(list(train.brand_name.values))
x, y = zip(*brand_count.most_common(15))

plt.figure(figsize=[6,8])
plt.barh(x, y)
for i, val in enumerate(y):
           plt.annotate(val, (y[i], x[i]), color='b')
plt.gca().invert_yaxis()
plt.ylabel('Brand name')
plt.xlabel('count')
plt.grid(False, axis='y')
plt.show()
brand_missing = train[train.brand_name=='missing'].shape[0]
print('Brand name is missing for {} datapoints, i.e. {:.2f} % of train data.' .format(brand_missing, 100.0*brand_missing/train.shape[0]))
train.gencat_name.describe()
gencat_count = Counter(list(train.gencat_name.values))
x, y = zip(*gencat_count.most_common(15))
plt.figure(figsize=[6,8])
plt.barh(x, y)
for i, val in enumerate(y):
           plt.annotate(val, (y[i], x[i]), color='b')
plt.gca().invert_yaxis()
plt.ylabel('General category')
plt.xlabel('count')
plt.grid(False, axis='y')
plt.show()
gencat_missing = train[train.gencat_name=='missing'].shape[0]
print('category name is missing for {} datapoints, i.e. {:.2f} % of train data.' .format(gencat_missing, 100.0*gencat_missing/train.shape[0]))
train.subcat1_name.describe()
subcat1_count = Counter(list(train.subcat1_name.values))
x, y = zip(*subcat1_count.most_common(15))
plt.figure(figsize=[6,10])
plt.barh(x, y)
for i, val in enumerate(y):
           plt.annotate(val, (y[i], x[i]), color='b')
plt.gca().invert_yaxis()
plt.ylabel('Sub-category1')
plt.xlabel('count')
plt.grid(False, axis='y')
plt.show()
subcat1_missing = train[train.subcat1_name=='missing'].shape[0]
print('subcategory1 name is missing for {} datapoints, i.e. {:.2f} % of train data.' .format(subcat1_missing, 100.0*subcat1_missing/train.shape[0]))
train.subcat2_name.describe()
subcat2_count = Counter(list(train.subcat2_name.values))
x, y = zip(*subcat2_count.most_common(15))
plt.figure(figsize=[6,10])
plt.barh(x, y)
for i, val in enumerate(y):
           plt.annotate(val, (y[i], x[i]), color='b')
plt.gca().invert_yaxis()
plt.ylabel('Sub-category2')
plt.xlabel('count')
plt.grid(False, axis='y')
plt.show()
subcat2_missing = train[train.subcat2_name=='missing'].shape[0]
print('subcategory2 name is missing for {} datapoints, i.e. {:.2f} % of train data.' .format(subcat2_missing, 100.0*subcat2_missing/train.shape[0]))
desc_missing = train[train.item_description=='missing'].shape[0]
print('item description is missing for {} datapoints, i.e. {:.5f} % of train data.' .format(desc_missing, 100.0*desc_missing/train.shape[0]))
train[train.item_description=='missing']
sns.FacetGrid(train,size=6) \
    .map(sns.kdeplot,"price") \
    .add_legend();
plt.title('price density distribution')
plt.show();
sns.boxplot(y='price', data=train, showfliers=False)
plt.show()
for i in range(0, 100, 10):
    var =train["price"].values
    var = np.sort(var,axis = None)
    print("{} percentile value is {}".format(i,var[int(len(var)*(float(i)/100))]))
print("100 percentile value is ",var[-1])
for i in range(90, 100, 1):
    var =train["price"].values
    var = np.sort(var,axis = None)
    print("{} percentile value is {}".format(i,var[int(len(var)*(float(i)/100))]))
print("100 percentile value is ",var[-1])
def preprocess_name(text_col):
    preprocessed_names = []
    for sentence in tqdm(text_col.values):
        sent = sentence.replace('\\r', ' ')
        sent = sent.replace('\\"', ' ')
        sent = sent.replace('\\n', ' ')
        sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
        preprocessed_names.append(sent.lower().strip())
    return preprocessed_names

stopwords = stopwords.words('english')
def preprocess_desc(text_col):
    preprocessed_descs = []
    for sentence in tqdm(text_col.values):
        sent = sentence.replace('\\r', ' ')
        sent = sent.replace('\\"', ' ')
        sent = sent.replace('\\n', ' ')
        sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
        sent = ' '.join(e for e in sent.split() if e not in stopwords)
        preprocessed_descs.append(sent.lower().strip())
    return preprocessed_descs
train['preprocessed_name'] = preprocess_name(train['name'])
test['preprocessed_name'] = preprocess_name(test['name'])

train['preprocessed_description'] = preprocess_desc(train['item_description'])
test['preprocessed_description'] = preprocess_desc(test['item_description'])
def clean_cat(cat_values):
    '''takes categorical column values as arguments and returns list of cleaned categories'''
    
    catogories = list(cat_values)

    cat_list = []
    for i in tqdm(catogories):
        i = re.sub('[^A-Za-z0-9]+', ' ', i)
        i = i.replace(' ','')
        i = i.replace('&','_')
        cat_list.append(i.strip())
    
    return cat_list 
train['gencat_name'] = clean_cat(train['gencat_name'].values)
test['gencat_name'] = clean_cat(test['gencat_name'].values)

train['subcat1_name'] = clean_cat(train['subcat1_name'].values)
test['subcat1_name'] = clean_cat(test['subcat1_name'].values)

train['subcat2_name'] = clean_cat(train['subcat2_name'].values)
test['subcat2_name'] = clean_cat(test['subcat2_name'].values)
sns.set(style='whitegrid')
plt.figure(figsize=(12,6))
sns.boxplot(x='item_condition_id', y='price', data=train, showfliers=False)
plt.title('item_condition-wise distribution of price')
plt.show()
plt.figure(figsize=(15,6))
sns.boxplot(y='price', x='gencat_name', data=train, showfliers=False)
plt.xticks(rotation=45)
plt.title('category-wise distribution of price')
plt.show()
sns.barplot(y='gencat_name', x='price', data=train)
plt.title('mean price of various categories')
plt.show()
plt.figure(figsize=(10,25))
sns.barplot(y='subcat1_name', x='price', data=train)
plt.title('mean price of various subcategories')
plt.show()
plt.figure(figsize=(10,25))
sns.barplot(y='subcat1_name', x='price', data=train, estimator=median)
plt.title('median price of various subcategories')
plt.show()
def get_name_first(name):
    
    name =  re.sub('[^A-Za-z0-9]+', ' ', name) .split()
    if len(name):
            return name[0].lower()
    return ''
        
        
train['name_first'] = train['name'].apply(get_name_first)
test['name_first'] = test['name'].apply(get_name_first)
def transform_test(base_col, feat_col):
    '''
    Returns feat_col column of test data by mapping from the values already calculated for the same column in train data
    
    Parameters:
    
    base_col: column based on which a transform(count, mean, median) has been applied
    
    feat_col: desired feature column after applying the transform
    '''
    #Create dictionary of feature values from train data
    di = pd.Series(train[feat_col].values, index=train[base_col].values).to_dict()
    
    #Map test data using dictionary and fill NAs with 0
    
    if base_col == 'item_condition_id':
        #No chance of NAs
        return test[base_col].map(di).astype(float)
        
    return test[base_col].map(di).fillna(0)
train['name_first_count'] = train.groupby('name_first')['name_first'].transform('count')
test['name_first_count'] = transform_test('name_first', 'name_first_count')

train['gencat_name_count'] = train.groupby('gencat_name')['gencat_name'].transform('count')
test['gencat_name_count'] = transform_test('gencat_name', 'gencat_name_count')

train['subcat1_name_count'] = train.groupby('subcat1_name')['subcat1_name'].transform('count')
test['subcat1_name_count'] = transform_test('subcat1_name', 'subcat1_name_count')

train['subcat2_name_count'] = train.groupby('subcat2_name')['subcat2_name'].transform('count')
test['subcat2_name_count'] = transform_test('subcat2_name', 'subcat2_name_count')

train['brand_name_count'] = train.groupby('brand_name')['brand_name'].transform('count')
test['brand_name_count'] = transform_test('brand_name', 'brand_name_count')
train['NameLower'] = train.name.str.count('[a-z]')
train['DescriptionLower'] = train.item_description.str.count('[a-z]')
train['NameUpper'] = train.name.str.count('[A-Z]')
train['DescriptionUpper'] = train.item_description.str.count('[A-Z]')
train['name_len'] = train['name'].apply(lambda x: len(x))
train['des_len'] = train['item_description'].apply(lambda x: len(x))
train['name_desc_len_ratio'] = train['name_len']/train['des_len']
train['desc_word_count'] = train['item_description'].apply(lambda x: len(x.split()))
train['mean_des'] = train['item_description'].apply(lambda x: 0 if len(x) == 0 else float(len(x.split())) / len(x)) * 10
train['name_word_count'] = train['name'].apply(lambda x: len(x.split()))
train['mean_name'] = train['name'].apply(lambda x: 0 if len(x) == 0 else float(len(x.split())) / len(x))  * 10
train['desc_letters_per_word'] = train['des_len'] / train['desc_word_count']
train['name_letters_per_word'] = train['name_len'] / train['name_word_count']
train['NameLowerRatio'] = train['NameLower'] / train['name_len']
train['DescriptionLowerRatio'] = train['DescriptionLower'] / train['des_len']
train['NameUpperRatio'] = train['NameUpper'] / train['name_len']
train['DescriptionUpperRatio'] = train['DescriptionUpper'] / train['des_len']
test['NameLower'] = test.name.str.count('[a-z]')
test['DescriptionLower'] = test.item_description.str.count('[a-z]')
test['NameUpper'] = test.name.str.count('[A-Z]')
test['DescriptionUpper'] = test.item_description.str.count('[A-Z]')
test['name_len'] = test['name'].apply(lambda x: len(x))
test['des_len'] = test['item_description'].apply(lambda x: len(x))
test['name_desc_len_ratio'] = test['name_len']/test['des_len']
test['desc_word_count'] = test['item_description'].apply(lambda x: len(x.split()))
test['mean_des'] = test['item_description'].apply(lambda x: 0 if len(x) == 0 else float(len(x.split())) / len(x)) * 10
test['name_word_count'] = test['name'].apply(lambda x: len(x.split()))
test['mean_name'] = test['name'].apply(lambda x: 0 if len(x) == 0 else float(len(x.split())) / len(x))  * 10
test['desc_letters_per_word'] = test['des_len'] / test['desc_word_count']
test['name_letters_per_word'] = test['name_len'] / test['name_word_count']
test['NameLowerRatio'] = test['NameLower'] / test['name_len']
test['DescriptionLowerRatio'] = test['DescriptionLower'] / test['des_len']
test['NameUpperRatio'] = test['NameUpper'] / test['name_len']
test['DescriptionUpperRatio'] = test['DescriptionUpper'] / test['des_len']
from nltk.corpus import stopwords

RE_PUNCTUATION = '|'.join([re.escape(x) for x in string.punctuation])
s_words = {x: 1 for x in stopwords.words('english')} #converting to dictionary for fast look up
non_alphanumpunct = re.compile(u'[^A-Za-z0-9\.?!,; \(\)\[\]\'\"\$]+')
#https://www.kaggle.com/peterhurford/lgb-and-fm-18th-place-0-40604

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
train['NamePunctCount'] = train.name.str.count(RE_PUNCTUATION)
train['DescriptionPunctCount'] = train.item_description.str.count(RE_PUNCTUATION)
train['NamePunctCountRatio'] = train['NamePunctCount'] / train['name_word_count']
train['DescriptionPunctCountRatio'] = train['DescriptionPunctCount'] / train['desc_word_count']
train['NameDigitCount'] = train.name.str.count('[0-9]')
train['DescriptionDigitCount'] = train.item_description.str.count('[0-9]')
train['NameDigitCountRatio'] = train['NameDigitCount'] / train['name_word_count']
train['DescriptionDigitCountRatio'] = train['DescriptionDigitCount']/train['desc_word_count']
train['stopword_ratio_desc'] = train['item_description'].apply(lambda x: len([w for w in x.split() if w in s_words])) / train['desc_word_count']
train['num_sum'] = train['item_description'].apply(lambda x: sum([to_number(s) for s in x.split()])) 
train['weird_characters_desc'] = train['item_description'].str.count(non_alphanumpunct)
train['weird_characters_name'] = train['name'].str.count(non_alphanumpunct)
train['prices_count'] = train['item_description'].str.count('[rm]')
train['price_in_name'] = train['item_description'].str.contains('[rm]', regex=False).astype('category')
test['NamePunctCount'] = test.name.str.count(RE_PUNCTUATION)
test['DescriptionPunctCount'] = test.item_description.str.count(RE_PUNCTUATION)
test['NamePunctCountRatio'] = test['NamePunctCount'] / test['name_word_count']
test['DescriptionPunctCountRatio'] = test['DescriptionPunctCount'] / test['desc_word_count']
test['NameDigitCount'] = test.name.str.count('[0-9]')
test['DescriptionDigitCount'] = test.item_description.str.count('[0-9]')
test['NameDigitCountRatio'] = test['NameDigitCount'] / test['name_word_count']
test['DescriptionDigitCountRatio'] = test['DescriptionDigitCount']/test['desc_word_count']
test['stopword_ratio_desc'] = test['item_description'].apply(lambda x: len([w for w in x.split() if w in s_words])) / test['desc_word_count']
test['num_sum'] = test['item_description'].apply(lambda x: sum([to_number(s) for s in x.split()])) 
test['weird_characters_desc'] = test['item_description'].str.count(non_alphanumpunct)
test['weird_characters_name'] = test['name'].str.count(non_alphanumpunct)
test['prices_count'] = test['item_description'].str.count('[rm]')
test['price_in_name'] = test['item_description'].str.contains('[rm]', regex=False).astype('category')
train['brand_mean_price'] = train.groupby('brand_name')['price'].transform('mean')
test['brand_mean_price'] = transform_test('brand_name', 'brand_mean_price')

train['name_mean_price'] = train.groupby('name_first')['price'].transform('mean')
test['name_mean_price'] = transform_test('name_first', 'name_mean_price')

train['gencat_mean_price'] = train.groupby('gencat_name')['price'].transform('mean')
test['gencat_mean_price'] = transform_test('gencat_name', 'gencat_mean_price')

train['subcat1_mean_price'] = train.groupby('subcat1_name')['price'].transform('mean')
test['subcat1_mean_price'] = transform_test('subcat1_name', 'subcat1_mean_price')

train['subcat2_mean_price'] = train.groupby('subcat2_name')['price'].transform('mean')
test['subcat2_mean_price'] = transform_test('subcat2_name', 'subcat2_mean_price')

train['condition_mean_price'] = train.groupby('item_condition_id')['price'].transform('mean')
test['condition_mean_price'] = transform_test('item_condition_id', 'condition_mean_price')
train['brand_median_price'] = train.groupby('brand_name')['price'].transform('median')
test['brand_median_price'] = transform_test('brand_name', 'brand_median_price')

train['name_median_price'] = train.groupby('name_first')['price'].transform('median')
test['name_median_price'] = transform_test('name_first', 'name_median_price')

train['gencat_median_price'] = train.groupby('gencat_name')['price'].transform('median')
test['gencat_median_price'] = transform_test('gencat_name', 'gencat_median_price')

train['subcat1_median_price'] = train.groupby('subcat1_name')['price'].transform('median')
test['subcat1_median_price'] = transform_test('subcat1_name', 'subcat1_median_price')

train['subcat2_median_price'] = train.groupby('subcat2_name')['price'].transform('median')
test['subcat2_median_price'] = transform_test('subcat2_name', 'subcat2_median_price')

train['condition_median_price'] = train.groupby('item_condition_id')['price'].transform('median')
test['condition_median_price'] = transform_test('item_condition_id', 'condition_median_price')
train.drop(['name', 'item_description'], axis=1, inplace=True)
test.drop(['name', 'item_description'], axis=1, inplace=True)
print(train.shape, test.shape)
plt.figure(figsize=(18,18))

plt.subplot(3,3,1)
sns.regplot(x='brand_mean_price', y='price', data=train, scatter_kws={'alpha':0.3}, line_kws={'color':'orange'})
plt.title('brand_mean_price vs price(target)')

plt.subplot(3,3,2)
sns.regplot(x='gencat_mean_price', y='price', data=train, scatter_kws={'alpha':0.3}, line_kws={'color':'orange'})
plt.title('category_mean_price vs price(target)')

plt.subplot(3,3,3)
sns.regplot(x='subcat1_mean_price', y='price', data=train, scatter_kws={'alpha':0.3}, line_kws={'color':'orange'})
plt.title('subcategory_mean_price vs price(target)')

plt.subplot(3,3,4)
sns.regplot(x='subcat2_mean_price', y='price', data=train, scatter_kws={'alpha':0.3}, line_kws={'color':'orange'})
plt.title('subcategory_mean_price vs price(target)')

plt.subplot(3,3,5)
sns.regplot(x='condition_mean_price', y='price', data=train, scatter_kws={'alpha':0.3}, line_kws={'color':'orange'})
plt.title('condition_mean_price vs price(target)')

plt.subplot(3,3,6)
sns.regplot(x='brand_median_price', y='price', data=train, scatter_kws={'alpha':0.3}, line_kws={'color':'orange'})
plt.title('brand_median_price vs price(target)')

plt.subplot(3,3,7)
sns.regplot(x='subcat1_median_price', y='price', data=train, scatter_kws={'alpha':0.3}, line_kws={'color':'orange'})
plt.title('subcategory_median_price vs price(target)')

plt.subplot(3,3,8)
sns.regplot(x='subcat2_median_price', y='price', data=train, scatter_kws={'alpha':0.3}, line_kws={'color':'orange'})
plt.title('subcategory_median_price vs price(target)')

plt.subplot(3,3,9)
sns.regplot(x='name_median_price', y='price', data=train, scatter_kws={'alpha':0.3}, line_kws={'color':'orange'})
plt.title('name_median_price vs price(target)')
            
plt.show()
n_rows = train.shape[0]
train = train[train.preprocessed_name != ''].reset_index(drop=True)

print('Dropped {} rows'.format(n_rows - train.shape[0]))
n_rows = train.shape[0]
train = train[train.preprocessed_description != ''].reset_index(drop=True)

print('Dropped {} rows'.format(n_rows - train.shape[0]))

print('Shape of train data: ', train.shape)
from sklearn.model_selection import train_test_split

y_tr = np.log1p(train['price'])
train.drop(['price'], axis=1, inplace=True)

train_df, cv_df , y_train, y_cv = train_test_split(train, y_tr, test_size=0.1, random_state=42)

print('Train size: {}, CV size: {}, Test size: {}' .format(train_df.shape, cv_df.shape, test.shape))
del train, y_tr
gc.collect()
#Cleaning brand name before using count vectorizer
# Using same preprocessing as used earlier for categories: 'clean_cat()' function

train_df['brand_name'] = clean_cat(train_df['brand_name'].values)
cv_df['brand_name'] = clean_cat(cv_df['brand_name'].values)
test['brand_name'] = clean_cat(test['brand_name'].values)
vectorizer = CountVectorizer(lowercase=False, binary=True)
train_brand_oneHot = vectorizer.fit_transform(train_df['brand_name'].values)

cv_brand_oneHot = vectorizer.transform(cv_df['brand_name'].values)
test_brand_oneHot = vectorizer.transform(test['brand_name'].values)

print("Shape of matrices after one hot encoding")
print(train_brand_oneHot.shape, "\n", cv_brand_oneHot.shape, "\n", test_brand_oneHot.shape)
vectorizer = CountVectorizer(lowercase=False, binary=True)
train_gencat_oneHot = vectorizer.fit_transform(train_df['gencat_name'].values)

cv_gencat_oneHot = vectorizer.transform(cv_df['gencat_name'].values)
test_gencat_oneHot = vectorizer.transform(test['gencat_name'].values)

print("Shape of matrices after one hot encoding")
print(train_gencat_oneHot.shape, "\n", cv_gencat_oneHot.shape, "\n", test_gencat_oneHot.shape)
vectorizer = CountVectorizer(lowercase=False, binary=True)
train_subcat1_oneHot = vectorizer.fit_transform(train_df['subcat1_name'].values)

cv_subcat1_oneHot = vectorizer.transform(cv_df['subcat1_name'].values)
test_subcat1_oneHot = vectorizer.transform(test['subcat1_name'].values)

print("Shape of matrices after one hot encoding")
print(train_subcat1_oneHot.shape, "\n", cv_subcat1_oneHot.shape, "\n", test_subcat1_oneHot.shape)
vectorizer = CountVectorizer(lowercase=False, binary=True)
train_subcat2_oneHot = vectorizer.fit_transform(train_df['subcat2_name'].values)

cv_subcat2_oneHot = vectorizer.transform(cv_df['subcat2_name'].values)
test_subcat2_oneHot = vectorizer.transform(test['subcat2_name'].values)

print("Shape of matrices after one hot encoding")
print(train_subcat2_oneHot.shape, "\n", cv_subcat2_oneHot.shape, "\n", test_subcat2_oneHot.shape)
vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=3, max_features=250000)

train_name_tfidf = vectorizer.fit_transform(train_df['preprocessed_name'].values)

cv_name_tfidf = vectorizer.transform(cv_df['preprocessed_name'].values)
test_name_tfidf = vectorizer.transform(test['preprocessed_name'].values)

print("Shape of matrices after vectorization")
print(train_name_tfidf.shape, "\n", cv_name_tfidf.shape, "\n", test_name_tfidf.shape)
vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=5, max_features=500000)

train_description_tfidf = vectorizer.fit_transform(train_df['preprocessed_description'].values)

cv_description_tfidf = vectorizer.transform(cv_df['preprocessed_description'].values)
test_description_tfidf = vectorizer.transform(test['preprocessed_description'].values)

print("Shape of matrices after vectorization")
print(train_description_tfidf.shape, "\n", cv_description_tfidf.shape, "\n", test_description_tfidf.shape)
submission_df = pd.DataFrame(test['test_id'])
print(submission_df.shape)
submission_df.head()
cols = set(train_df.columns.values) - {'train_id'}
skip_cols = {'preprocessed_name', 'item_condition_id', 'brand_name',
  'shipping', 'preprocessed_description', 'gencat_name',
  'subcat1_name', 'subcat2_name', 'name_first', 'price_in_name'}

cols_to_normalize = cols - skip_cols
print("Normalizing following columns: ", cols_to_normalize)

def normalize(df):
    result1 = df.copy()
    for feature_name in df.columns:
        if (feature_name in cols_to_normalize):
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            result1[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result1

train_normalized = normalize(train_df)
cv_normalized = normalize(cv_df)
test_normalized = normalize(test)
del train_df, cv_df, test
gc.collect()
#Separating and storing all numerical features

X_tr = train_normalized[list(cols_to_normalize)]
X_val = cv_normalized[list(cols_to_normalize)]
X_te = test_normalized[list(cols_to_normalize)]

X_tr.head(2)
from scipy.sparse import csr_matrix

# Storing categorical features to sparse matrix

X_tr_cat = csr_matrix(pd.get_dummies(train_normalized[['item_condition_id', 'shipping', 'price_in_name']], sparse=True).values)

X_cv_cat = csr_matrix(pd.get_dummies(cv_normalized[['item_condition_id', 'shipping', 'price_in_name']], sparse=True).values)

X_te_cat = csr_matrix(pd.get_dummies(test_normalized[['item_condition_id', 'shipping', 'price_in_name']], sparse=True).values)

print(X_tr_cat.shape, X_cv_cat.shape, X_te_cat.shape)
del train_normalized, cv_normalized, test_normalized
gc.collect()
from scipy.sparse import hstack

# stack all categorical and text sparse matrices

train_sparse = hstack((train_brand_oneHot, train_gencat_oneHot, train_subcat1_oneHot, train_subcat2_oneHot, \
               train_name_tfidf, train_description_tfidf, X_tr_cat)).tocsr()

cv_sparse = hstack((cv_brand_oneHot, cv_gencat_oneHot, cv_subcat1_oneHot, cv_subcat2_oneHot, \
               cv_name_tfidf, cv_description_tfidf, X_cv_cat)).tocsr()

test_sparse = hstack((test_brand_oneHot, test_gencat_oneHot, test_subcat1_oneHot, test_subcat2_oneHot, \
               test_name_tfidf, test_description_tfidf, X_te_cat)).tocsr()
print(train_sparse.shape, cv_sparse.shape, test_sparse.shape)
# stack dense feature matrix with categorical and text vectors

X_train = hstack((X_tr.values, train_sparse)).tocsr()

X_cv = hstack((X_val.values, cv_sparse)).tocsr()

X_test = hstack((X_te.values, test_sparse)).tocsr()
print('Train size: {}, CV size: {}, Test size: {}' .format(X_train.shape, X_cv.shape, X_test.shape))
del vectorizer
del train_brand_oneHot, train_gencat_oneHot, train_subcat1_oneHot, train_subcat2_oneHot, \
            train_name_tfidf, train_description_tfidf, X_tr_cat

del cv_brand_oneHot, cv_gencat_oneHot, cv_subcat1_oneHot, cv_subcat2_oneHot, cv_name_tfidf, cv_description_tfidf, X_cv_cat

del test_brand_oneHot, test_gencat_oneHot, test_subcat1_oneHot, test_subcat2_oneHot, \
               test_name_tfidf, test_description_tfidf, X_te_cat
gc.collect()
from sklearn.metrics import mean_squared_error as mse
from math import sqrt
from sklearn.linear_model import Ridge
alpha = [1, 2, 3, 3.5, 4, 4.5, 5, 6, 7] 
cv_rmsle_array=[] 
for i in tqdm(alpha):
    model = Ridge(solver="sag", random_state=42, alpha=i)
    model.fit(X_train, y_train)
    preds_cv = model.predict(X_cv)
    cv_rmsle_array.append(sqrt(mse(y_cv, preds_cv)))

for i in range(len(cv_rmsle_array)):
    print ('RMSLE for alpha = ',alpha[i],'is',cv_rmsle_array[i])
    
best_alpha = np.argmin(cv_rmsle_array)

fig, ax = plt.subplots()
ax.plot(alpha, cv_rmsle_array)
ax.scatter(alpha, cv_rmsle_array)
for i, txt in enumerate(np.round(cv_rmsle_array,3)):
    ax.annotate((alpha[i],np.round(txt,3)), (alpha[i],cv_rmsle_array[i]))

plt.title("Cross Validation Error for each alpha")
plt.xlabel("Alpha")
plt.ylabel("Error")
plt.show()
print("Best alpha: ",  alpha[best_alpha])
model = Ridge(solver="sag", random_state=42, alpha=alpha[best_alpha])
model.fit(X_train, y_train)
ridge_preds_tr = model.predict(X_train)
ridge_preds_cv = model.predict(X_cv)
ridge_preds_te = model.predict(X_test)

print('Train RMSLE:', sqrt(mse(y_train, ridge_preds_tr)))

ridge_rmsle = sqrt(mse(y_cv, ridge_preds_cv))
print("Cross validation RMSLE: ", ridge_rmsle)
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB(alpha=0.01)
model.fit(X_train, y_train>= 4)

mnb_preds_tr = model.predict_proba(X_train)[:, 1]
mnb_preds_cv = model.predict_proba(X_cv)[:, 1]
mnb_preds_te = model.predict_proba(X_test)[:, 1]
# from sklearn.feature_selection.univariate_selection import SelectKBest, f_regression

fselect = SelectKBest(f_regression, k=48000)
train_features = fselect.fit_transform(train_sparse, y_train)

cv_features = fselect.transform(cv_sparse)
test_features = fselect.transform(test_sparse)
print('Shapes after SelectKBest:', train_features.shape, cv_features.shape, test_features.shape)
# stack feature matrix with Ridge, MNB model predictions, engineered features
X_train = hstack((X_tr.values, ridge_preds_tr.reshape(-1,1), mnb_preds_tr.reshape(-1,1), train_features)).tocsr()

X_cv = hstack((X_val.values, ridge_preds_cv.reshape(-1,1), mnb_preds_cv.reshape(-1,1), cv_features)).tocsr()

X_test = hstack((X_te.values, ridge_preds_te.reshape(-1,1), mnb_preds_te.reshape(-1,1), test_features)).tocsr()

print('Train size: {}, CV size: {}, Test size: {}' .format(X_train.shape, X_cv.shape, X_test.shape))
del train_features, cv_features
gc.collect()
print('Time taken: ', time.time()-start_time)
snapshot2 = tracemalloc.take_snapshot()
top_stats = snapshot2.compare_to(snapshot1, 'lineno')

print("[ Top 10 ]")
for stat in top_stats[:10]:
    print(stat)
submission_df['price'] = np.exp(ridge_preds_te) - 1

submission_df.to_csv('ridge_submission.csv', index=False)
scipy.sparse.save_npz("cv_final.npz", X_cv)
np.save('y_cv', y_cv)

del X_cv, y_cv
gc.collect()
scipy.sparse.save_npz("train_final.npz", X_train)
np.save('y_train', y_train)

del X_train, y_train
gc.collect()
scipy.sparse.save_npz("test_final.npz", X_test)

del X_test
gc.collect()