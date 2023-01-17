# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.preprocessing import LabelBinarizer

from scipy.sparse import csr_matrix, hstack

from sklearn.linear_model import Ridge

from sklearn.model_selection import KFold

import string

import gc

import os

import re



import warnings

warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def rmsle(y, preds):

    return np.sqrt(np.square(np.log(preds + 1) - np.log(y + 1)).mean())



def treat_missing(dataset):

    dataset['name'].fillna(value='unavailable', inplace=True)

    dataset['category_name'].fillna(value='unavailable', inplace=True)

    dataset['brand_name'].fillna(value='unavailable', inplace=True)

    dataset['item_description'].fillna(value='unavailable', inplace=True)



def treat_brand(dataset):

    brands = dataset['brand_name'].unique().astype(str)

    print(pd.isnull(dataset['brand_name']).sum())

    brands_str = re.compile(r'\b(?:%s)\b' % '|'.join(brands))

    dataset['brand_name'] = dataset.apply(lambda row: row['brand_name'] if pd.notnull(row['brand_name']) or brands_str.match(row['name']) is None else brands_str.match(row['name']).group(0), axis=1)    

    print(pd.isnull(dataset['brand_name']).sum())

    del brands

    del brands_str

    gc.collect()



def treat_case(dataset):

    dataset['category_name'] = dataset['category_name'].str.lower()

    dataset['brand_name'] = dataset['brand_name'].str.lower()

    dataset['name'] = dataset['name'].str.lower()

    dataset['item_description'] = dataset['item_description'].str.lower()

    

def treat_punctuations(dataset):

    dataset['brand_name'] = dataset['brand_name'].str.replace("'","")

    dataset['name'] = dataset['name'].str.replace("'","")

    dataset['item_description'] = dataset['item_description'].str.replace("'","")



def process_dataset(dataset):

    treat_case(dataset)

    treat_punctuations(dataset)

    treat_brand(dataset)

    treat_missing(dataset)
# Read Datasets

train = pd.read_csv('../input/guess-my-price/train.tsv',sep='\t')

test = pd.read_csv('../input/guess-my-price/test.tsv',sep='\t')

train.drop(columns=['random'],inplace=True,axis=1)

test.drop(columns=['random'],inplace=True,axis=1)

train = train.drop(train[train.price <= 1.0].index).reset_index(drop=True)



print(train.shape)

print(test.shape)
# Splitting out category columns

def treat_category(cats):

    cat1 = "no category 1"

    cat2 = "no category 2"

    cat3 = "no category 3"

    if pd.notnull(cats):

        all_cats = cats.split('/')

        if len(all_cats) > 0:

            cat3 = all_cats[2]

            cat2 = all_cats[1]

            cat1 = all_cats[0]

    return cat1.lower(), cat2.lower(), cat3.lower()



test_preds_overall = pd.DataFrame(test[['train_id']].reset_index(drop=True))



train[['cat1','cat2','cat3']] = pd.DataFrame(train.category_name.apply(treat_category).tolist(), columns = ['cat1','cat2','cat3'])

test[['cat1','cat2','cat3']] = pd.DataFrame(test.category_name.apply(treat_category).tolist(), columns = ['cat1','cat2','cat3'])

train.head()
# If count of category 3 is less than 100, take category 2. If category 2 is less than 100, take category 1

count_cat3 = pd.DataFrame(train['cat3'].value_counts())

print('Top 10',count_cat3[0:10])

print('\n')

print('Bottom 10',count_cat3[-10:])

print('\n')



count_cat2 = pd.DataFrame(train['cat2'].value_counts())

print('Top 10',count_cat2[0:10])

print('\n')

print('Bottom 10',count_cat2[-10:])

print('\n')



count_cat1 = pd.DataFrame(train['cat1'].value_counts())

print('Top 10',count_cat1[0:10])

print('\n')

print('Bottom 10',count_cat1[-10:])

print('\n')



count_cutoff = 100

count_cat3 = count_cat3[count_cat3.cat3 >= count_cutoff]

count_cat2 = count_cat2[count_cat2.cat2 >= count_cutoff]



train['cat3'] = train.apply(lambda row: row['cat3'] if row['cat3'] in count_cat3.index else (row['cat2'] if row['cat2'] in count_cat2.index else row['cat1']), axis=1)

test['cat3'] = test.apply(lambda row: row['cat3'] if row['cat3'] in count_cat3.index else (row['cat2'] if row['cat2'] in count_cat2.index else row['cat1']), axis=1)
# Process Dataframe

nrow_train = train.shape[0]

train_test_combined = train.append(test,sort=True).reset_index(drop=True)

print(train_test_combined.shape)



process_dataset(train_test_combined)

train = train_test_combined[:nrow_train]

test = train_test_combined[nrow_train:]

test.drop(['price'], axis = 1, inplace = True)
# Combining name + category_name + brand_name. Implementing TF-IDF separately on these columns makes a very large matrix

train_test_combined['name'] = train_test_combined['name'].astype(str) + ' ' + train_test_combined['category_name'].astype(str) + ' ' + train_test_combined['brand_name']



tv = TfidfVectorizer(max_features=None,

                     ngram_range=(1, 3), min_df=2, token_pattern=r'(?u)\b\w+\b') # Regex source: https://stackoverflow.com/questions/35043085/what-does-u-do-in-a-regex

X_name = tv.fit_transform(train_test_combined['name'])

X_description = tv.fit_transform(train_test_combined['item_description'])

X_category = tv.fit_transform(train_test_combined['category_name'])



lb = LabelBinarizer(sparse_output=True)

X_brand = lb.fit_transform(train_test_combined['brand_name'])

X_dummies = csr_matrix(pd.get_dummies(train_test_combined[['item_condition_id', 'shipping']], sparse=True).values)

sparse_train_test_combined = hstack((X_dummies, X_description, X_brand, X_name, X_category)).tocsr()

X = sparse_train_test_combined[:nrow_train]

X_test = sparse_train_test_combined[nrow_train:]

y = np.log1p(train['price'])
X.shape
del train_test_combined

del sparse_train_test_combined

del X_dummies

del X_description

del X_brand

del X_category

del X_name

gc.collect()
# Fitting a ridge reggression model at overall level data

model = Ridge(solver="sag", fit_intercept=True, random_state=369)

model.fit(X, y)

test_preds_overall['Overall_Price'] = model.predict(X=X_test)

test_preds_overall.head()
train.head()
test_preds_cat = pd.DataFrame(columns=['train_id','Category_Price'])

cats = train['cat3'].unique().astype(str)

count_threshold = 100



for cat in cats:

    warnings.filterwarnings('ignore')

    train_cat = train[train.cat3 == cat].reset_index(drop=True)

    print(cat, train_cat.shape[0])

    

    # If count of category 3 is less than threshold, we will not build model

    if train_cat.shape[0] < count_threshold:

        continue

        

    test_cat = test[test.cat3 == cat].reset_index(drop=True)

    cat_preds = pd.DataFrame(test_cat[['train_id']].reset_index(drop=True))

    nrow_cat = train_cat.shape[0]

    y = np.log1p(train_cat["price"])

    max_cat = y.max()

    min_cat = y.min()

    

    train_test_combined =  pd.DataFrame(pd.concat([train_cat, test_cat], axis = 0))

    del train_cat

    del test_cat

    

    # Taking count of category name to add some diversity before ensemble

    cv = CountVectorizer()

    X_category = cv.fit_transform(train_test_combined['category_name'])

    

    tv = TfidfVectorizer(max_features=None, ngram_range=(1, 3), min_df=2,token_pattern=r'(?u)\b\w+\b') # Regex source: https://stackoverflow.com/questions/35043085/what-does-u-do-in-a-regex

    train_test_combined['name'] = train_test_combined['name'] + ' ' + train_test_combined['brand_name']

    X_name = tv.fit_transform(train_test_combined['name'])

    X_description = tv.fit_transform(train_test_combined['item_description'])

    

    lb = LabelBinarizer(sparse_output=True)

    X_brand = lb.fit_transform(train_test_combined['brand_name'])

    X_dummies = csr_matrix(pd.get_dummies(train_test_combined[['item_condition_id', 'shipping']], sparse=True).values)

    sparse_train_test_combined = hstack((X_dummies, X_description, X_brand, X_name, X_category)).tocsr()

    X = sparse_train_test_combined[:nrow_cat]

    X_test = sparse_train_test_combined[nrow_cat:]

    

    del train_test_combined

    del sparse_train_test_combined

    del X_dummies

    del X_category

    del X_description

    del X_brand

    del X_name

    gc.collect()



    model = Ridge(solver="saga", fit_intercept=False, random_state=459)

    model.fit(X, y)

    cat_preds['Category_Price'] = model.predict(X=X_test)

    cat_preds['Category_Price'] = np.clip(cat_preds['Category_Price'], min_cat, max_cat)

    test_preds_cat = pd.DataFrame(pd.concat([test_preds_cat, cat_preds], axis = 0))
# Joining overall predictions with category level predictions

preds_all = test_preds_overall.merge(test_preds_cat,on='train_id',how='left')

# If category level predictions are null, take the score from overall predictions

preds_all['Both'] = preds_all.apply(lambda row: row['Overall_Price'] if pd.isnull(row['Category_Price']) else row['Category_Price'], axis=1)

print(preds_all['Both'].head(20))

# Ensemble of overall predictions and category level predictions

preds_all['price'] = np.expm1((preds_all['Both'] + preds_all['Overall_Price']) / 2)

preds_all[['train_id','price']].to_csv("Submission.csv",index=False)

print(preds_all['Overall_Price'].min())



print(preds_all.head())