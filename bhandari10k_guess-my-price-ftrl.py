# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import string

import gc

import os

import re



from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.preprocessing import LabelBinarizer

from scipy.sparse import csr_matrix, hstack

from sklearn.linear_model import Ridge

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from wordbatch.models import FTRL, FM_FTRL



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



def split_cat(cats):

    cat2 = "no category 2"

    cat3 = "no category 3"

    if pd.notnull(cats):

        all_cats = cats.split('/')

        if len(all_cats) > 0:

            cat3 = all_cats[2]

            cat2 = all_cats[0]

    return cat2.lower(), cat3.lower()



def handle_missing_inplace(dataset):

    dataset['category_name'].fillna(value='missing', inplace=True)

    dataset['brand_name'].fillna(value='missing', inplace=True)

    dataset['item_description'].fillna(value='missing', inplace=True)



def fill_brands(dataset, test):

    brands = pd.concat([dataset['brand_name'], test['brand_name']], axis=0).unique().astype(str)

    print(pd.isnull(dataset['brand_name']).sum())

    brands_str = re.compile(r'\b(?:%s)\b' % '|'.join(brands))

    dataset['brand_name'] = dataset.apply(lambda row: row['brand_name'] if pd.notnull(row['brand_name']) or brands_str.match(row['name']) is None else brands_str.match(row['name']).group(0), axis=1)

    test['brand_name'] = test.apply(lambda row: row['brand_name'] if pd.notnull(row['brand_name']) or brands_str.match(row['name']) is None else brands_str.match(row['name']).group(0), axis=1)

    print(pd.isnull(dataset['brand_name']).sum())

    del brands

    del brands_str

    gc.collect()



def to_lower(dataset):

    

    dataset['category_name'] = dataset['category_name'].str.lower()

    dataset['brand_name'] = dataset['brand_name'].str.lower()

    dataset['brand_name'] = dataset['brand_name'].str.replace("'","")

    dataset['name'] = dataset['name'].str.lower()

    dataset['name'] = dataset['name'].str.replace("'","")

    dataset['item_description'] = dataset['item_description'].str.lower()



def process_dataset(dataset, test):

    to_lower(dataset)

    to_lower(test)

    fill_brands(dataset, test)

    handle_missing_inplace(dataset)

    handle_missing_inplace(test)
train = pd.read_table('../input/guess-my-price/train.tsv', engine='c')

train = train[train.price != 0].reset_index(drop=True)

test = pd.read_table('../input/guess-my-price/test.tsv', engine='c')

print(test.shape)

test_top = test.copy()

print(test_top.shape)

del test

gc.collect()

test_top_predsa: pd.DataFrame = test_top[['train_id']].reset_index(drop=True)

train[['cat2','cat3']] = pd.DataFrame(train.category_name.apply(split_cat).tolist(), columns = ['cat2','cat3'])

test_top[['cat2','cat3']] = pd.DataFrame(test_top.category_name.apply(split_cat).tolist(), columns = ['cat2','cat3'])

count_cat3 = pd.DataFrame(train['cat3'].value_counts())

count_cutoff = 50 #250

count_cat3 = count_cat3[count_cat3.cat3 >= count_cutoff]

train['cat3'] = train.apply(lambda row: row['cat3'] if row['cat3'] in count_cat3.index else row['cat2'], axis=1)

test_top['cat3'] = test_top.apply(lambda row: row['cat3'] if row['cat3'] in count_cat3.index else row['cat2'], axis=1)



process_dataset(train, test_top)

cats = train['cat3'].unique().astype(str)

nrow_train = train.shape[0]

y = np.log1p(train['price'])

max_cat = y.max()

min_cat = y.min()

merge: pd.DataFrame = pd.concat([train, test_top])

merge['name'] = merge['name'].astype(str) + ' ' + merge['category_name'].astype(str) + ' ' + merge['brand_name']

tv = TfidfVectorizer(max_features=None,

                     ngram_range=(1, 3), min_df=2, token_pattern=r'(?u)\b\w+\b')

X_name = tv.fit_transform(merge['name'])

X_description = tv.fit_transform(merge['item_description'])

X_category = tv.fit_transform(merge['category_name'])



lb = LabelBinarizer(sparse_output=True)

X_brand = lb.fit_transform(merge['brand_name'])

X_dummies = csr_matrix(pd.get_dummies(merge[['item_condition_id', 'shipping']], sparse=True).values)

sparse_merge = hstack((X_dummies, X_description, X_brand, X_name, X_category)).tocsr()

X = sparse_merge[:nrow_train]

X_test_top = sparse_merge[nrow_train:]
del merge

del sparse_merge

del X_dummies

del X_description

del X_brand

del X_name

gc.collect()
def rmsle(y, y0):

    assert len(y) == len(y0)

    return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y0), 2)))



train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.05, random_state=100)

model = FTRL(alpha=0.01, beta=0.1, L1=0.00001, L2=1.0, D=X.shape[1], iters=47, inv_link="identity", threads=1)



model.fit(train_X, train_y)

preds = model.predict(X=valid_X)

print("FM_FTRL dev RMSLE:", rmsle(np.expm1(valid_y), np.expm1(preds)))
test_preds = model.predict(X=X_test_top)

test_preds = np.clip(test_preds, min_cat, max_cat)

submission = pd.DataFrame({'train_id':test_top.train_id})

submission['price'] = np.expm1(test_preds)

submission.to_csv('submission.csv', index=False) 

submission.head()