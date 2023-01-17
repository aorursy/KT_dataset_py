## All the Libraries used



import os

import numpy as np

import json

import pandas as pd

from collections import Counter

import csv

import matplotlib.pyplot as plt

from sklearn import neighbors, metrics

##import seaborn as sns

import statsmodels.api as sm

import statsmodels.formula.api as smf

from sklearn import feature_selection, linear_model

import sklearn

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

import string

from sklearn.decomposition import TruncatedSVD







pd.set_option('display.max_rows', 10)

pd.set_option('display.notebook_repr_html', True)

pd.set_option('display.max_columns', 10)



%matplotlib inline



plt.style.use('ggplot')

data = pd.DataFrame.from_csv('../input/train.tsv', sep='\t')

test_df = pd.DataFrame.from_csv('../input/test.tsv', sep='\t')



data.head()
df = data.copy()

test = test_df.copy()
df['brand_name'] = df.brand_name.apply(str)

df['name'] = df.name.apply(str)

df['item_description'] = df.item_description.apply(str)

del df.index.name



## test

test['brand_name'] = test.brand_name.apply(str)

test['name'] = test.name.apply(str)

test['item_description'] = test.item_description.apply(str)

del test.index.name
## whether name is nan or not



def notnan(row):

    if row != 'nan':

        return 1

    else:

        return 0



df['have_name'] = df.name.apply(lambda row : notnan(row))



##

test['have_name'] = test.name.apply(lambda row : notnan(row))

## splitting the category by '/'



def split(row):

    try:

        text = row

        txt1, txt2, txt3 = text.split('/')

        return txt1, txt2, txt3

    except:

        return np.nan, np.nan, np.nan

    

df['cat1'], df['cat2'], df['cat3'] =  zip(*df.category_name.apply(lambda val: split(val)))

test['cat1'], test['cat2'], test['cat3'] =  zip(*test.category_name.apply(lambda val: split(val)))



## transform categorical data into label



df['cat1'] = df.cat1.apply(str)

df['cat2'] = df.cat2.apply(str)

df['cat3'] = df.cat3.apply(str)



label = sklearn.preprocessing.LabelEncoder()

label.fit(df['cat1'])

df['cat1num'] = label.transform(df['cat1'])



label.fit(df['cat2'])

df['cat2num'] = label.transform(df['cat2'])



label.fit(df['cat3'])

df['cat3num'] = label.transform(df['cat3'])



df = df.drop('category_name', axis = 1)



##test

test['cat1'] = test.cat1.apply(str)

test['cat2'] = test.cat2.apply(str)

test['cat3'] = test.cat3.apply(str)



label = sklearn.preprocessing.LabelEncoder()

label.fit(test['cat1'])

test['cat1num'] = label.transform(test['cat1'])



label.fit(test['cat2'])

test['cat2num'] = label.transform(test['cat2'])



label.fit(test['cat3'])

test['cat3num'] = label.transform(test['cat3'])



test = test.drop('category_name', axis = 1)
## Whether brand_name is in name and or item_description

df['in_name'] = np.where(df['name'] >= df['brand_name'], 1, 0)

df['in_desc'] = np.where(df['item_description'] >= df['brand_name'], 1, 0)



##test

test['in_name'] = np.where(test['name'] >= test['brand_name'], 1, 0)

test['in_desc'] = np.where(test['item_description'] >= test['brand_name'], 1, 0)





## Whether item has brand_name

df['have_brand'] = df.brand_name.apply(lambda row : notnan(row))



##test

test['have_brand'] = test.brand_name.apply(lambda row : notnan(row))





## labeling brand_name



label.fit(df['brand_name'])

df['brand_namenum'] = label.transform(df['brand_name'])



##test

label.fit(test['brand_name'])

test['brand_namenum'] = label.transform(test['brand_name'])
## parsing item_description



## whether item_descprition has description



def havedesc(row):

    if row == 'No description yet':

        a = 0

    else:

        a = 1

    return a



df['have_desc'] = df.item_description.apply(lambda row : havedesc(row))



test['have_desc'] = test.item_description.apply(lambda row : havedesc(row))



tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1,3))



df_tfidf = tfidf_vec.fit_transform(df['item_description'].values.tolist())



test_tfidf = tfidf_vec.fit_transform(test['item_description'].values.tolist())
n_comp = 40

svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')

svd_obj.fit(df_tfidf)

train_svd = pd.DataFrame(svd_obj.transform(df_tfidf))



train_svd.columns = ['svd_item_'+str(i) for i in range(n_comp)]



svd_obj.fit(test_tfidf)

test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))



test_svd.columns = ['svd_item_'+str(i) for i in range(n_comp)]

test_svd
df = pd.concat([df, train_svd], axis=1)



test = pd.concat([test, test_svd], axis=1)
## model = sklearn.ensemble.GradientBoostingClassifier()
##clas = RandomForestClassifier()

##clas.fit(X_test,y_test)
df.to_csv('parsed.csv')
##test.to_csv('parsedtest.csv')
import numpy as np

import timeit

import math



# vectorized error calc

def rmsle(y, y0):

    assert len(y) == len(y0)

    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))



#looping error calc

def rmsle_loop(y, y_pred):

    assert len(y) == len(y_pred)

    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]

    return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5



# create random values to demonstrate speed difference

y1 = np.random.rand(1000000)

y2 = np.random.rand(1000000)



t0 = timeit.default_timer()

err = rmsle_loop(y1,y2)

elapsed = timeit.default_timer()-t0

print('Using loops:')

print('RMSLE: {:.3f}\nTime: {:.3f} seconds'.format(err, elapsed))



t0 = timeit.default_timer()

err = rmsle(y1,y2)

elapsed = timeit.default_timer()-t0

print('\nUsing vectors:')

print('RMSLE: {:.3f}\nTime: {:.3f} seconds'.format(err, elapsed))