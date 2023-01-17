import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xgboost as xgb
%matplotlib inline
import seaborn as sb
train = pd.read_table('../input/train.tsv', engine='c')
test = pd.read_table('../input/test.tsv', engine='c')
train.head(1)
train.info()
# Plot the histogram of the price
bin_values = np.arange(start=0, stop=200, step=1)
train.hist(column='price',bins=bin_values, figsize=[14,6])
# Show the first ten brand_name
train.brand_name.value_counts().loc[lambda x: x.index != ''][:10]
import time
start = time.time()
plt.hist(train.price, bins=300,range=(0,250), normed=False)
plt.show()
end = time.time()
print("Time is %f" %(end-start))
plt.hist(np.log(train.price), bins=300, range=(0,6), normed=False)
plt.show()
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
import scipy

train["category_name"] = train["category_name"].fillna("Other")
train["brand_name"] = train["brand_name"].fillna("unknown")
test["category_name"] = test["category_name"].fillna("Other")
test["brand_name"] = test["brand_name"].fillna("unknown")
# Split the category and lable them
def cat_split(row):
    try:
        text = row
        txt1, txt2, txt3 = text.split('/')
        return txt1, txt2, txt3
    except:
        return np.nan, np.nan, np.nan


train["cat_1"], train["cat_2"], train["cat_3"] = zip(*train.category_name.apply(lambda val: cat_split(val)))
test["cat_1"], test["cat_2"], test["cat_3"] = zip(*test.category_name.apply(lambda val: cat_split(val)))
train.head()
train.cat_1.value_counts()[:10]
# making dictionaries for different categories 
keys = train.cat_1.unique().tolist() + test.cat_1.unique().tolist()
keys = list(set(keys)) # use set() to get the unique key
values = range(0,keys.__len__())
cat1_dict = dict(zip(keys, values))
cat1_dict

keys2 = train.cat_2.unique().tolist() + test.cat_2.unique().tolist()
keys2 = list(set(keys2))
values2 = list(range(keys2.__len__()))
cat2_dict = dict(zip(keys2, values2))

keys3 = train.cat_3.unique().tolist() + test.cat_3.unique().tolist()
keys3 = list(set(keys3))
values3 = list(range(keys3.__len__()))
cat3_dict = dict(zip(keys3, values3))
# code the categories
def cat_lable(row):
    txt1 = row['cat_1']
    txt2 = row['cat_2']
    txt3 = row['cat_3']
    return cat1_dict[txt1], cat2_dict[txt2], cat3_dict[txt3]

train["cat_1_label"], train["cat_2_label"], train["cat_3_lable"] \
= zip(*train.apply(lambda val: cat_lable(val), axis =1))
# zip(*) means unzip

test["cat_1_label"], test["cat_2_label"], test["cat_3_lable"] \
= zip(*test.apply(lambda val: cat_lable(val), axis =1))
def if_catname(row):
    """function to give if brand name is there or not"""
    if row == row:
        return 1
    else:
        return 0
    
train['if_cat'] = train.category_name.apply(lambda row : if_catname(row))
test['if_cat'] = test.category_name.apply(lambda row : if_catname(row))
train.head()
# brand name related features 
def if_brand(row):
    """function to give if brand name is there or not"""
    if row == row:
        return 1
    else:
        return 0
    
train['if_brand'] = train.brand_name.apply(lambda row : if_brand(row))
test['if_brand'] = test.brand_name.apply(lambda row : if_brand(row))
train.head()
# makinfg brand name dict features 
keys = train.brand_name.dropna().unique()
values = list(range(keys.__len__()))
brand_dict = dict(zip(keys, values))

def brand_label(row):
    """function to assign brand label"""
    try:
        return brand_dict[row]
    except:
        return np.nan

train['brand_label'] = train.brand_name.apply(lambda row: brand_label(row))
test['brand_label'] = test.brand_name.apply(lambda row: brand_label(row))
train.head()
def if_description(row):
    """function to say if description is present or not"""
    if row == 'No description yet':
        a = 0
    else:
        a = 1
    return a

train['is_description'] = train.item_description.apply(lambda row : if_description(row))
test['is_description'] = test.item_description.apply(lambda row : if_description(row))
train.head()
print(train.shape[0])
train = train.loc[train.item_description == train.item_description]
test = test.loc[test.item_description == test.item_description]
train = train.loc[train.name == train.name]
test = test.loc[test.name == test.name]
print(train.shape[0])
print("Dropped records where item description was nan")
import pandas as pd  #pandas for using dataframe and reading csv 
import numpy as np   #numpy for vector operations and basic maths 
import urllib        #for url stuff
import re            #for processing regular expressions
import datetime      #for datetime operations
import calendar      #for calendar for datetime operations
import time          #to get the system time
import scipy         #for other dependancies
from sklearn.cluster import KMeans # for doing K-means clustering
from haversine import haversine # for calculating haversine distance
import math          #for basic maths operations
import seaborn as sns #for making plots
import matplotlib.pyplot as plt # for plotting
import os                # for os commands
import nltk
from nltk.corpus import stopwords
import string
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import ensemble, metrics, model_selection, naive_bayes
tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1,1))
full_tfidf = tfidf_vec.fit_transform(train['item_description'].values.tolist() + test['item_description'].values.tolist())
train_tfidf = tfidf_vec.transform(train['item_description'].values.tolist())
test_tfidf = tfidf_vec.transform(test['item_description'].values.tolist())
n_comp = 50
svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
svd_obj.fit(full_tfidf)
train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))
train_svd.columns = ['svd_item_'+str(i) for i in range(n_comp)]
test_svd.columns = ['svd_item_'+str(i) for i in range(n_comp)]
train_df = pd.concat([train, train_svd], axis=1)
test_df = pd.concat([test, test_svd], axis=1)
pd.set_option('display.max_columns',100)
train_df
train_df.fillna(0, inplace=True)
test_df.fillna(0, inplace=True)
print(train_df.isnull().sum())
train = train_df.copy()
test = test_df.copy()
print("Difference of features in train and test are {}".format(np.setdiff1d(train.columns, test.columns)))
print("")
do_not_use_for_training = ['cat_1','test_id','cat_2','cat_3','train_id','name', 'category_name', 'brand_name', 'price', 'item_description']
feature_names = [f for f in train.columns if f not in do_not_use_for_training]
print("We will be using following features for training {}.".format(feature_names))
print("")
print("Total number of features are {}.".format(len(feature_names)))
y = np.log(train['price'].values + 1)
from sklearn.model_selection import train_test_split
Xtr, Xv, ytr, yv = train_test_split(train[feature_names].values, y, test_size=0.2, random_state=1987)
dtrain = xgb.DMatrix(Xtr, label=ytr)
dvalid = xgb.DMatrix(Xv, label=yv)
dtest = xgb.DMatrix(test[feature_names].values)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
xgb_par = {'min_child_weight': 20, 'eta': 0.05, 'colsample_bytree': 0.5, 'max_depth': 15,
            'subsample': 0.9, 'lambda': 2.0, 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,
            'eval_metric': 'rmse', 'objective': 'reg:linear'}
model_1 = xgb.train(xgb_par, dtrain, 500, watchlist, early_stopping_rounds=50, maximize=False, verbose_eval=50)
print('Modeling RMSLE %.5f' % model_1.best_score)
pred = model_1.predict(dtest)
test['price'] = np.expm1(pred)
test[["test_id", "price"]].to_csv('submission2.csv', index=False)