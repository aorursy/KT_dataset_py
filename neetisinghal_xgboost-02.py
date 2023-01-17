# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
#xgboost
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import urllib        #for url stuff
import re            #for processing regular expressions
import datetime      #for datetime operations
import calendar      #for calendar for datetime operations
import time          #to get the system time
import scipy         #for other dependancies
from sklearn.cluster import KMeans # for doing K-means clustering
#from haversine import haversine # for calculating haversine distance
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
# Input data files are available in the "../input/" directory.
train = pd.read_csv("../input/train.tsv", sep = "\t")
test = pd.read_csv("../input/test.tsv", sep = "\t")
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


print(train.isnull().sum())
train.fillna(0, inplace=True)
print(train.isnull().sum())
train_copy = train.copy()
test_copy = test.copy()

def split_cat(text):
    try: return text.split("/")
    except: return ("No Label", "No Label", "No Label")
    
    train['general_category'], train['subcat_1'], train['subcat_2'] = \
zip(*train['category_name'].apply(lambda x: split_cat(x)))
train.head()


y = np.log(train['price'].values + 1)
print("difference in features are {}".format(np.setdiff1d(train_copy.columns, test_copy.columns)))
print("")
do_not_use_for_training = ['subcat_1','test_id','subcat_2','general_category','train_id','name', 'category_name', 'brand_name', 'price', 'item_description']
feature_names = [f for f in train.columns if f not in do_not_use_for_training]
print("features for training {}.".format(feature_names))
print("")
print("Total features are {}.".format(len(feature_names)))

from sklearn.model_selection import train_test_split
X_train, X_v, y_train, y_v = train_test_split(train[feature_names].values, y, test_size=0.2, random_state=1987)

print("Splitting is done")

data_train = xgb.DMatrix(X_train, label=y_train)
data_valid = xgb.DMatrix(X_v, label=y_v)


data_test = xgb.DMatrix(test[feature_names].values)
watchlist = [(data_train, 'train'), (data_valid, 'valid')]
start = time.time()
xgb_par = {'min_child_weight': 20, 'eta': 0.05, 'colsample_bytree': 0.5, 'max_depth': 15,
            'subsample': 0.9, 'lambda': 2.0, 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,
            'eval_metric': 'rmse', 'objective': 'reg:linear'}

model_1 = xgb.train(xgb_par, data_train, 80, watchlist, early_stopping_rounds=20, maximize=False, verbose_eval=20)
print('Modeling RMSLE %.5f' % model_1.best_score)
end = time.time()
print("Time taken in training is {}.".format(end - start))



start = time.time()
yvalid = model_1.predict(data_valid)
ytest = model_1.predict(data_test)
end = time.time()


start = time.time()
fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
sns.distplot(yvalid, ax=ax[0], color='red', label='Validation')
sns.distplot(ytest, ax=ax[1], color='green', label='Test')
ax[0].legend(loc=0)
ax[1].legend(loc=0)
plt.show()
end = time.time()
print("Time taken by above cell is {}.".format((end-start)))

if test.shape[0] == ytest.shape[0]:
    print('ok') 


test['price'] = np.exp(ytest) - 1
test[['test_id', 'price']].to_csv('neeti_submission_1.csv', index=False)
print("done")


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.