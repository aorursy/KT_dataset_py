# Imports

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline

# machine learning
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation
import xgboost as xgb
# get homedepot & test csv files as a DataFrame
homedepot_df     = pd.read_csv('../input/train.csv', encoding="ISO-8859-1")
descriptions_df  = pd.read_csv('../input/product_descriptions.csv')
test_df          = pd.read_csv('../input/test.csv', encoding="ISO-8859-1")

# preview the data
homedepot_df.head()
homedepot_df.info()
print("----------------------------")
test_df.info()
# Convert product_title & product_description to cunt_words_in_title & cunt_words_in_description, 
# where cunt_words_in_title counts how many times a word in search_term appeared in product_title, 
# and so for cunt_words_in_description

# Now, instead of dealing with strings, we have numbers that indicate 
# the relevance between the search_term and product title & description.

# Step 1: Merge descriptions dataframe
homedepot_df = pd.merge(homedepot_df, descriptions_df, how='left', on='product_uid')
test_df      = pd.merge(test_df, descriptions_df, how='left', on='product_uid')

homedepot_df.head()
# Step 2: Normalize the title and description using stemming algorithms(i.e. SnowballStemmer)
# @see https://xapian.org/docs/stemming.html
# @see http://www.nltk.org/api/nltk.stem.html

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('english')

def str_stemmer(s):
    return " ".join([stemmer.stem(word) for word in s.lower().split()])

homedepot_df['search_term']         = homedepot_df['search_term'].apply(str_stemmer)
homedepot_df['product_title']       = homedepot_df['product_title'].apply(str_stemmer)
homedepot_df['product_description'] = homedepot_df['product_description'].apply(str_stemmer)

test_df['search_term']         = test_df['search_term'].apply(str_stemmer)
test_df['product_title']       = test_df['product_title'].apply(str_stemmer)
test_df['product_description'] = test_df['product_description'].apply(str_stemmer)
# Step 3: Count how many times each word in search_term appeared in product_title & product_description

def count_words(strs):
    str_words, str_search = strs
    return sum(int(str_search.find(word) >= 0) for word in str_words.split())
    
homedepot_df['cunt_words_in_title']       = homedepot_df[['product_title', 'search_term']].apply(count_words,axis=1)
homedepot_df['cunt_words_in_description'] = homedepot_df[['product_description', 'search_term']].apply(count_words,axis=1)

test_df['cunt_words_in_title']       = test_df[['product_title', 'search_term']].apply(count_words,axis=1)
test_df['cunt_words_in_description'] = test_df[['product_description', 'search_term']].apply(count_words,axis=1)
# Step 4: Drop unnecessary columns

homedepot_df.drop(['product_title','product_description','search_term'], inplace=True, axis=1)
test_df.drop(['product_title','product_description','search_term'], inplace=True, axis=1)
# define training and testing sets

X_train = homedepot_df.drop(["id","relevance"],axis=1)
Y_train = homedepot_df["relevance"]
X_test  = test_df.drop("id",axis=1).copy()
# Linear Regression

lreg = LinearRegression()

lreg.fit(X_train, Y_train)

Y_pred = lreg.predict(X_test)

lreg.score(X_train, Y_train)
# Xgboost 

params = {"objective": "reg:linear", "max_depth": 10}

T_train_xgb = xgb.DMatrix(X_train, Y_train)
X_test_xgb  = xgb.DMatrix(X_test)

gbm = xgb.train(params, T_train_xgb, 20)
Y_pred = gbm.predict(X_test_xgb)
# Create submission

submission = pd.DataFrame()
submission["id"]        = test_df["id"]
submission["relevance"] = Y_pred
submission["relevance"][submission["relevance"] < 1] = 1
submission["relevance"][submission["relevance"] > 3] = 3

submission.to_csv('homedepot.csv', index=False)