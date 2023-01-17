import nltk

import string

import re

import numpy as np

import pandas as pd

import pickle

#import lda



import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="white")



from nltk.stem.porter import *

from nltk.tokenize import word_tokenize, sent_tokenize

from nltk.corpus import stopwords

from sklearn.feature_extraction import stop_words



from collections import Counter

from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import LatentDirichletAllocation,TruncatedSVD



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

%matplotlib inline



import bokeh.plotting as bp

from bokeh.models import HoverTool, BoxSelectTool

from bokeh.models import ColumnDataSource

from bokeh.plotting import figure, show, output_notebook

#from bokeh.transform import factor_cmap



# import warnings

# warnings.filterwarnings('ignore')

# import logging

# logging.getLogger("lda").setLevel(logging.WARNING)

from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb
import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv('../input/train.tsv',delimiter='\t')

train.head()
train.describe()
test = pd.read_csv('../input/test.tsv',delimiter='\t')

test.head()
train.info()
np.log1p(train['price']).hist();
plt.scatter(x=train.index,y=np.log1p(train['price']));
(train['price']==0).sum()
train['price'] = np.log1p(train['price'])
len(train)
# vectorized error calc

def rmsle(y, y0):

    assert len(y) == len(y0)

    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))



#looping error calc

def rmsle_loop(y, y_pred):

    assert len(y) == len(y_pred)

    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]

    return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5
sns.countplot(x='item_condition_id',data=train);
sns.countplot(x='shipping',data=train);
train = train[train['price']!=0]

len(train)
def wordCount(text):

    # convert to lower case and strip regex

    try:

         # convert to lower case and strip regex

        text = text.lower()

        regex = re.compile('[' +re.escape(string.punctuation) + '0-9\\r\\t\\n]')

        txt = regex.sub(" ", text)

        # tokenize

        # words = nltk.word_tokenize(clean_txt)

        # remove words in stop words

        words = [w for w in txt.split(" ") \

                 if not w in stop_words.ENGLISH_STOP_WORDS and len(w)>3]

        return len(words)

    except: 

        return 0
# remove missing values in item description

train = train[pd.notnull(train['item_description'])]
train['name_len'] = train['name'].apply(lambda x:len(x))

train['item_description_len'] = train['item_description'].apply(lambda x:len(x))

test['name_len'] = test['name'].apply(lambda x:len(x))

test['item_description_len'] = test['item_description'].apply(lambda x:len(x))
train['item_description_len'].head()
len(test)
# reference: BuryBuryZymon at https://www.kaggle.com/maheshdadhich/i-will-sell-everything-for-free-0-55

def split_cat(text):

    try: return text.split("/")

    except: return ("No Label", "No Label", "No Label")
train['general_cat'], train['subcat_1'], train['subcat_2'] = zip(*train['category_name'].apply(lambda x: split_cat(x)))

train.head()
# repeat the same step for the test set

test['general_cat'], test['subcat_1'], test['subcat_2'] = zip(*test['category_name'].apply(lambda x: split_cat(x)))
train["brand_name"] = train["brand_name"].fillna('None')

train["category_name"] = train["category_name"].fillna('None')

test["brand_name"] = test["brand_name"].fillna('None')

test["category_name"] = test["category_name"].fillna('None')
lis = ['brand_name','category_name']

for i in lis:

    total = train[i].append(test[i])

    le = LabelEncoder()

    le.fit(total.astype(str).values)

    train[i] = le.transform(train[i].astype(str))

    test[i] = le.transform(test[i].astype(str))

    print(i)
import time

start = time.time()

tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1,1))

full_tfidf = tfidf_vec.fit_transform(train['item_description'].values.tolist() + test['item_description'].values.tolist())

train_tfidf = tfidf_vec.transform(train['item_description'].values.tolist())

test_tfidf = tfidf_vec.transform(test['item_description'].values.tolist())



n_comp = 40

svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')

svd_obj.fit(full_tfidf)

train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))

test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))

    

train_svd.columns = ['svd_item_'+str(i) for i in range(n_comp)]

test_svd.columns = ['svd_item_'+str(i) for i in range(n_comp)]

train = pd.concat([train, train_svd], axis=1)

test = pd.concat([test, test_svd], axis=1)

end = time.time()

print("time taken {}".format(end - start))
train.columns
train = train.dropna(axis=0,how='any')
features = ['item_condition_id','category_name', 'brand_name','shipping','name_len',

            'item_description_len',

            'svd_item_0', 'svd_item_1', 'svd_item_2', 'svd_item_3', 'svd_item_4',

            'svd_item_5', 'svd_item_6', 'svd_item_7', 'svd_item_8', 'svd_item_9',

            'svd_item_10', 'svd_item_11', 'svd_item_12', 'svd_item_13',

            'svd_item_14', 'svd_item_15', 'svd_item_16', 'svd_item_17',

            'svd_item_18', 'svd_item_19', 'svd_item_20', 'svd_item_21',

            'svd_item_22', 'svd_item_23', 'svd_item_24', 'svd_item_25',

            'svd_item_26', 'svd_item_27', 'svd_item_28', 'svd_item_29',

            'svd_item_30', 'svd_item_31', 'svd_item_32', 'svd_item_33',

            'svd_item_34', 'svd_item_35', 'svd_item_36', 'svd_item_37',

            'svd_item_38', 'svd_item_39']
len(train[features]),len(train['price'])
model = lgb.LGBMRegressor(learning_rate=0.1,n_estimators=500,max_depth=7)

model.fit(train[features],train['price'])

pred = model.predict(test[features])
pred = np.absolute(pred)

pred = np.exp(pred)-1
sub = pd.DataFrame()

sub['test_id'] = test['test_id']

sub['price'] = pred

sub.to_csv('lgb.csv', index=False)

print('writting is done.')