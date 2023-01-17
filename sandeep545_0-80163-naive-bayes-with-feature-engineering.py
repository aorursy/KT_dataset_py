# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# from google.colab import drive

# drive.mount('/content/drive')
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')



test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')



submission = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
df.head()
print('Shape:', df.shape)
print('value counts of target\n')

df.target.value_counts()
sns.countplot(df.target)

plt.show()
df.info()
col = df.columns



for i in range(len(col)-1):

    if df[col[i]].isnull().sum() != 0:

        c = df[col[i]].isnull().sum()

        print('\n{} column has {} null values'.format(col[i], c))
print('\nPercentage of null values in keyword:', round((df.keyword.isnull().sum() / df.shape[0]) * 100, 3))

print('\nPercentage of null values in location:', round((df.location.isnull().sum() / df.shape[0]) * 100, 3))
print('Number of unique values in keyword:', df.keyword.nunique(), '\n')

print('Number of unique values in location:', df.location.nunique(), '\n')
print('\n-> Preprocessed text data:\n')



for i in range(0, df.shape[0], 300):

    print('\n{}'.format(i)+':', df['text'][i])

# Importing libraries



import re

from nltk.corpus import stopwords

from nltk.stem.snowball import SnowballStemmer

from tqdm import tqdm



# Create an instance for SnowballStemmer

ss = SnowballStemmer('english')
# Defining a function to convert short words like couldn't to full word could not

def short_form(full_form):

    

    full_form = full_form.lower()      

    

    full_form = re.sub(r"won't", "will not", full_form)

    full_form = re.sub(r"wouldn't", "would not", full_form)

    full_form = re.sub(r"can't", "can not", full_form)

    full_form = re.sub(r"don't", "don not", full_form)

    full_form = re.sub(r"shouldn't", "should not", full_form)

    full_form = re.sub(r"couldn't", "could not", full_form)

    full_form = re.sub(r"\'re", " are", full_form)

    full_form = re.sub(r"\'s", " is", full_form)

    full_form = re.sub(r"\'d", " would", full_form)

    full_form = re.sub(r"\'ll", " will", full_form)

    full_form = re.sub(r"\'ve", " have", full_form)

    full_form = re.sub(r"\'m", " am", full_form)

  

    return full_form



# To remove URL

def url(ur):

    ur = re.sub(r"http\S+", '', ur)

    return ur



# Defining a function to remove punctuations, numbers, stopwords and get stem of words

def punc(pun):

    pun = re.sub('[^a-zA-Z]', ' ', pun)

    pun = pun.lower()

    pun = pun.split()

    pun = [ss.stem(sw) for sw in pun if sw not in stopwords.words('english')]

    pun = ' '.join(pun)

    return pun
import nltk

nltk.download("stopwords")
test.head()
col = test.columns



for i in range(len(col)-1):

    if test[col[i]].isnull().sum() != 0:

        c = test[col[i]].isnull().sum()

        print('\n{} column has {} null values'.format(col[i], c))
print('\nPercentage of null values in keyword:', round((test.keyword.isnull().sum() / test.shape[0]) * 100, 3))

print('\nPercentage of null values in location:', round((test.location.isnull().sum() / test.shape[0]) * 100, 3))
print('Number of unique values in keyword:', test.keyword.nunique(), '\n')

print('Number of unique values in location:', test.location.nunique(), '\n')
import copy
dfe = copy.deepcopy(df)
# Filling na with previous data point

# Before that we will fill null values with string 'null' for better detection



dfe['keyword'].fillna('nul', inplace = True)



for i in range(32):

    if dfe['keyword'][i] != 'nul':

        dfe['keyword'][0] = dfe['keyword'][i]
for i in range(dfe.shape[0]):

    if dfe['keyword'][i] == 'nul':

        dfe['keyword'][i] = dfe['keyword'][i-1]
# Location column



# Filling na with previous data point

# Before that we will fill null values with string 'null' for better detection



dfe['location'].fillna('nul', inplace = True)



for i in range(32):

    if dfe['location'][i] != 'nul':

        dfe['location'][0] = dfe['location'][i]
for i in range(dfe.shape[0]):

    if dfe['location'][i] == 'nul':

        dfe['location'][i] = dfe['location'][i-1]
from tqdm import tqdm



loc_train_clean = []



for i, s in enumerate(tqdm(dfe['location'].values)):

    

    u = url(s)

    sf = short_form(u)

    pu = punc(sf)

    loc_train_clean.append(pu)
from tqdm import tqdm



text_train_clean = []



for i, s in enumerate(tqdm(dfe['text'].values)):

    

    u = url(s)

    sf = short_form(u)

    pu = punc(sf)

    text_train_clean.append(pu)
print('\n-> Preprocessed text data:\n')



dfe['location'] = loc_train_clean



dfe['text'] = text_train_clean
dfc = copy.deepcopy(dfe)



# dfc.to_csv('train_clean.csv', index = False)
teste = copy.deepcopy(test)
# Filling na with previous data point

# Before that we will fill null values with string 'null' for better detection



teste['keyword'].fillna('nul', inplace = True)



for i in range(20):

    if teste['keyword'][i] != 'nul':

        teste['keyword'][0] = teste['keyword'][i]
for i in range(teste.shape[0]):

    if teste['keyword'][i] == 'nul':

        teste['keyword'][i] = teste['keyword'][i-1]
# Location column



# Filling na with previous data point

# Before that we will fill null values with string 'null' for better detection



teste['location'].fillna('nul', inplace = True)



for i in range(32):

    if teste['location'][i] != 'nul':

        teste['location'][0] = teste['location'][i]
for i in range(test.shape[0]):

    if teste['location'][i] == 'nul':

        teste['location'][i] = teste['location'][i-1]




from tqdm import tqdm



loc_test_clean = []



for i, s in enumerate(tqdm(teste['location'].values)):

    

    u = url(s)

    sf = short_form(u)

    pu = punc(sf)

    loc_test_clean.append(pu)
from tqdm import tqdm



text_test_clean = []



for i, s in enumerate(tqdm(teste['text'].values)):

    

    u = url(s)

    sf = short_form(u)

    pu = punc(sf)

    text_test_clean.append(pu)
print('\n-> Preprocessed text data:\n')



teste['location'] = loc_test_clean



teste['text'] = text_test_clean
testc = copy.deepcopy(teste)



# testc.to_csv('test_clean.csv', index = False)
dfe['key_loc_text'] = dfe['keyword'] + dfe['location'] + dfe['text']



teste['key_loc_text'] = teste['keyword'] + teste['location'] + teste['text']
from sklearn.preprocessing import LabelEncoder



lb = LabelEncoder()



dfe['keyword'] = lb.fit_transform(dfe['keyword'])



teste['keyword'] = lb.transform(teste['keyword'])
# Import CountVectorizer library

from sklearn.feature_extraction.text import CountVectorizer



# Create an instance

# Bi-gram

cv_klt = CountVectorizer(ngram_range = (1, 2))



# Fit and transform train data

tr_klt_b = cv_klt.fit_transform(dfe['key_loc_text'])



# Transform test data

te_klt_b = cv_klt.transform(teste['key_loc_text'])
# Import normalize library

from sklearn.preprocessing import normalize



# Normalize train data

tr_klt_n = normalize(tr_klt_b)



# Normalize test data

te_klt_n = normalize(te_klt_b)
# Import CountVectorizer library

from sklearn.feature_extraction.text import CountVectorizer



# Create an instance

# Bi-gram

cv_l = CountVectorizer(ngram_range = (1, 2))



# Fit and transform train data

tr_l_b = cv_l.fit_transform(dfe['location'])



# Transform test data

te_l_b = cv_l.transform(teste['location'])
# Import normalize library

from sklearn.preprocessing import normalize



# Normalize train data

tr_l_n = normalize(tr_l_b)



# Normalize test data

te_l_n = normalize(te_l_b)
# Import CountVectorizer library

from sklearn.feature_extraction.text import CountVectorizer



# Create an instance

# Bi-gram

cv_t = CountVectorizer(ngram_range = (1, 2))



# Fit and transform train data

tr_t_b = cv_t.fit_transform(dfe['text'])



# Transform test data

te_t_b = cv_t.transform(teste['text'])
# Import normalize library

from sklearn.preprocessing import normalize



# Normalize train data

tr_t_n = normalize(tr_t_b)



# Normalize test data

te_t_n = normalize(te_t_b)
tr_key = dfe['keyword'].values.reshape(-1, 1)



te_key = teste['keyword'].values.reshape(-1, 1)
from scipy.sparse import hstack



x_tr_e = hstack((tr_key, tr_klt_n, tr_l_n, tr_t_n))



x_te_e = hstack((te_key, te_klt_n, te_l_n, te_t_n))
y = dfe['target']
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import GridSearchCV



nbe = MultinomialNB()



param = {'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 10, 100, 1000, 10000]}





clf_nbe = GridSearchCV(estimator = nbe, param_grid = param, scoring = 'f1', cv = 4)



clf_nbe.fit(x_tr_e, y)
print('\n-> Best score:', clf_nbe.best_score_, '\n')

print('*'*50, '\n')



print('\n-> Best estimators:', clf_nbe.best_estimator_)
nbe_pre = clf_nbe.predict(x_te_e)



sub_nb_e = copy.deepcopy(submission)
sub_nb_e['target'] = nbe_pre



# sub_nb_e.to_csv('sub_nbe_2.csv', index = False)
import pandas as pd
# sub_nb_e = pd.read_csv('/content/drive/My Drive/Tweets Disaster or not/nlp-getting-started/sub_nbe_2.csv')
print("\nShape of test predicted data:", sub_nb_e.shape, '\n')
print("\nHead of 10 of test predicted data:\n")



sub_nb_e.head(10)