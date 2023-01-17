# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



from collections import Counter



from nltk import word_tokenize

from nltk.stem.porter import PorterStemmer

porter_stemmer = PorterStemmer()



from nltk.stem import SnowballStemmer

s_stemmer = SnowballStemmer('english')



from nltk.util import ngrams



import Levenshtein 



from tqdm import tqdm, tqdm_pandas



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv')

df_test  = pd.read_csv('../input/test.csv')
print(df_train.info())

print(df_test.info())
df_train.head()
df_test.head()
df_train.groupby('is_duplicate').id.count().plot.bar()
#Метрика Жоккара кастомированная к строкам#

def JokkarMetric(x, y): 

    x = set(word_tokenize(x))

    y = set(word_tokenize(y))

    return (len(x.intersection(y)) +.0) / (len(x.union(y)) + .0)
#Чистка

df_train.question1 = df_train.question1.map(lambda x : str(x).lower())

df_train.question2 = df_train.question2.map(lambda x : str(x).lower())
df_train["Jokkar"] = df_train.apply(func=lambda x : JokkarMetric(x.question1, x.question2), axis=1)

df_train.head()
sns.distplot(df_train[df_train.is_duplicate==0].Jokkar)

sns.distplot(df_train[df_train.is_duplicate==1].Jokkar)
def L1Metric(x, y): 

    x = Counter(word_tokenize(x))

    y = Counter(word_tokenize(y))

    return np.abs(len(x) - len(y)+ .0)
df_train["L1Metric"] = df_train.apply(func=lambda x : L1Metric(x.question1, x.question2), axis=1)

df_train.head()
sns.distplot(df_train[df_train.is_duplicate==0].L1Metric)

sns.distplot(df_train[df_train.is_duplicate==1].L1Metric)
def InterMetric(x, y): 

    x = set(word_tokenize(x))

    y = set(word_tokenize(y))

    return len(x.intersection(y)) + .0
df_train["InterMetric"] = df_train.apply(func=lambda x : InterMetric(x.question1, x.question2), axis=1)

df_train.head()
sns.distplot(df_train[df_train.is_duplicate==0].InterMetric)

sns.distplot(df_train[df_train.is_duplicate==1].InterMetric)
def BigramJokkarMetric(x, y):

    x = set(ngrams(word_tokenize(x), 2))

    y = set(ngrams(word_tokenize(y), 2))

    return (len(x.intersection(y)) +.01) / (len(x.union(y)) + .01)
df_train["BigramJokkar"] = df_train.apply(func=lambda x : BigramJokkarMetric(x.question1, x.question2), axis=1)

df_train.head()
sns.distplot(df_train[df_train.is_duplicate==0].BigramJokkar)

sns.distplot(df_train[df_train.is_duplicate==1].BigramJokkar)
def QuesMetric(x, y):

    x = word_tokenize(x)

    y = word_tokenize(y)

    return (x[0] == y[0]) + .0
df_train['QuestEq'] = df_train.apply(func = lambda x : QuesMetric(x.question1, x.question2), axis=1)

df_train.head()
def Quest1Metric(x, y):

    x = word_tokenize(x)

    y = word_tokenize(y)

    return ((len(x) > 1) and (len(y) > 1) and (x[1] == y[1]))+.0
df_train['Quest1Eq'] = df_train.apply(func = lambda x : Quest1Metric(x.question1, x.question2), axis=1)

df_train.head()
sns.factorplot(x="QuestEq", y="Quest1Eq", hue="is_duplicate", data=df_train, kind="bar")
def LevenMetric(x, y):

    x = x.replace(' ', '')

    y = y.replace(' ', '')

    return Levenshtein.distance(x, y)
df_train["LevenMetric"] = df_train.apply(func=lambda x : LevenMetric(x.question1, x.question2), axis=1)

df_train.head()
sns.distplot(df_train[df_train.is_duplicate==0].LevenMetric)

sns.distplot(df_train[df_train.is_duplicate==1].LevenMetric)
def Spec1(x, y):

    return (1 ^ ('?' in x) ^ ('?' in y)) + .0

def Spec2(x, y):

    return (1 ^ ('[math]' in x) ^ ('[math]' in y)) + .0

def Spec3(x, y):

    return (1 ^ ('.' in x) ^ ('.' in y)) + .0
df_train['Spec1'] = df_train.apply(func=lambda x : Spec1(x.question1, x.question2), axis=1)

df_train['Spec2'] = df_train.apply(func=lambda x : Spec2(x.question1, x.question2), axis=1)

df_train['Spec3'] = df_train.apply(func=lambda x : Spec3(x.question1, x.question2), axis=1)
train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)

test_qs = pd.Series(df_test['question1'].tolist() + df_test['question2'].tolist()).astype(str)
dist_train = train_qs.apply(len)

dist_test = test_qs.apply(len)



pal = sns.color_palette()

plt.figure(figsize=(15, 10))

plt.hist(dist_train, bins=200, range=[0, 200], color=pal[2], normed=True, label='train')

plt.hist(dist_test, bins=200, range=[0, 200], color=pal[1], normed=True, alpha=0.5, label='test')

plt.title('Normalised histogram of character count in questions', fontsize=15)

plt.legend()

plt.xlabel('Number of characters', fontsize=15)

plt.ylabel('Probability', fontsize=15)



print('mean-train {:.2f} std-train {:.2f} mean-test {:.2f} std-test {:.2f} max-train {:.2f} max-test {:.2f}'.format(dist_train.mean(), 

                          dist_train.std(), dist_test.mean(), dist_test.std(), dist_train.max(), dist_test.max()))
from collections import Counter



# If a word appears only once, we ignore it completely (likely a typo)

# Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller

def get_weight(count, eps=10000, min_count=2):

    if count < min_count:

        return 0

    else:

        return 1 / (count + eps)



eps = 5000 

words = (" ".join(train_qs)).lower().split()

counts = Counter(words)

weights = {word: get_weight(count) for word, count in counts.items()}
from nltk.corpus import stopwords



stops = set(stopwords.words("english"))



def tfidf_word_match_share(row):

    q1words = {}

    q2words = {}

    for word in str(row['question1']).lower().split():

        if word not in stops:

            q1words[word] = 1

    for word in str(row['question2']).lower().split():

        if word not in stops:

            q2words[word] = 1

    if len(q1words) == 0 or len(q2words) == 0:

        # The computer-generated chaff includes a few questions that are nothing but stopwords

        return 0

    

    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]

    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

    

    R = np.sum(shared_weights) / np.sum(total_weights)

    return R
plt.figure(figsize=(15, 5))

df_train['tfidf'] = df_train.apply(tfidf_word_match_share, axis=1, raw=True)

plt.hist(df_train['tfidf'][df_train['is_duplicate'] == 0].fillna(0), bins=20, normed=True, label='Not Duplicate')

plt.hist(df_train['tfidf'][df_train['is_duplicate'] == 1].fillna(0), bins=20, normed=True, alpha=0.7, label='Duplicate')

plt.legend()

plt.title('Label distribution over tfidf_word_match_share', fontsize=15)

plt.xlabel('word_match_share', fontsize=15)
def word_match_share(row):

    q1words = {}

    q2words = {}

    for word in str(row['question1']).lower().split():

        if word not in stops:

            q1words[word] = 1

    for word in str(row['question2']).lower().split():

        if word not in stops:

            q2words[word] = 1

    if len(q1words) == 0 or len(q2words) == 0:

        # The computer-generated chaff includes a few questions that are nothing but stopwords

        return 0

    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]

    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]

    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))

    return R



plt.figure(figsize=(15, 5))

df_train['word_match'] = df_train.apply(word_match_share, axis=1, raw=True)

plt.hist(df_train['word_match'][df_train['is_duplicate'] == 0], bins=20, normed=True, label='Not Duplicate')

plt.hist(df_train['word_match'][df_train['is_duplicate'] == 1], bins=20, normed=True, alpha=0.7, label='Duplicate')

plt.legend()

plt.title('Label distribution over word_match_share', fontsize=15)

plt.xlabel('word_match_share', fontsize=15)
%%time

df_test.question1 = df_test.question1.map(lambda x : str(x).lower())

df_test.question2 = df_test.question2.map(lambda x : str(x).lower())
%%time

df_test["Jokkar"] = df_test.apply(func=lambda x : JokkarMetric(x.question1, x.question2), axis=1)
%%time

df_test["L1Metric"] = df_test.apply(func=lambda x : L1Metric(x.question1, x.question2), axis=1)
%%time

df_test["InterMetric"] = df_test.apply(func=lambda x : InterMetric(x.question1, x.question2), axis=1)
%%time

df_test["BigramJokkar"] = df_test.apply(func=lambda x : BigramJokkarMetric(x.question1, x.question2), axis=1)
%%time

df_test['QuestEq'] = df_test.apply(func = lambda x : QuesMetric(x.question1, x.question2), axis=1)
%%time

df_test['Quest1Eq'] = df_test.apply(func = lambda x : Quest1Metric(x.question1, x.question2), axis=1)
%%time

df_test["LevenMetric"] = df_test.apply(func=lambda x : LevenMetric(x.question1, x.question2), axis=1)
%%time

df_test['Spec1'] = df_test.apply(func=lambda x : Spec1(x.question1, x.question2), axis=1)
%%time

df_test['Spec2'] = df_test.apply(func=lambda x : Spec2(x.question1, x.question2), axis=1)
%%time

df_test['Spec3'] = df_test.apply(func=lambda x : Spec3(x.question1, x.question2), axis=1)
%%time

df_test['tfidf'] = df_test.apply(tfidf_word_match_share, axis=1, raw=True)
%%time

df_test['word_match'] = df_test.apply(word_match_share, axis=1, raw=True)
df_test.head()
x_train = df_train.drop(['question1', 'question2', 'qid1', 'qid2', 'is_duplicate', 'id', 'LevenMetric'], axis=1).values

y_train = df_train.is_duplicate.values

x_test = df_test.drop(['question1', 'question2', 'test_id', 'LevenMetric'], axis=1).values
print(len(df_train[df_train.is_duplicate==0]), len(df_train[df_train.is_duplicate==1]))

df_train = pd.concat([df_train[df_train.is_duplicate==1], df_train]).sample(n = len(df_train))
import xgboost as xgb
# Параметры

params = {}

params['objective'] = 'binary:logistic'

params['eval_metric'] = 'logloss'

params['eta'] = 0.02

params['max_depth'] = 4
d_train = xgb.DMatrix(x_train, label=y_train)

watchlist = [(d_train, 'train')]
bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)
d_test = xgb.DMatrix(x_test)
y_test = bst.predict(d_test)
sub = pd.DataFrame()

sub['test_id'] = df_test['test_id']

sub['is_duplicate'] = y_test

sub.to_csv('sub.csv', index=False)
