# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

df_train.head()
#Чистка

df_train.question1 = df_train.question1.map(lambda x : str(x).lower())

df_train.question2 = df_train.question2.map(lambda x : str(x).lower())
train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)

test_qs = pd.Series(df_test['question1'].tolist() + df_test['question2'].tolist()).astype(str)
train_qs.head()
from nltk.corpus import stopwords



stops = set(stopwords.words("english"))



def WordMatch(row):

    q1 = set(str(row['question1']).split()).difference(stops)

    q2 = set(str(row['question2']).split()).difference(stops)

    

    if len(q1) == 0 or len(q2) == 0:

        return 0

    

    inter1 = q1.difference(q2)

    inter2 = q2.difference(q1)

    return (len(inter1) + len(inter2))/(len(q1) + len(q2) + .0)





df_train['WordMatch'] = df_train.apply(WordMatch, axis=1, raw=True)
sns.distplot(df_train[df_train.is_duplicate==0].WordMatch, kde=False)

sns.distplot(df_train[df_train.is_duplicate==1].WordMatch, kde=False)
from nltk.corpus import stopwords



stops = set(stopwords.words("english"))



def Jakkar(row):

    q1 = set(str(row['question1']).split()).difference(stops)

    q2 = set(str(row['question2']).split()).difference(stops)

    

    if len(q1) == 0 or len(q2) == 0:

        return 0

    

    inter = q1.intersection(q2)

    un = q1.union(q2)

    return 1 - (len(inter))/(len(un) + .0)





df_train['Jakkar'] = df_train.apply(Jakkar, axis=1, raw=True)
sns.distplot(df_train[df_train.is_duplicate==0].Jakkar, kde=False)

sns.distplot(df_train[df_train.is_duplicate==1].Jakkar, kde=False)
tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 1))



tfidf_txt = pd.Series(test_qs.tolist() + train_qs.tolist())

tfidf.fit_transform(tfidf_txt)
from collections import Counter



def get_weight(count, eps=10000, min_count=2):

    if count < min_count:

        return 0.0

    else:

        return 1.0 / (count + eps)





words = (" ".join(train_qs)).lower().split()

counts = Counter(words)

weights = {word: get_weight(count) for word, count in counts.items()}
def WeightMatch(row):

    q1 = set(str(row['question1']).split()).difference(stops)

    q2 = set(str(row['question2']).split()).difference(stops)

    

    if len(q1) == 0 or len(q2) == 0:

        return 0

    

    shared_weights = [weights.get(w, 0) for w in q1.intersection(q2)]

    total_weights = [weights.get(w, 0) for w in q1.union(q2)]

    

    R = np.sum(shared_weights) / (np.sum(total_weights) + .0)

    return R



df_train['WeightMatch'] = df_train.apply(WeightMatch, axis=1, raw=True)
sns.distplot(df_train[df_train.is_duplicate==0].WeightMatch, kde=False)

sns.distplot(df_train[df_train.is_duplicate==1].WeightMatch, kde=False)
x_test = pd.DataFrame()



x_test['WordMatch'] = df_test.apply(WordMatch, axis=1, raw=True)

x_test['Jakkar'] = df_test.apply(Jakkar, axis=1, raw=True)
x_test['WeightMatch'] = df_test.apply(WeightMatch, axis=1, raw=True)
x_train = df_train.drop(['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate'], axis=1)

y_train = df_train['is_duplicate'].values
pos_train = x_train[y_train == 1]

neg_train = x_train[y_train == 0]



# Now we oversample the negative class

# There is likely a much more elegant way to do this...

p = 0.165

scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1

while scale > 1:

    neg_train = pd.concat([neg_train, neg_train])

    scale -=1

neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])

print(len(pos_train) / (len(pos_train) + len(neg_train)))



x_train = pd.concat([pos_train, neg_train])

y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()

del pos_train, neg_train
import xgboost as xgb



# Set our parameters for xgboost

params = {}

params['objective'] = 'binary:logistic'

params['eval_metric'] = 'logloss'

params['eta'] = 0.02

params['max_depth'] = 4



d_train = xgb.DMatrix(x_train, label=y_train)



watchlist = [(d_train, 'train')]



bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)
d_test = xgb.DMatrix(x_test)

p_test = bst.predict(d_test)



sub = pd.DataFrame()

sub['test_id'] = df_test['test_id']

sub['is_duplicate'] = p_test

sub.to_csv('simple.csv', index=False)