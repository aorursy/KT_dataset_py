# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
%load_ext autoreload

%autoreload 2



%matplotlib inline

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from IPython.display import display

from collections import Counter

from sklearn import metrics

from sklearn.metrics import log_loss

import matplotlib.pyplot as plt



from time import time

from scipy.stats import randint as sp_randint

from sklearn.model_selection import RandomizedSearchCV
df = pd.read_csv('../input/train.csv', low_memory=False)
df.head()
def countword(q):

    text = q.split(' ')

    count = Counter(text)

    return count
df['question1'] = df['question1'].astype('str')

df['question2'] = df['question2'].astype('str')

df['counter1'] = df['question1'].apply(lambda x:countword(x))

df['counter2'] = df['question2'].apply(lambda x:countword(x))

df['len1'] = df['question1'].apply(len)

df['len2'] = df['question2'].apply(len)

df['wordnum1'] = df['counter1'].apply(len)

df['wordnum2']= df['counter2'].apply(len)

df['sameWordNum'] = df.apply(lambda x:len(x['counter1'] & x['counter2']),axis=1)

df['len_diff'] = df.apply(lambda x:abs(x['len1']-x['len2']), axis=1)

df['word_num_diff'] = df.apply(lambda x:abs(x['wordnum1']-x['len2']), axis=1)

df['same_word_perc'] = df.apply(lambda x:abs(2.0*x['sameWordNum']/(x['len1']+x['len2'])), axis=1)
df.columns
feature = df.drop(['id','qid1','qid2','question1','question2', 'is_duplicate','counter1',

                    'counter2'],1).columns
train_size = int(len(df)*0.8)

trind = np.random.permutation(len(df))[:train_size]

teind = np.random.permutation(len(df))[train_size:]
Xtrain = df.loc[trind, feature].copy().values

Ytrain = df.loc[trind, 'is_duplicate'].copy().values

Xval = df.loc[teind, feature].copy().values

Yval = df.loc[teind, 'is_duplicate'].copy().values
Xval.shape
rf = RandomForestClassifier(n_jobs=-1, criterion = 'entropy')

rf.fit(Xtrain, Ytrain)
# accurary

rf.score(Xtrain, Ytrain), rf.score(Xval, Yval)
tree_num = [10,20,30,40,50]

ll = []

for i, num in enumerate(tree_num):

    rf = RandomForestClassifier(n_estimators=num, n_jobs=-1, criterion = 'entropy')

    rf.fit(Xtrain, Ytrain)

    Yfit= rf.predict(Xval)

    ll.append(log_loss(Yfit, Yval))
plt.plot(tree_num,ll)

plt.xlabel('Tree Number')

plt.ylabel('Log Loss')

plt.title('Tree number VS Loss')
min_sample_leaf = [1,3,5,10,25,100]

ll = []

for i, num in enumerate(min_sample_leaf):

    rf = RandomForestClassifier(n_estimators=30, min_samples_leaf = num, n_jobs=-1, criterion = 'entropy')

    rf.fit(Xtrain, Ytrain)

    Yfit= rf.predict(Xval)

    ll.append(log_loss(Yfit, Yval))
plt.plot(min_sample_leaf,ll)

plt.xlabel('Min sample per leaf')

plt.ylabel('Log Loss')

plt.title('Min sample per leaf VS Loss')
max_feature = [0.1,0.3,0.5,0.7,0.9]

ll = []

for i, num in enumerate(max_feature):

    rf = RandomForestClassifier(n_estimators=30, max_features = num, n_jobs=-1, criterion = 'entropy')

    rf.fit(Xtrain, Ytrain)

    Yfit= rf.predict(Xval)

    ll.append(log_loss(Yfit, Yval))
plt.plot(max_feature,ll)

plt.xlabel('Min sample per leaf')

plt.ylabel('Log Loss')

plt.title('Min sample per leaf VS Loss')
param_dist = {"max_features": [0.1,0.3,0.5,0.7,0.9],

             "min_samples_leaf": [1,3,5,10,25,100],

             "n_estimators": [10,20,30,40,50]}



# run randomized search

n_iter_search = 10 

rf = RandomForestClassifier()

random_search = RandomizedSearchCV(rf, param_distributions=param_dist,

                                   n_iter=n_iter_search)

start = time()

random_search.fit(Xtrain, Ytrain)
def report(results, n_top=3):

    for i in range(1, n_top + 1):

        candidates = np.flatnonzero(results['rank_test_score'] == i)

        for candidate in candidates:

            print("Model with rank: {0}".format(i))

            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(

                  results['mean_test_score'][candidate],

                  results['std_test_score'][candidate]))

            print("Parameters: {0}".format(results['params'][candidate]))

            print("")



report(random_search.cv_results_)
# Best Parameters: 'n_estimators': 40, 'min_samples_leaf': 25, 'max_features': 0.1