# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
news = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/True.csv')

fake = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/Fake.csv')
news.columns, fake.columns
news = news[['title', 'text']]

fake = fake[['title', 'text']]
from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()
import re
news['label'] = 0

fake['label'] = 1
combined = news.append(fake)
combined['text'] = combined['text'].apply(lambda x: re.split('[ ,.:;]', x))
stop_words = stopwords.words('english')
combined['text'] = combined['text'].apply(lambda x: [y for y in x if y not in stop_words])
wordnet_lemmatizer.lemmatize('cubs', 'v')
combined['text'] = combined['text'].apply(lambda x: [wordnet_lemmatizer.lemmatize(y, 'v') for y in x])
combined['combined'] = combined.apply(lambda x: x['title']+' '+' '.join(x['text']), axis=1)
from sklearn.model_selection import train_test_split
X_trn,X_tst, y_trn, y_tst = train_test_split(combined['combined'].values, combined['label'])
from sklearn.feature_extraction.text import TfidfVectorizer



vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.8, min_df=0.1)

X_trn_vec = vectorizer.fit_transform(X_trn)
X_tst_vec = vectorizer.transform(X_tst)
feature_names = vectorizer.get_feature_names()
feature_names = np.asarray(feature_names)
print(f'number of samples: {len(combined)}, number of features: {len(feature_names)}')
from sklearn import metrics
type(X_trn_vec), type(y_trn)
y_trn = np.asarray(y_trn)
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import KFold



kf = KFold(n_splits=8, shuffle=True)

score_trains = {0.1:[], 0.5:[], 1.0:[]}

score_tests = {0.1:[], 0.5:[], 1.0:[]}



fold=0



def evaluate_f(alpha, fold,*data):

    x_train, y_train, x_test, y_test = data

    clf = MultinomialNB(alpha=alpha)

    clf.fit(x_train, y_train)

    pred = clf.predict(x_train)

    score1 = metrics.accuracy_score(y_train, pred)



    pred = clf.predict(x_test)

    score2 = metrics.accuracy_score(y_test, pred)

    print(f"train accuracy:   {score1}, test accuracy: {score2}")

    return score1, score2



for train_inds, test_inds in kf.split(X_trn_vec):

    print('_'*80)

    print(f"Training with Naive Bayes {fold}")

    x_train = X_trn_vec[train_inds] 

    y_train = y_trn[train_inds]

    x_test = X_trn_vec[test_inds]

    y_test = y_trn[test_inds]

    sc1, sc2 = evaluate_f(0.1, fold, x_train, y_train, x_test, y_test)

    score_trains[0.1].append(sc1)

    score_tests[0.1].append(sc2)

    

    sc1, sc2 = evaluate_f(0.5, fold, x_train, y_train, x_test, y_test)

    score_trains[0.5].append(sc1)

    score_tests[0.5].append(sc2)

    

    sc1, sc2 = evaluate_f(1.0, fold, x_train, y_train, x_test, y_test)

    score_trains[1.0].append(sc1)

    score_tests[1.0].append(sc2)

    fold+=1
import matplotlib.pyplot as plt
indices = np.arange(fold)



score_0d1 = score_trains[0.1]

score_0d5 = score_trains[0.5]

score_1d0 = score_trains[1.0]



test_0d1 = score_tests[0.1]

test_0d5 = score_tests[0.5]

test_1d0 = score_tests[1.0]



plt.figure(figsize=(12, 8))

plt.title("Score")

plt.bar(indices, score_0d1, .1,label="training accuracy alpha 0.1", color='navy')

plt.bar(indices + .1, test_0d1, .1, label="test accuracy alpha 0.1", color='c')

plt.bar(indices + .3, score_0d5, .1, label="training accuracy alpha 0.5", color='darkorange')

plt.bar(indices + .4, test_0d5, .1, label="test accuracy alpha 0.5", color='brown')

plt.bar(indices + .6, score_1d0, .1, label="training accuracy alpha 1.0", color='green')

plt.bar(indices + .7, test_1d0, .1, label="test accuracy alpha 1.0", color='gray')





plt.yticks(())

plt.legend(loc='best')

plt.subplots_adjust(left=.25)

plt.subplots_adjust(top=.95)

plt.subplots_adjust(bottom=.05)



plt.show()



pred = clf.predict(X_tst_vec)

score1 = metrics.accuracy_score(y_tst, pred)

score1
X_all = vectorizer.transform(combined['combined'].values)
y_all = combined['label'].values
clf = MultinomialNB(alpha=0.1)

clf.fit(X_all, y_all)
pred = clf.predict(X_all)

metrics.accuracy_score(y_all, pred)
clf.classes_
cls = {0:'news', 1:'fake'}

X = combined['combined'].values

y = combined['label'].values
X[0], cls[clf.predict(X_all[0])[0]], cls[y[0]] 
X[-1], cls[clf.predict(X_all[-1])[0]], cls[y[-1]] 