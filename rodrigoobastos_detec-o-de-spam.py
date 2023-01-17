import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn import naive_bayes
spam = pd.read_csv("../input/spamdata/train_data.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
spam.head()
abs((spam.corr())['ham']).sort_values(ascending= False)
(abs((spam.corr())['ham']).sort_values(ascending= False)).iloc[1:20].plot(kind="bar")
spamy = spam.ham
spamx = spam.iloc[:,0:57]
spamcor = spamx[["word_freq_your", "word_freq_000", "char_freq_$", "word_freq_remove", "word_freq_you", "word_freq_free", "word_freq_business", "word_freq_hp", "capital_run_length_total", "word_freq_order", "word_freq_hpl", "word_freq_receive", "word_freq_our", "char_freq_!", "word_freq_over", "word_freq_credit", "word_freq_money", "capital_run_length_longest", "word_freq_internet"]]
spamcor.head()
spammean = spamx/spamx.mean()
spammeancor = spammean[["word_freq_your", "word_freq_000", "char_freq_$", "word_freq_remove", "word_freq_you", "word_freq_free", "word_freq_business", "word_freq_hp", "capital_run_length_total", "word_freq_order", "word_freq_hpl", "word_freq_receive", "word_freq_our", "char_freq_!", "word_freq_over", "word_freq_credit", "word_freq_money", "capital_run_length_longest", "word_freq_internet"]]
spammean.head()
spammax = spamx/spamx.max()
spammaxcor = spammax[["word_freq_your", "word_freq_000", "char_freq_$", "word_freq_remove", "word_freq_you", "word_freq_free", "word_freq_business", "word_freq_hp", "capital_run_length_total", "word_freq_order", "word_freq_hpl", "word_freq_receive", "word_freq_our", "char_freq_!", "word_freq_over", "word_freq_credit", "word_freq_money", "capital_run_length_longest", "word_freq_internet"]]
spammax.head()
scoresknn = []
indice = []
mscores = []
for x in range (1, 50):
    knn = KNeighborsClassifier(n_neighbors=x)
    scores = cross_val_score(knn, spamx, spamy, cv=10, scoring='f1')
    mscores.append(scores.mean())
scoresknn.append(max(mscores))
indice.append(mscores.index(max(mscores)))
mscores = []
for x in range (1, 50):
    knn = KNeighborsClassifier(n_neighbors=x)
    scores = cross_val_score(knn, spamcor, spamy, cv=10, scoring='f1')
    mscores.append(scores.mean())
scoresknn.append(max(mscores))
indice.append(mscores.index(max(mscores)))
mscores = []
for x in range (1, 50):
    knn = KNeighborsClassifier(n_neighbors=x)
    scores = cross_val_score(knn, spammean, spamy, cv=10, scoring='f1')
    mscores.append(scores.mean())
scoresknn.append(max(mscores))
indice.append(mscores.index(max(mscores)))
mscores = []
for x in range (1, 50):
    knn = KNeighborsClassifier(n_neighbors=x)
    scores = cross_val_score(knn, spammeancor, spamy, cv=10, scoring='f1')
    mscores.append(scores.mean())
scoresknn.append(max(mscores))
indice.append(mscores.index(max(mscores)))
mscores = []
for x in range (1, 50):
    knn = KNeighborsClassifier(n_neighbors=x)
    scores = cross_val_score(knn, spammax, spamy, cv=10, scoring='f1')
    mscores.append(scores.mean())
scoresknn.append(max(mscores))
indice.append(mscores.index(max(mscores)))
mscores = []
for x in range (1, 50):
    knn = KNeighborsClassifier(n_neighbors=x)
    scores = cross_val_score(knn, spammaxcor, spamy, cv=10, scoring='f1')
    mscores.append(scores.mean())
scoresknn.append(max(mscores))
indice.append(mscores.index(max(mscores)))
scoresknn
indice
scoresnbg = []
nbg = naive_bayes.GaussianNB()
scores = cross_val_score(nbg, spamx, spamy, cv=10, scoring='f1')
scoresnbg.append(scores.mean())
scores = cross_val_score(nbg, spamcor, spamy, cv=10, scoring='f1')
scoresnbg.append(scores.mean())
scores = cross_val_score(nbg, spammean, spamy, cv=10, scoring='f1')
scoresnbg.append(scores.mean())
scores = cross_val_score(nbg, spammeancor, spamy, cv=10, scoring='f1')
scoresnbg.append(scores.mean())
scores = cross_val_score(nbg, spammax, spamy, cv=10, scoring='f1')
scoresnbg.append(scores.mean())
scores = cross_val_score(nbg, spammaxcor, spamy, cv=10, scoring='f1')
scoresnbg.append(scores.mean())
scoresnbg
scoresnbm = []
nbm = naive_bayes.MultinomialNB()
scores = cross_val_score(nbm, spamx, spamy, cv=10, scoring='f1')
scoresnbm.append(scores.mean())
scores = cross_val_score(nbm, spamcor, spamy, cv=10, scoring='f1')
scoresnbm.append(scores.mean())
scores = cross_val_score(nbm, spammean, spamy, cv=10, scoring='f1')
scoresnbm.append(scores.mean())
scores = cross_val_score(nbm, spammeancor, spamy, cv=10, scoring='f1')
scoresnbm.append(scores.mean())
scores = cross_val_score(nbm, spammax, spamy, cv=10, scoring='f1')
scoresnbm.append(scores.mean())
scores = cross_val_score(nbm, spammaxcor, spamy, cv=10, scoring='f1')
scoresnbm.append(scores.mean())
scoresnbm
scoresnbc = []
nbc = naive_bayes.ComplementNB()
scores = cross_val_score(nbc, spamx, spamy, cv=10, scoring='f1')
scoresnbc.append(scores.mean())
scores = cross_val_score(nbc, spamcor, spamy, cv=10, scoring='f1')
scoresnbc.append(scores.mean())
scores = cross_val_score(nbc, spammean, spamy, cv=10, scoring='f1')
scoresnbc.append(scores.mean())
scores = cross_val_score(nbc, spammeancor, spamy, cv=10, scoring='f1')
scoresnbc.append(scores.mean())
scores = cross_val_score(nbc, spammax, spamy, cv=10, scoring='f1')
scoresnbc.append(scores.mean())
scores = cross_val_score(nbc, spammaxcor, spamy, cv=10, scoring='f1')
scoresnbc.append(scores.mean())
scoresnbc
scoresnbb = []
nbb = naive_bayes.BernoulliNB()
scores = cross_val_score(nbb, spamx, spamy, cv=10, scoring='f1')
scoresnbb.append(scores.mean())
scores = cross_val_score(nbb, spamcor, spamy, cv=10, scoring='f1')
scoresnbb.append(scores.mean())
scores = cross_val_score(nbb, spammean, spamy, cv=10, scoring='f1')
scoresnbb.append(scores.mean())
scores = cross_val_score(nbb, spammeancor, spamy, cv=10, scoring='f1')
scoresnbb.append(scores.mean())
scores = cross_val_score(nbb, spammax, spamy, cv=10, scoring='f1')
scoresnbb.append(scores.mean())
scores = cross_val_score(nbb, spammaxcor, spamy, cv=10, scoring='f1')
scoresnbb.append(scores.mean())
scoresnbb
knn = KNeighborsClassifier(n_neighbors=10)
scores = cross_val_score(knn, spammeancor, spamy, cv=10, scoring='f1')
scores.mean()
knn.fit(spammeancor,spamy)
test = pd.read_csv("../input/spamtestdata/test_features.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
testmean = test/test.mean()
testx = testmean[["word_freq_your", "word_freq_000", "char_freq_$", "word_freq_remove", "word_freq_you", "word_freq_free", "word_freq_business", "word_freq_hp", "capital_run_length_total", "word_freq_order", "word_freq_hpl", "word_freq_receive", "word_freq_our", "char_freq_!", "word_freq_over", "word_freq_credit", "word_freq_money", "capital_run_length_longest", "word_freq_internet"]]
testy = knn.predict(testx)
testy
Ytabela = pd.DataFrame(index=test.Id,columns=['ham'])
Ytabela['ham'] = testy
Ytabela