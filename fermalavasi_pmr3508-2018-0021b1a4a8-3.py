import pandas as pd
import sklearn
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import cross_val_score
spam = pd.read_csv("../input/spamdataset/train_data.csv",
       sep=r'\s*,\s*',
       engine='python',
       na_values="?")
spam.head()
spam.shape
testSpam = pd.read_csv("../input/spamdataset/test_features.csv",
           sep=r'\s*,\s*',
           engine='python',
           na_values="?")
testSpam.head()
testSpam.shape
Xspam = spam[["word_freq_free","word_freq_address","word_freq_money","word_freq_cs","char_freq_!","char_freq_$",
             "word_freq_re","word_freq_edu","word_freq_george","word_freq_pm","word_freq_hp","word_freq_hpl",
             "word_freq_addresses"]]
Yspam = spam.ham
XtestSpam = testSpam[["word_freq_free","word_freq_address","word_freq_money","word_freq_cs","char_freq_!","char_freq_$",
                     "word_freq_re","word_freq_edu","word_freq_george","word_freq_pm","word_freq_hp","word_freq_hpl",
                     "word_freq_addresses"]]
maior_k = 0
maior_score = 0
for k in range(1,50):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, Xspam, Yspam, cv=28)
    media_scores = scores.mean()
    if media_scores > maior_score:
        maior_score = media_scores
        maior_k = k
knn = KNeighborsClassifier(n_neighbors=maior_k)
maior_k
scores = cross_val_score(knn, Xspam, Yspam, cv=10)
scores
scores.mean()
knn.fit(Xspam,Yspam)
YtestPred = knn.predict(XtestSpam)
result = np.vstack((testSpam["Id"], YtestPred)).T
x = ["Id","ham"]
Resultado = pd.DataFrame(columns = x, data = result)
Resultado.to_csv("Resultado.csv", index = False)
Resultado.head()
X = Xspam
Y = Yspam
clf = BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
clf.fit(X, Y)
YtestPredNB = clf.predict(XtestSpam)
scoresNB = cross_val_score(clf, Xspam, Yspam, cv=10)
scoresNB
scoresNB.mean()
resultNB = np.vstack((testSpam["Id"], YtestPredNB)).T
x = ["Id","ham"]
ResultadoNB = pd.DataFrame(columns = x, data = resultNB)
ResultadoNB.to_csv("ResultadoNB.csv", index = False)
ResultadoNB.head()
