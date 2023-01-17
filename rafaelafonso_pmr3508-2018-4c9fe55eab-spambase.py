import pandas as pd
import sklearn
bag = pd.read_csv("../input/tarefa2-spambase/train_data.csv",
      sep=r'\s*,\s*',
      engine='python',
      na_values="?")
bag.shape
bag.head()
import matplotlib.pyplot as plt
bag["ham"].value_counts().plot(kind="bar")
bag["ham"].value_counts()
1429*100/(2251+1429)
bag['capital_run_length_average'].corr(bag['ham'])
bag['word_freq_remove'].corr(bag['ham'])
bag['word_freq_your'].corr(bag['ham'])
bag['word_freq_free'].corr(bag['ham'])
bag['word_freq_000'].corr(bag['ham'])
bag['char_freq_!'].corr(bag['ham'])
bag['char_freq_$'].corr(bag['ham'])
Xbag = bag[["capital_run_length_average","word_freq_000", "word_freq_remove", 
            "word_freq_your", "word_freq_free", "char_freq_$","char_freq_!"]]
Ybag = bag.ham
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, Xbag, Ybag, cv=10)
scores
sum(scores) / len(scores)
bagTest = pd.read_csv("../input/tarefa2-spambase/test_features.csv",
      sep=r'\s*,\s*',
      engine='python',
      na_values="?")
knn.fit(Xbag,Ybag)
XbagTest = bagTest[["capital_run_length_average","word_freq_000", "word_freq_remove", 
                    "word_freq_your", "word_freq_free", "char_freq_$","char_freq_!"]]
YtestPred = knn.predict(XbagTest)
Id = bagTest["Id"]
submission = pd.DataFrame({"Id": Id, "ham": YtestPred})
submission.head()
submission.to_csv("submission.csv", index = False)
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
nbG = GaussianNB()
scores_nbG = cross_val_score(nbG, Xbag, Ybag, cv=5)
scores_nbG
sum(scores_nbG) / len(scores_nbG)
nbM = MultinomialNB()
scores_nbM = cross_val_score(nbM, Xbag, Ybag, cv=10)
scores_nbM
sum(scores_nbM) / len(scores_nbM)
nbB = BernoulliNB(binarize = 0.09)
scores_nbB = cross_val_score(nbB, Xbag, Ybag, cv=10)
scores_nbB
sum(scores_nbB) / len(scores_nbB)
nbB.fit(Xbag,Ybag)
XbagTestNB = bagTest[["capital_run_length_average","word_freq_000", "word_freq_remove", 
                      "word_freq_your", "word_freq_free", "char_freq_$","char_freq_!"]]
YtestPredNB = nbB.predict(XbagTestNB)
Id = bagTest["Id"]
submissionNB = pd.DataFrame({"Id": Id, "ham": YtestPredNB})
submissionNB.head()
submissionNB.to_csv("submissionNB.csv", index = False)