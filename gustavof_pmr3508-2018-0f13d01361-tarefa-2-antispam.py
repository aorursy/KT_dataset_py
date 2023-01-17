import pandas as pd
import sklearn
import numpy as nm
filetest = "../input/treinoteste/train_data.csv"
spam_train = pd.read_csv(filetest, 
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
spam_train.shape
spam_train.head()
spam_train["ham"].value_counts()
x = 2251 + 1429
y = 2251/x
y
spam_train.corr().ham
Xspam_train = spam_train [[ "word_freq_remove", "word_freq_order", "word_freq_receive", 
                         "word_freq_free", "word_freq_business",
                          "word_freq_you", "word_freq_your", "word_freq_000",
                          "word_freq_hp", "word_freq_hpl", "char_freq_!", 
                          "char_freq_$", "capital_run_length_total" ]]
Yspam_train = spam_train.ham
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
x = 1
for i in range(0,15):
    knn = KNeighborsClassifier(n_neighbors=(x))
    scores = cross_val_score(knn, Xspam_train, Yspam_train, cv=10)
    print(x)
    print(nm.mean(scores))
    x = x + 1
knn = KNeighborsClassifier(n_neighbors=(3))
scores = cross_val_score(knn, Xspam_train, Yspam_train, cv=10)
nm.mean(scores)
knn.fit(Xspam_train, Yspam_train)
Teste = pd.read_csv("../input/treinoteste/test_features.csv",
      sep=r'\s*,\s*',
      engine='python',
      na_values="?")
XTeste = Teste [[ "word_freq_remove", "word_freq_order", "word_freq_receive", 
                         "word_freq_free", "word_freq_business",
                          "word_freq_you", "word_freq_your", "word_freq_000",
                          "word_freq_hp", "word_freq_hpl", "char_freq_!", 
                          "char_freq_$", "capital_run_length_total" ]]
YTeste = knn.predict(XTeste)
YTeste.shape
Id = Teste["Id"]
fim = pd.DataFrame({"Id": Id, "ham": YTeste})
fim.to_csv("predictionspam.csv", index = False)
fim
#Com esse KNN obtivemos 83.9% de acurácia 
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
Gau = GaussianNB()
scores_Gau = cross_val_score(Gau, Xspam_train, Yspam_train, cv=5)
nm.mean(scores_Gau)
x = 0.01
for i in range(1,15):
    Bern = BernoulliNB(binarize = x)
    scores_Bern = cross_val_score(Bern, Xspam_train, Yspam_train, cv=5)
    print(x)
    print(nm.mean(scores_Bern))
    x = i*0.01
Bern = BernoulliNB(binarize = 0.06)
scores_Bern = cross_val_score(Bern, Xspam_train, Yspam_train, cv=5)
nm.mean(scores_Bern)
Multi = MultinomialNB()
scores_Multi = cross_val_score(Multi, Xspam_train, Yspam_train, cv=20)
nm.mean(scores_Multi)
Bern.fit(Xspam_train, Yspam_train)
XBernTeste = Teste [[ "word_freq_remove", "word_freq_order", "word_freq_receive", 
                         "word_freq_free", "word_freq_business",
                          "word_freq_you", "word_freq_your", "word_freq_000",
                          "word_freq_hp", "word_freq_hpl", "char_freq_!", 
                          "char_freq_$", "capital_run_length_total" ]]
YBernTeste = Bern.predict(XBernTeste)
Id = Teste["Id"]
fim2 = pd.DataFrame({"Id": Id, "ham": YBernTeste})
fim2.to_csv("predictionspam2.csv", index = False)
fim2
#Utilizando Bernoulli obtivemos uma acurácia de 91.5%, melhor que a obtida com kNN