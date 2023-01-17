import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
bow = pd.read_csv("../input/train-data/train_data.csv")
bow.head()
bow["ham"].value_counts().plot(kind="pie")
bow["ham"].value_counts()
2251/(2251+1429)
features = ['word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d',
            'word_freq_our', 'word_freq_over', 'word_freq_remove', 'word_freq_internet',
            'word_freq_order', 'word_freq_mail', 'word_freq_receive', 'word_freq_will',
            'word_freq_people', 'word_freq_report', 'word_freq_addresses', 'word_freq_free',
            'word_freq_business', 'word_freq_email', 'word_freq_you',  'word_freq_credit',
            'word_freq_your',  'word_freq_font', 'word_freq_000', 'word_freq_money',
            'word_freq_hp', 'word_freq_hpl', 'word_freq_george', 'word_freq_650',
            'word_freq_lab', 'word_freq_labs', 'word_freq_telnet', 'word_freq_857',
            'word_freq_data', 'word_freq_415', 'word_freq_85', 'word_freq_technology',
            'word_freq_1999', 'word_freq_parts', 'word_freq_pm', 'word_freq_direct',
            'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project',
            'word_freq_re', 'word_freq_edu', 'word_freq_table', 'word_freq_conference',
            'char_freq_(', 'char_freq_[', 'char_freq_!', 'char_freq_$', 'char_freq_#',
            'capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total']
correlacao = []

for i in features:
    valor = bow[i].corr(bow['ham'])
    if valor < 0:
        valor = -valor
    correlacao.append(valor)
dicionario = dict(zip(features, correlacao))
novo_dicionario = {}
for w in sorted(dicionario, key=dicionario.get, reverse=True):
    novo_dicionario.update([[w, dicionario[w]]])
novo_dicionario
Xbow = bow[["word_freq_your", "word_freq_000", "char_freq_$", "word_freq_remove",
            "word_freq_you", "word_freq_free", "char_freq_!", "word_freq_money",
            "word_freq_credit", "word_freq_george", "word_freq_650", "word_freq_project"]]
Ybow = bow.ham
maior = 0
indice = 0
for i in range(1, 50):
    knn = KNeighborsClassifier(n_neighbors=i)
    media = sum(cross_val_score(knn, Xbow, Ybow, cv=10))/len(cross_val_score(knn, Xbow, Ybow, cv=10))
    if (media > maior):
        maior = media
        indice = i
knn = KNeighborsClassifier(n_neighbors = indice)
indice, maior
bowTest = pd.read_csv("../input/test-features/test_features.csv",
      sep=r'\s*,\s*',
      engine='python',
      na_values="?")
knn.fit(Xbow,Ybow)
XbowTest = bowTest[["word_freq_your", "word_freq_000", "char_freq_$", "word_freq_remove",
                    "word_freq_you", "word_freq_free", "char_freq_!", "word_freq_money",
                    "word_freq_credit", "word_freq_george", "word_freq_650", "word_freq_project"]]
YtestPred = knn.predict(XbowTest)
Ids = bowTest["Id"]
submission = pd.DataFrame({"Id": Ids, "ham": YtestPred})
submission.head()
submission.to_csv("submission.csv", index = False)
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
GNB = GaussianNB()
scores_GNB = cross_val_score(GNB, Xbow, Ybow, cv=10)
scores_GNB
sum(scores_GNB) / len(scores_GNB)
MNB = MultinomialNB()
scores_MNB = cross_val_score(MNB, Xbow, Ybow, cv=10)
scores_MNB
sum(scores_MNB) / len(scores_MNB)
melhor = 0
indice = 0
media = 0
for i in range(100):
    BNB = BernoulliNB(binarize = i/100)
    scores_BNB = cross_val_score(BNB, Xbow, Ybow, cv=10)
    media = sum(scores_BNB)/len(scores_BNB)
    if media > melhor:
        melhor = media
        indice = i
BNB = BernoulliNB(binarize = indice/100)
indice, melhor
BNB.fit(Xbow,Ybow)
XbowTestNB = bowTest[["word_freq_your", "word_freq_000", "char_freq_$", "word_freq_remove",
                    "word_freq_you", "word_freq_free", "char_freq_!", "word_freq_money",
                    "word_freq_credit", "word_freq_george", "word_freq_650", "word_freq_project"]]
YtestPredNB = BNB.predict(XbowTestNB)
Ids = bowTest["Id"]
submissionNB = pd.DataFrame({"Id": Ids, "ham": YtestPredNB})
submissionNB.to_csv("submissionNB.csv", index = False)