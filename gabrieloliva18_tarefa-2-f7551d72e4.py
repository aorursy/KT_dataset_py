import numpy as np
import pandas as pd
import sklearn as sk
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn import naive_bayes as nb
from sklearn.metrics import fbeta_score
treino = pd.read_csv("../input/tarefa2/train_data.csv")
treino.head(10)
treino.info()
ntreino = treino.drop(columns = ["Id"])
correlacao = ntreino.corr().abs()
s = correlacao.unstack()
so = s.sort_values(kind="heapsort", ascending = False)
n = so.drop_duplicates()
n.head(20)
natreino = ntreino.drop(columns = ["word_freq_415","word_freq_857"])
spam = natreino.query("ham == False")
ham = natreino.query("ham == True")
correlacao = natreino.corr().abs()
s = correlacao.unstack()
so = s.sort_values(kind="heapsort", ascending = False)
n = so.drop_duplicates()
n.head(10)
mean_spam = spam.mean().sort_values(kind="heapsort", ascending = False)
mean_spam.head(15)
mean_ham = ham.mean().sort_values(kind="heapsort", ascending = False)
mean_ham.head(15)
locations = [1, 2]
heights = [mean_ham.capital_run_length_total, mean_spam.capital_run_length_total]
labels = ["Ham", "Spam"]
plt.bar(locations, heights, tick_label=labels)
plt.title('Media do total de caracteres maiusculos')
plt.ylabel('Media do total de caracteres maiusculos')
plt.xlabel('Email')
locations = [1, 2]
heights = [mean_ham.word_freq_you, mean_spam.word_freq_you]
labels = ["Ham", "Spam"]
plt.bar(locations, heights, tick_label=labels)
plt.title('Media da frequencia da palavra you')
plt.ylabel('Media da frequencia da palavra you')
plt.xlabel('Email')
features = natreino.drop(columns = "ham")
target = natreino.ham
nbgauss = nb.GaussianNB()
nbgauss.fit(features, target)
score = cross_val_score(nbgauss, features, target, cv = 10, scoring = "f1")
score.mean()
nbmulti = nb.MultinomialNB()
nbmulti.fit(features, target)
score = cross_val_score(nbmulti, features, target, cv = 10, scoring = "f1")
score.mean()
nbber = nb.BernoulliNB()
nbber.fit(features, target)
score = cross_val_score(nbber, features, target, cv = 10, scoring = "f1")
score.mean()
teste = pd.read_csv("../input/tarefa2/test_features.csv")
teste.head(5)
teste.shape
nteste = teste.drop(columns = ["word_freq_415","word_freq_857"])
predicao = nbber.predict(nteste.drop(columns = "Id"))
resultados = pd.DataFrame({'Id':nteste.Id,'ham':predicao[:]})
resultados.to_csv("resultados.csv", index = False)
resultados.shape
resultados.head(5)
