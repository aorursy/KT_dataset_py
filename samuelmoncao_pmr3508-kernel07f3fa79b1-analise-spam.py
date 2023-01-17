# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import collections, numpy
#Aqui, analisaremos a quantidade de linhas e headers dos dados que serão utilizados

train = pd.read_csv("../input/pmrdata/train_data.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
train.shape
#Aqui analisaremos o header dos dados de treino

train.head()
#Apesar de os valores naop serem frequentes em todas as linhas, 
#utilizaremos essa análise para estudar o valor máximo de ocorrências

#Com isso, teremos quais são as palavras que demonstram que a mensagem
#possivelmente seria um spam

train.describe()
#Contaremos os verdadeiros e falsos para as análises que serão feitas posteriormente
train["ham"].value_counts()
#Baseado na análise da ocorrência das palavras, faremos o classificador KNN levando em
#conta apenas as colunas que possuem valores próximos ou maior que 20, visto que temos
#uma frequência razoável de dados para casos de menor valor.

Xtrain = train[["word_freq_address","word_freq_3d","word_freq_mail",
                   "word_freq_free","word_freq_you","word_freq_george",
                   "word_freq_font","word_freq_edu","word_freq_project",
                   "char_freq_!"]]
Ytrain = train.ham
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
from sklearn.model_selection import cross_val_score
knn.fit(Xtrain, Ytrain)
scores = cross_val_score(knn, Xtrain, Ytrain, cv=10)

#Com isso, temos:
#1. A qualidade do nosso classificador KNN
scores.mean()
#2. A qualidade do nosso classificador Bernoulli
scores_bernoulli = cross_val_score(BernoulliNB(), Xtrain, Ytrain, cv=10)
np.mean(scores_bernoulli)
#3. A qualidade do nosso classificador Gaussiano
scores_gaussiano = cross_val_score(GaussianNB(), Xtrain, Ytrain, cv=10)
np.mean(scores_gaussiano)
#4. A qualidade do nosso classificador Multinomial
scores_multinomial = cross_val_score(MultinomialNB(), Xtrain, Ytrain, cv=10)
np.mean(scores_multinomial)
#Baseados nos testes anteriores, temos que nosso classificador KNN na verdade seria 

#Nesta etapa, conseguimos contar quantos Verdadeiros realmente temos
collections.Counter(Ytrain)
#Agora, contaremos quantos positivos nosso classificador KNN rotulou
Xcont = train[["word_freq_address","word_freq_3d","word_freq_mail",
                   "word_freq_free","word_freq_you","word_freq_george",
                   "word_freq_font","word_freq_edu","word_freq_project",
                   "char_freq_!"]]
Ycont = knn.predict(Xcont)

#Convertemos então os valores booleanos para inteiros (True = 1 e False = 0)
Ycont_int = Ycont + 0
Ytrain_int = Ytrain + 0

#Agora, utilizamos o metodo abaixo para entender qual foi a taxa de Falsos Positivos e Verdadeiros Positivos
confusion_matrix(Ytrain_int, Ycont_int)
cmat = confusion_matrix(Ytrain_int, Ycont_int)

cmat
#Aqui, identificamos que o valor de recall seria de 0.87
target_names = ['class 0', 'class 1']
classification_report(Ytrain_int, Ycont_int, target_names=target_names)
'TP - True Positive: {} rótulos'.format(cmat[1,1])
'TP - True Negative: {} rótulos'.format(cmat[0,0])
#Temos então o TruePositiveRate
TP_Rate = 2076/2251

TP_Rate
#Temos então o TrueNegativeRate
TN_Rate = 1223/1429

TN_Rate
#utilizando os dados de teste
teste = pd.read_csv("../input/pmrdata/test_features.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
Xteste = teste[["word_freq_address","word_freq_3d","word_freq_mail",
                   "word_freq_free","word_freq_you","word_freq_george",
                   "word_freq_font","word_freq_edu","word_freq_project",
                   "char_freq_!"]]
Yteste = knn.predict(Xteste)