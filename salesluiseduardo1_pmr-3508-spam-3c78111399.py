import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import cross_val_score
from sklearn import naive_bayes
from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score
import os
#classificador para detectar spam
#inicialmente estamos só transferindo os dados da base de treino
train = pd.read_csv("../input/spamdata/train_data (3).csv")
train.head()
train.shape
ntrain = train.dropna() #analisando se há missing data
ntrain.shape #não havia missing data
train.describe() # mineração de dados
train.corr() #vamos descbrir como os dados estão sendo correlacionados para poder identificar padrões com o fato de ser ham ou spam
plt.matshow(train.corr())
plt.colorbar() #analise gráfica das correlações
correlacao = train.corr()
correlacao_ham = correlacao["ham"].abs()
correlacao_ham.sort_values() #estabelecemos aqui a ordem de menor para o maior da correlação com spam
validos = correlacao[abs(correlacao.ham) > 0.25] # pegamos aqui apenas as features de maior peso na definição da label
listadevalidos = list(validos.index.drop("ham"))
listadevalidos
test = pd.read_csv("../input/spamdata/test_features.csv") #abrindo base de testes
test.head()
test.shape
Ytrain=train.ham
Xtrain=train[['word_freq_remove',
 'word_freq_free',
 'word_freq_business',
 'word_freq_you',
 'word_freq_your',
 'word_freq_000',
 'word_freq_hp',
 'char_freq_$',
 'capital_run_length_total']] #utilizando apenas as features que consideramos relevantes
Xtest = test[['word_freq_remove',
 'word_freq_free',
 'word_freq_business',
 'word_freq_you',
 'word_freq_your',
 'word_freq_000',
 'word_freq_hp',
 'char_freq_$',
 'capital_run_length_total']]
dados_comparar=pd.read_csv("../input/spamdata/sample_submission_1.csv")#dados para comparar ao final a accuracy do classificador
Ydados_comparar=dados_comparar["ham"]
GaussianClassifier = naive_bayes.GaussianNB()#Analisaremos 3 diferentes métodos de Naives e escolheremos o que tiver maior score nos cross validation
scoresGC = cross_val_score(GaussianClassifier, Xtrain, Ytrain, cv=10)
scoresGC.mean()
BernoulliClassifier = naive_bayes.BernoulliNB()
scoresBC = cross_val_score(BernoulliClassifier, Xtrain, Ytrain, cv=10)
scoresBC.mean()
MultinomialClassifier = naive_bayes.MultinomialNB()
scoresMC = cross_val_score(MultinomialClassifier, Xtrain, Ytrain, cv=10)
scoresMC.mean()
NaivesClassifier=naive_bayes.BernoulliNB() #como o Bernoulli apresentou os resultados mais próximos, vamos configurar o classificador com base nele
NaivesClassifier.fit(Xtrain,Ytrain)
NaivesYTest=NaivesClassifier.predict(Xtest)
NaivesYTest
accuracy_score(Ydados_comparar,NaivesYTest) #no caso com Naive Bayes
knn= KNeighborsClassifier(n_neighbors=5) #agora vejamos com o método KNN e compararemos 3 casos
scoresKN= cross_val_score(knn, Xtrain, Ytrain, cv=10)
scoresKN.mean()
knn= KNeighborsClassifier(n_neighbors=30)
scoresKN= cross_val_score(knn, Xtrain, Ytrain, cv=10)
scoresKN.mean()
knn= KNeighborsClassifier(n_neighbors=50)
scoresKN= cross_val_score(knn, Xtrain, Ytrain, cv=10)
scoresKN.mean()
knn= KNeighborsClassifier(n_neighbors=5) #a que possuiu melhor resultsdo no treino foi a com neow k
knn.fit(Xtrain,Ytrain)
ntest=test.dropna()
ntest.shape
KnnYtest = knn.predict(Xtest)
KnnYtest
accuracy_score(Ydados_comparar,KnnYtest)