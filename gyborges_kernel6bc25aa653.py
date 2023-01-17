#imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn 
from sklearn import feature_extraction, model_selection, naive_bayes, metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import cross_val_score
from sklearn import naive_bayes
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,fbeta_score
import os
#leitura dos arquivos 
trainData = pd.read_csv("../input/spamdb/train_data.csv")
testData = pd.read_csv("../input/spamdb/test_features.csv")

#cabeçalho
trainData.head()
#cabeçalho
testData.head()
#número de linhas e colunas
trainData.shape

#número de linhas e colunas
testData.shape
#checando diminui o número para verificar se há lacunas
trainDataSemLacunas = trainData.dropna()
trainDataSemLacunas.shape 
#checando diminui o número para verificar se há lacunas
testDataSemLacunas = testData.dropna()
testDataSemLacunas.shape 
#Vendo número de spams e hams
trainData["ham"].value_counts()
#verificando a correlação dos dados
trainData.corr()
#organizando as corrlações em ordem crescente
trainDataCorr = trainData.corr()
trainDataHamCorr = trainDataCorr["ham"].abs()
trainDataHamCorr.sort_values()
#Pegando somente as correlações maiores que 0,2
corrUtilizadas = trainDataCorr[abs(trainDataCorr.ham) > 0.15]
list(corrUtilizadas.index.drop("ham"))
#Pegando somente as correlações da lista anterior
YtrainData = trainData.ham
XtrainData = trainData[['word_freq_our',
 'word_freq_over',
 'word_freq_remove',
 'word_freq_internet',
 'word_freq_order',
 'word_freq_receive',
 'word_freq_free',
 'word_freq_business',
 'word_freq_you',
 'word_freq_credit',
 'word_freq_your',
 'word_freq_000',
 'word_freq_money',
 'word_freq_hp',
 'word_freq_hpl',
 'char_freq_!',
 'char_freq_$',
 'capital_run_length_longest',
 'capital_run_length_total']] 
XtestData = testData[['word_freq_our',
 'word_freq_over',
 'word_freq_remove',
 'word_freq_internet',
 'word_freq_order',
 'word_freq_receive',
 'word_freq_free',
 'word_freq_business',
 'word_freq_you',
 'word_freq_credit',
 'word_freq_your',
 'word_freq_000',
 'word_freq_money',
 'word_freq_hp',
 'word_freq_hpl',
 'char_freq_!',
 'char_freq_$',
 'capital_run_length_longest',
 'capital_run_length_total']] 



#testando varios n's e pegando o com melhor resultado
melhorResultado = 0
melhorN = 0
for i in range(5,30):
    
    knn = KNeighborsClassifier(n_neighbors=(i))
    resultados = cross_val_score(knn, XtrainData, YtrainData, cv=10)
    mean = np.mean(resultados)
    if mean > melhorResultado:
        melhorResultado = mean
        melhorN = i
    
print(melhorResultado)
print(melhorN)
knn = KNeighborsClassifier(n_neighbors=melhorN)

# testando o metodo Gaussiano e de Bernoulli para ver qual  o melhor
naiveBayesGaussian = naive_bayes.GaussianNB()
resultados = cross_val_score(naiveBayesGaussian, XtrainData, YtrainData, cv=10)
print("Gaussian")
print(resultados.mean())

naiveBayesBernoulli = naive_bayes.BernoulliNB()
resultados = cross_val_score(naiveBayesBernoulli, XtrainData, YtrainData, cv=10)
print("Bernoulli")
print(resultados.mean())
#Como o Metodo de Naive Bayes Gaussiano foi o com melhor resultado, será ele o utilizado para fazer as predições:
naiveBayesBernoulli = naive_bayes.BernoulliNB()
naiveBayesBernoulli.fit(XtrainData,YtrainData)
#criando o arquivo a ser entregue
YtestData = naiveBayesBernoulli.predict(XtestData)
predictions = pd.DataFrame({"id":testData.Id, "ham":YtestData})
predictions.to_csv("predictions.csv", index=False)
predictions