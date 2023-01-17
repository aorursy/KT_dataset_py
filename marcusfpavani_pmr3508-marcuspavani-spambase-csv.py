import numpy as np
import pandas as pd
import copy as cp

import sklearn
from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
from IPython.display import Image

%matplotlib inline 

from statistics import mode
import os.path
print (os.listdir("../input"))
trainData = pd.read_csv("../input/spambase/train_data.csv", header=0, index_col=["Id"], na_values="?")
testData = pd.read_csv("../input/spambase/test_features.csv", header=0, index_col=["Id"], na_values="?")
trainData.head(n=10)
trainHam  = trainData.query('ham == 1')  # Para melhor visualização dos dados, dividimos a base de testes em 2
trainSpam = trainData.query('ham == 0')  # Uma base de emails desejáveis (Ham) e uma de indesejáveis (Spam).
trainHam  = trainHam.drop(columns=['ham','capital_run_length_average','capital_run_length_longest','capital_run_length_total'])
trainSpam = trainSpam.drop(columns=['ham','capital_run_length_average','capital_run_length_longest','capital_run_length_total'])
listOfCols = list(trainHam)
positiveDiff = 0;
negativeDiff = 0;

avgPositiveDiff = 0;
avgNegativeDiff = 0;

for col in listOfCols:
    diff = trainSpam[col].mean() - trainHam[col].mean()
    if diff > 0:
        positiveDiff += 1;
        avgPositiveDiff += diff
    if diff < 0:
        negativeDiff += 1;
        avgNegativeDiff += diff
    
avgPositiveDiff /= positiveDiff
avgNegativeDiff /= negativeDiff
avgPositiveDiff
avgNegativeDiff
relevantCols = []
columnsToDrop = []

for col in listOfCols:
    diff = trainSpam[col].mean() - trainHam[col].mean()
    if diff > 0:
        if diff > 0.5*avgPositiveDiff:
            relevantCols.append(col)
        else:
            columnsToDrop.append(col)
    if diff < 0:
        if diff < 0.5*avgNegativeDiff:
            relevantCols.append(col)
        else:
            columnsToDrop.append(col)
        
relevantCols
labels = ["Ham", "Spam"]
plt.bar([1,2], [trainHam['word_freq_our'].mean(), trainSpam['word_freq_our'].mean()], tick_label=labels)
plt.ylabel('Frequência média da palavra "our"')
plt.xlabel('Tipo de email')
plt.gca().set_ylim([0,2.5])
labels = ["Ham", "Spam"]
plt.bar([1,2], [trainHam['word_freq_remove'].mean(), trainSpam['word_freq_remove'].mean()], tick_label=labels)
plt.ylabel('Frequência média da palavra "remove"')
plt.xlabel('Tipo de email')
plt.gca().set_ylim([0,2.5])
labels = ["Ham", "Spam"]
plt.bar([1,2], [trainHam['word_freq_free'].mean(), trainSpam['word_freq_free'].mean()], tick_label=labels)
plt.ylabel('Frequência média da palavra "free"')
plt.xlabel('Tipo de email')
plt.gca().set_ylim([0,2.5])
labels = ["Ham", "Spam"]
plt.bar([1,2], [trainHam['word_freq_you'].mean(), trainSpam['word_freq_you'].mean()], tick_label=labels)
plt.ylabel('Frequência média da palavra "you"')
plt.xlabel('Tipo de email')
plt.gca().set_ylim([0,2.5])
labels = ["Ham", "Spam"]
plt.bar([1,2], [trainHam['word_freq_your'].mean(), trainSpam['word_freq_your'].mean()], tick_label=labels)
plt.ylabel('Frequência média da palavra "your"')
plt.xlabel('Tipo de email')
plt.gca().set_ylim([0,2.5])
labels = ["Ham", "Spam"]
plt.bar([1,2], [trainHam['word_freq_hp'].mean(), trainSpam['word_freq_hp'].mean()], tick_label=labels)
plt.ylabel('Frequência média da palavra "hp"')
plt.xlabel('Tipo de email')
plt.gca().set_ylim([0,2.5])
labels = ["Ham", "Spam"]
plt.bar([1,2], [trainHam['word_freq_hpl'].mean(), trainSpam['word_freq_hpl'].mean()], tick_label=labels)
plt.ylabel('Frequência média da palavra "hpl"')
plt.xlabel('Tipo de email')
plt.gca().set_ylim([0,2.5])
labels = ["Ham", "Spam"]
plt.bar([1,2], [trainHam['word_freq_george'].mean(), trainSpam['word_freq_george'].mean()], tick_label=labels)
plt.ylabel('Frequência média da palavra "george"')
plt.xlabel('Tipo de email')
plt.gca().set_ylim([0,2.5])
labels = ["Ham", "Spam"]
plt.bar([1,2], [trainHam['word_freq_meeting'].mean(), trainSpam['word_freq_meeting'].mean()], tick_label=labels)
plt.ylabel('Frequência média da palavra "meeting"')
plt.xlabel('Tipo de email')
plt.gca().set_ylim([0,2.5])
labels = ["Ham", "Spam"]
plt.bar([1,2], [trainHam['word_freq_re'].mean(), trainSpam['word_freq_re'].mean()], tick_label=labels)
plt.ylabel('Frequência média da palavra "re"')
plt.xlabel('Tipo de email')
plt.gca().set_ylim([0,2.5])
labels = ["Ham", "Spam"]
plt.bar([1,2], [trainHam['word_freq_edu'].mean(), trainSpam['word_freq_edu'].mean()], tick_label=labels)
plt.ylabel('Frequência média da palavra "edu"')
plt.xlabel('Tipo de email')
plt.gca().set_ylim([0,2.5])
labels = ["Ham", "Spam"]
plt.bar([1,2], [trainHam['char_freq_!'].mean(), trainSpam['char_freq_!'].mean()], tick_label=labels)
plt.ylabel('Frequência média do caracter "!"')
plt.xlabel('Tipo de email')
plt.gca().set_ylim([0,2.5])
columnsToDrop
for col in columnsToDrop:
    trainData = trainData.drop(columns=[col])
trainData.head()
trainHam  = trainData.query('ham == 1')  # Para melhor visualização dos dados, dividimos a base de testes em 2
trainSpam = trainData.query('ham == 0')  # Uma base de emails desejáveis (Ham) e uma de indesejáveis (Spam).
trainHam  = trainHam.drop(columns=['ham'])
trainSpam = trainSpam.drop(columns=['ham'])
labels = ["Ham", "Spam"]
plt.bar([1,2], [trainHam['capital_run_length_average'].mean(), trainSpam['capital_run_length_average'].mean()], tick_label=labels)
plt.ylabel('Média de Sequência de Letras Maiúsculas"')
plt.xlabel('Tipo de email')
labels = ["Ham", "Spam"]
plt.bar([1,2], [trainHam['capital_run_length_longest'].mean(), trainSpam['capital_run_length_longest'].mean()], tick_label=labels)
plt.ylabel('Maior Sequência de Letras Maiúsculas"')
plt.xlabel('Tipo de email')
labels = ["Ham", "Spam"]
plt.bar([1,2], [trainHam['capital_run_length_total'].mean(), trainSpam['capital_run_length_total'].mean()], tick_label=labels)
plt.ylabel('Total de Letras Maiúsculas"')
plt.xlabel('Tipo de email')
trainDataX = trainData.drop(columns="ham", inplace=False)
trainDataY = trainData["ham"]
gnb = GaussianNB()
gnb.fit(trainDataX, trainDataY)
scores = cross_val_score(gnb, trainDataX, trainDataY, cv=10)
scores
testData.head()
for col in columnsToDrop:
    testData = testData.drop(columns=[col])
testData.head()
predictedData = gnb.predict(testData)
output = pd.DataFrame(testData.index)
output["Ham"] = predictedData
output
output.to_csv("PMR3508_MarcusPavani_Spambase.csv", index=False)