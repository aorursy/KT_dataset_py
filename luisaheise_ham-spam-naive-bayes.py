# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib as plt
from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
treino = pd.read_csv("../input/nb0001/train_data_nb.csv")
treino.head()
NotSpam = treino.loc[treino['ham'] == True]
Spam = treino.loc[treino['ham'] == False]
frequenciaMedia = np.zeros(shape=(48))
for i in range(48):
    palavra = Spam.columns[i]
    frequenciaMedia[i] = Spam[palavra].mean()
    
plt.figure(figsize=(20,10))
plt.title("Frequência Média de Cada Palavra em Spams")
plt.bar(Spam.columns[:48], frequenciaMedia)
plt.xticks(Spam.columns[:48], rotation='vertical')
plt.show()
frequenciaMediaNS = np.zeros(shape=(48))
for i in range(48):
    palavra = NotSpam.columns[i]
    frequenciaMediaNS[i] = NotSpam[palavra].mean()

quocientes = np.zeros(shape=(48))
for j in range(48):
    quociente = frequenciaMedia[j]/(frequenciaMedia[j] + frequenciaMediaNS[j])
    quocientes[j] = quociente
plt.figure(figsize=(20,10))
plt.title("Poder Discriminatório das Palavras")
plt.ylabel('Quociente S/(S+NS)')
plt.bar(Spam.columns[:48], quocientes)
plt.xticks(Spam.columns[:48], rotation='vertical')
plt.show()
teste = pd.read_csv("../input/nb0001/test_features.csv")
treinoY = treino.ham
treinoX = treino.drop('ham', axis=1)
gnb = naive_bayes.GaussianNB()
bnb = naive_bayes.BernoulliNB()

gnb.fit(treinoX, treinoY)
bnb.fit(treinoX, treinoY)
bscores = cross_val_score(bnb, treinoX, treinoY, cv=10)
gscores = cross_val_score(gnb, treinoX, treinoY, cv=10)
scores = bscores.mean(), gscores.mean()
plt.figure(figsize=(4,6))
plt.ylabel('Média da Acurácia - Cross Validation')
plt.bar(["Bernoulli", "Gaussiano"], scores)
roc1 = cross_val_predict(bnb, treinoX, treinoY, cv=10, method = 'predict_proba')
fpb, tpb ,thresholds =roc_curve(treinoY, roc1[:,1])


plt.plot(fpb,tpb)
plt.plot([0, 1], [0, 1], color='purple', linestyle='-.')
plt.xlabel('Especificidade')
plt.ylabel('Sensividade')
plt.title('ROC para Bernoulli')
plt.show()

roc2 = cross_val_predict(gnb, treinoX, treinoY, cv=10, method = 'predict_proba')
fpg, tpg ,thresholds =roc_curve(treinoY, roc2[:,1])

plt.plot(fpg,tpg)
plt.plot([0, 1], [0, 1], color='purple', linestyle='-.')
plt.xlabel('Especificidade')
plt.ylabel('Sensividade')
plt.title('ROC para Gaussiano')
plt.show()
b = auc(fpb, tpb)
g = auc(fpg, tpg)
print('Bernoulli:' + str(b))
print('Gaussiano:' + str(g))
y_pred = cross_val_predict(bnb, treinoX, treinoY, cv=10)
sklearn.metrics.fbeta_score(treinoY, y_pred, 3)
previsoes = bnb.predict(teste)
predicao = pd.DataFrame(index = teste.Id)
predicao["ham"] = previsoes
predicao.to_csv('predictions.csv')