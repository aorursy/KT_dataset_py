# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

from sklearn import metrics

from time import time
df = pd.read_csv("../input/credit_train.csv")

df.head()
df.describe()

plt.figure(figsize=(30,30))

sns.heatmap(df.corr(),annot=True,linewidths=0.01 ,cmap='coolwarm', cbar=True)

plt.show()
features =['Number of Credit Problems','Bankruptcies','Tax Liens','Loan Status']

dfRisco = df[features]

dfRisco.head()
dfRisco.describe()
dfRisco.isnull()

dfRiscoLimpo = dfRisco[df['Number of Credit Problems'].notnull() & df['Bankruptcies'].notnull() & df['Tax Liens'].notnull() & df['Loan Status'].notnull()]

dfRiscoLimpo.describe()
sns.pairplot(data=dfRiscoLimpo, hue='Loan Status')
X = dfRiscoLimpo[features[:3]]

y = dfRiscoLimpo[features[3:]]

X.head()
#Dividir dados de treinamento e teste

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.20)

print(len(x_train),len(x_test))
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()



tempoTreinamento = time()

nb.fit(x_train,y_train)

tempoTreinamento = time() - tempoTreinamento



tempoPredicao = time()

predicao = nb.predict(x_test)

tempoPredicao = time() - tempoPredicao



acuracia = metrics.accuracy_score(predicao,y_test)



comparaClassificadores = pd.DataFrame([['Naive Bayes',acuracia,tempoTreinamento,tempoPredicao]],columns=['Classificador', 'Acurácia', '(t)Treinamento','(t)Predição'])



print('A acurácia do naive bayes foi: ',acuracia)
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()



tempoTreinamento = time()

dtc.fit(x_train,y_train)

tempoTreinamento = time() - tempoTreinamento



tempoPredicao = time()

predicao = dtc.predict(x_test)

tempoPredicao = - time() - tempoPredicao 



acuracia = metrics.accuracy_score(predicao,y_test)



comparaClassificadores.append(pd.DataFrame([['DecisionTreeClassifier',acuracia,tempoTreinamento,tempoPredicao]]))



print('A acurácia do DecisionTreeClassifier foi: ',metrics.accuracy_score(predicao,y_test))
from sklearn import svm

svc = svm.SVC()



tempoTreinamento = time()

svc.fit(x_train,y_train)

tempoTreinamento = time() - tempoTreinamento



tempoPredicao = time()

predicao = svc.predict(x_test)

tempoPredicao = tempoPredicao - time()



acuracia = metrics.accuracy_score(predicao,y_test)



comparaClassificadores.append([['SVC',acuracia,tempoTreinamento,tempoPredicao]])



print('A acurácia do SVC foi: ',metrics.accuracy_score(predicao,y_test))
comparaClassificadores
dfRiscoLimpo[:3]
parametros = [[1.0,0.0,0.0]]

print("Risco de pagamento: " + svc.predict(parametros))