# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



# Any results you write to the current directory are saved as output.
# Import das bibliotecas necessárias para rodar o programa

import numpy as np

import matplotlib.pyplot as plt

import sklearn.metrics

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score

import graphviz

from sklearn import tree

from sklearn.tree import export_graphviz

from sklearn.metrics import roc_auc_score,roc_curve
# Lendo os dados do Dataset

dados = pd.read_csv('weatherAUS.csv')

print(dados.shape)
# Visualizando as primeiras 5 instâncias do Dataset

dados.head()
# Analisando mais detalhadamente cada coluna do Dataset

dados.describe()
# Verificando o tipo dos dados presentes na tabela

dados.dtypes
# Descobrindo se há dados faltantes, caso seja True pode-se afirmar que existem dados faltantes na respectiva coluna

dados.isnull().any()
# Colocando as colunas com dados faltantes em uma lista para serem tratadas futuramente. Nessa lista são colocadas

# apenas as colunas com dados numéricos

colunas_dados_faltantes = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', \

                           'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', \

                           'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', \

                           'Cloud3pm', 'Temp9am', 'Temp3pm']



# Substituindo valores numéricos com a média dos valores não nulos presentes na coluna

for coluna in colunas_dados_faltantes:

    dados[coluna] = dados[coluna].fillna(dados[coluna].mean())

    

# Colocando as colunas com dados faltantes em uma lista. Nessa lista são colocadas as colunas com dados categóricos

colunas_dados_categoricos = ['WindGustDir', 'WindDir3pm', 'WindDir9am']



# Substituindo valores categóricos pela moda dos valores não nulos presentes na coluna

for coluna in colunas_dados_categoricos:

    dados[coluna] = dados[coluna].fillna(dados[coluna].mode()[0])



# Verificando se ainda existem dados faltantes no Dataset

dados.isnull().any()
# Substituindo os valores de objeto dos atributos 'RainToday' pra 0 e 1 (antes eram "Yes" e "No")

# e substituindo valores faltantes por 0

dados['RainToday'] = dados['RainToday'].replace({'No': 0, 'Yes': 1}).fillna(0)
# Substituindo No e Yes por 0 e 1 para o atributo alvo

dados['RainTomorrow'] = dados['RainTomorrow'].replace({'No': 0, 'Yes': 1})
# Verificando quantas cidades existem no Dataset

len(set(dados['Location']))
# Escolhemos LabelEncoder porque há 49 localizações diferentes e, portanto, 

# seriam necessários mais 49 atributos utilizando One-HotEncoder



from sklearn import preprocessing



colunas_dados_categoricos.append('Location')



for coluna in colunas_dados_categoricos:

    le = preprocessing.LabelEncoder()

    le.fit(dados[coluna])

    dados[coluna] = le.transform(dados[coluna])
# Visualizando as primeiras 5 instâncias do Dataset, para analisar as mudanças feitas

dados.head()
# Novamente analisando mais detalhadamente cada coluna do Dataset, agora sem os dados faltantes

dados.describe()
# Analisando a correlação dos dados do Dataset

# Para isso, plotamos a matriz de correlação entre os valores numéricos



rain_data_num = dados[['MinTemp','MaxTemp','Rainfall','WindSpeed9am','WindSpeed3pm',

                           'Humidity9am','Humidity3pm','Pressure9am','Pressure3pm',

                           'Temp9am','Temp3pm','RainToday','RainTomorrow', 'RISK_MM']]

plt.figure(figsize=(12,8))

sns.heatmap(rain_data_num.corr(),annot=True,cmap='bone',linewidths=0.25)
# Desconsiderando a coluna Temp9am

dados = dados.drop(columns=['Temp9am', 'MaxTemp', 'MinTemp'])
# Desconsiderando a coluna: RISK_MM

# Note: You should exclude the variable Risk-MM when training a binary classification model. 

# Not excluding it will leak the answers to your model and reduce its predictability.

dados = dados.drop(columns=['RISK_MM'])
rain_data_num = dados[['Rainfall','WindSpeed9am','WindSpeed3pm',

                           'Humidity9am','Humidity3pm','Pressure9am','Pressure3pm',

                           'Temp3pm','RainToday','RainTomorrow']]

plt.figure(figsize=(12,8))

sns.heatmap(rain_data_num.corr(),annot=True,cmap='bone',linewidths=0.25)
#Plotando gráficos relacionandos à coluna Humidity3pm com RainTomorrow para analisar como os dados da humidade 

# do ar as 3pm se relacionam com a probabilidade de chover amanhã.



sns.catplot(x='RainTomorrow', y='Humidity3pm', hue='RainTomorrow',

            kind="violin", split=False, data=dados);
from statistics import stdev

# Divisão dos dados para treinamento e teste

Y = dados.pop('RainTomorrow').values

X = dados.drop(columns=['Date']).values



x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20,random_state=42)



print(x_train.shape)

print(x_test.shape)
from sklearn.naive_bayes import GaussianNB

modelo = GaussianNB()

modelo.fit(x_train, y_train)

Score_1=modelo.score(x_test, y_test)



# Cálculo de Acurácia

print('Acurácia do modelo Naive-Bayes utilizando holdout de 20%%: %.4f%%' % (Score_1*100))



score_3 = cross_val_score(modelo, x_test, y_test, cv=10)

print('Acurácia do modelo Naive Bayes utilizando 10-fold: %.4f%% +- %.4f%%' % (score_3.mean()*100, stdev(score_3)*100))
# Matriz de Confusão (Naive Bayes)

targetnames = ['RainTomorrow ','Not RainTomorrow']



y_pred = modelo.predict(x_test)

confusion_matrix= sklearn.metrics.confusion_matrix(y_test, y_pred)



df_cm = pd.DataFrame(confusion_matrix, index = [i for i in targetnames], columns = [i for i in targetnames])

print(confusion_matrix)



cmap = sns.light_palette("navy", as_cmap=True)

plt.figure(figsize=(8, 6))

plt.title('Confusion matrix of the classifier')

sns.heatmap(df_cm, annot=True, cmap=cmap)
# Algoritmo de Classificação Árvore de Decisão



modelo_2 = DecisionTreeClassifier(criterion='entropy')

modelo_2.fit(x_train, y_train)



modelo_2_Score_1=modelo_2.score(x_test, y_test)

print('Acurácia do modelo Decision Tree utilizando holdout de 20%%: %.4f%%' % (modelo_2_Score_1*100))

print(modelo_2_Score_1)



modelo_2_score_2 = cross_val_score(modelo_2, x_test, y_test, cv=10)

print('Acurácia do modelo Decision Tree utilizando cross validation 10-fold: %.4f%% +- %.4f%%' % (modelo_2_score_2.mean()*100, stdev(modelo_2_score_2)*100))

# Plotando a Árvore de Decisão



feature_names = ['Date','Location','Rainfall','Evaporation','Sunshine','WindGustDir',

                 'WindGustSpeed','WindDir9am','WindDir3pm','WindSpeed9am','WindSpeed3pm','Humidity9am',

                 'Humidity3pm','Pressure9am','Pressure3pm','Cloud9am','Cloud3pm','Temp3pm']

target_names = ['Rain', 'Not Rain']



export_graphviz(modelo_2, out_file='tree.dot', feature_names=feature_names, 

                class_names=target_names, filled=True, rounded=True,special_characters=True)



from subprocess import call

call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=200'])



from IPython.display import Image

Image(filename = 'tree.png')
# Matriz de Confusão (Decision Tree)



targetnames = ['RainTomorrow ','Not RainTomorrow']



y_pred_2 = modelo_2.predict(x_test)

confusion_matrix= sklearn.metrics.confusion_matrix(y_test, y_pred_2)



print(confusion_matrix)



df_cm = pd.DataFrame(confusion_matrix, index = [i for i in targetnames], columns = [i for i in targetnames])

cmap = sns.light_palette("navy", as_cmap=True)



plt.figure(figsize=(8, 6))

plt.title('Confusion matrix of the classifier')

sns.heatmap(df_cm, annot=True, cmap=cmap)
from sklearn.svm import SVC



modelo_3 = SVC(kernel='rbf', gamma='scale')

modelo_3.fit(x_train, y_train)



modelo_3_score1 = modelo_3.score(x_test, y_test)

modelo_3_score2 = cross_val_score(modelo_3, x_test, y_test, cv=10)



print('Acurácia do modelo SVM utilizando holdout de 20%%: %.4f%%' % (modelo_3_score1*100))

print('Acurácia do modelo SVM utilizando cross validation 10-fold: %.4f%% +- %.4f%%' % (modelo_3_score2.mean()*100, stdev(modelo_3_score2)*100))
targetnames = ['RainTomorrow ','Not RainTomorrow']



y_pred = modelo_3.predict(x_test)

confusion_matrix= sklearn.metrics.confusion_matrix(y_test, y_pred)



df_cm = pd.DataFrame(confusion_matrix, index = [i for i in targetnames], columns = [i for i in targetnames])

print(confusion_matrix)



cmap = sns.light_palette("navy", as_cmap=True)

plt.figure(figsize=(8, 6))

plt.title('Confusion matrix of the classifier')

sns.heatmap(df_cm, annot=True, cmap=cmap)