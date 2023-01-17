# Funções antigas
%matplotlib inline
import pandas as pd
import sklearn
import matplotlib
import matplotlib.pyplot as plt

# FUnções novas
import numpy as np
from collections import Counter
from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm
from IPython.display import Image
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
dados = pd.read_csv("../input/tarefa-2/train_data.csv", na_values="?")
dados.head()
dados["ham"].value_counts().plot(kind="bar")
plt.title('Comparação entre número de emails desejáveis (True) e spam (False)')
plt.ylabel('Número de emails observados')
plt.xlabel('Tipo de email')
dados["ham"].isnull().sum()
dados1=dados.drop("Id", axis=1)
correla=(dados1.corr()["ham"])
correla
features_treino=dados.drop(columns=['ham'])
label_treino=dados['ham']
clasif = GaussianNB()
clasif.fit(features_treino, label_treino)
grupos = cross_val_score(clasif,features_treino,label_treino,cv=10) # validacao cruzada, geracao de vetor com grupos
grupos.mean()
dados.teste = pd.read_csv("../input/tarefa-2/test_features.csv")
Id = dados.teste["Id"]
dados.teste.head()
pred = clasif.predict(dados.teste)
pred
d = {'Id' : Id, 'ham' : pred}
my_df = pd.DataFrame(d)
my_df.head()
my_df.to_csv('pred.csv',index=False, sep=',', line_terminator='\n', header = ["Id", "ham"])
my_df.head()
