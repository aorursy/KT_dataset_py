import numpy as np 

import pandas as pd 

import seaborn as sns

from scipy.stats import skew , kurtosis 

from sklearn.preprocessing import scale

import matplotlib.pyplot as plt



import os

print(os.listdir("../input/heart-disease-uci"))

doenca_coracao = pd.read_csv('../input/heart-disease-uci/heart.csv')

doenca_coracao.head()
# Visualiza os tipos de dados que compoem a tabela

doenca_coracao.info()
doenca_coracao= doenca_coracao.rename(columns= {'age':'idade', 'sex': 'sexo' , 'cp': 'dor_no_peito' , 'trestbps': 'pressao_sanguinea_repouso' , 'chol' : 'colesterol',

                                             'fbs': 'glicose_jejum' , 'restecg' : 'resultado_ecg' ,'thalach' : 'frequencia_cardíaca_maxima',

                                             'exang' : 'angina_induzida_exercicio' , 'oldpeak' : 'st_depressao' , 'slope' : 'st_inclinacao',

                                             'ca' : 'qtd_vasos' , 'thal' : 'talassemia','target': 'alvo'})
# Exibir as 10 primeiras linhas no conjunto de dados

doenca_coracao.head(10)
# Exibir as últimas 10 linhas no conjunto de dados

doenca_coracao.tail(10)
# Verifique se há valores nulos nos dados

doenca_coracao.isnull().sum()
# Tenta encontrar alguma correlação obvia

sns.pairplot(data=doenca_coracao)
# Tenta encontrar alguma correlação evidente

sns.pairplot(data=doenca_coracao,hue="alvo",palette="bright")
sns.pairplot(data=doenca_coracao,vars=["st_depressao", "frequencia_cardíaca_maxima"],hue="alvo",palette="bright")
# Cria um diagrama de correlação

plt.figure(figsize=(14,10))

sns.heatmap(doenca_coracao.corr(),annot=True,cmap='hsv',fmt='.2f',linewidths=2)

plt.show()
x = doenca_coracao.drop(['alvo'], axis = 1)

y = doenca_coracao.alvo.values
# Import Libraries

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.80)
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()
# Passa os dataframes para ajuste do modelo

logmodel.fit(x_train,y_train)
# Cria o modelo preditivo

LR_pred = logmodel.predict(x_test)

LR_pred
# Verifica a acurácia

from sklearn.metrics import accuracy_score

LR_accuracy = accuracy_score(LR_pred,y_test)

LR_accuracy
from sklearn.neighbors import KNeighborsClassifier

classifier2 = KNeighborsClassifier(n_neighbors=3)

classifier2
# Passa os dataframes para ajuste do modelo

classifier2.fit(x_train,y_train)
# Cria o modelo preditivo

knn_pred = classifier2.predict(x_test)

knn_pred
# Verifica a acurácia

from sklearn.metrics import accuracy_score

accuracy_knn=accuracy_score(knn_pred,y_test)

accuracy_knn
from sklearn.naive_bayes import GaussianNB

classifier3 = GaussianNB()

classifier3
# Passa os dataframes para ajuste do modelo

classifier3.fit(x_train,y_train)
# Cria o modelo preditivo

NBC_pred = classifier3.predict(x_test)

NBC_pred
# Verifica a acurácia

from sklearn.metrics import accuracy_score

NBC_accuracy = accuracy_score(NBC_pred,y_test)

NBC_accuracy
from sklearn.tree import DecisionTreeClassifier

classifier4 = DecisionTreeClassifier(criterion='entropy',random_state=0)

classifier4
# Passa os dataframes para ajuste do modelo

classifier4.fit(x_train,y_train)
# Cria o modelo preditivo

DT_pred = classifier4.predict(x_test)

DT_pred
# Verifica a acurácia

from sklearn.metrics import accuracy_score

accuracy_DT = accuracy_score(DT_pred,y_test)

accuracy_DT