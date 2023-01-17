#Imports
import pandas as pd
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import preprocessing

#Base
data = pd.read_csv("../input/spambase/train_data.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
data.columns
import matplotlib.pyplot as plt
# Vamos analisar apenas as colunas de frequência, sem os atributos que nao tem nada a ver com frequencia
freq_data = data[[c for c in data.columns if 'freq' in c]] 
# Vamos somar todas as frequencias das ocorrencias e ordenar, para sabermos quais colunas tem mais
# ocorrências e quais tem menos
plt.figure(figsize=(15,5))
plt.title('Frequência acumulada das palavras ao longo de todas as observações', size=20)
plt.ylabel('Frequência acumulada', size=15)
plot = freq_data.sum().plot(kind='bar')

# Agora vamos à ordenação dos dados
sorted_columns = np.argsort(np.array(freq_data.sum()))
print("Colunas ordenadas por ordem de menor para a maior frequência:")
print("---------------")
freq_data.columns[sorted_columns]
#Separação das features e do target
X_train = data[[c for c in data.columns if c not in ['Id','ham']]]
Y_train = data['ham']
Ids_train = data['Id']
print('A porcentagem de HAM na base de treino eh: ' , Y_train[Y_train==True].shape[0]/Y_train.shape[0])
print('Esse pode ser considerado nosso baseline para acuracia')
#Primeiro teste
print('Primeiro teste. Usando todas as ',X_train.columns.shape[0], 'colunas')
knn = KNeighborsClassifier(n_neighbors=1)
scores = cross_val_score(knn, X_train, Y_train, cv=10)
print('Acuracia knn (k=1): ', scores.mean())

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
scores = cross_val_score(gnb, X_train, Y_train, cv=10)
print('Acuracia naive bayes: ', scores.mean())
##NAIVE BAYES
gnb = GaussianNB()
score_list=[]
for k in range(1,58):
    X_train_new = SelectKBest(chi2, k=k).fit_transform(X_train, Y_train)
    scores = cross_val_score(gnb, X_train_new, Y_train, cv=10)
    score_list.append(scores.mean())

plt.title('Seleção do número de features', size=13)
plt.ylabel('Accuracy')
plt.xlabel('Número de features')
plot = plt.plot(score_list)
print('Usando Naive Bayes')
print('A maior acuracia foi: ', np.max(score_list),'e ocorreu com o uso de',np.argmax(score_list),'features')
X_train_new = SelectKBest(chi2, k=23).fit_transform(X_train, Y_train)
score_list=[]
for K_features in np.arange(1,100,5):
    knn = KNeighborsClassifier(n_neighbors=K_features)
    scores = cross_val_score(knn, X_train_new, Y_train, cv=10)
    score_list.append(scores.mean())
print('Usando KNN')
print('A maior acuracia foi: ', np.max(score_list),'e ocorreu com o uso de num_neighbors=',np.arange(1,100,5)[np.argmax(score_list)])
