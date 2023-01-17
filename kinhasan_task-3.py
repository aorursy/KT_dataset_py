import pandas as pd # Trabalhar os dados
import matplotlib as matplot # Produzir graficos e fazer uma nalise mais vizual dos dados
import numpy as np # biblioteca que trabalha em conjunto com a matplot
import sklearn as skl 
base_train = pd.read_csv("../input/train.csv") #lendo a base de treino
base_test = pd.read_csv("../input/test.csv") #lendo a base de teste
#dimensoes da base de treino
base_train.shape
#Lendo as primeiras 25 linhas
base_train_twentyfive = pd.read_csv("../input/train.csv", nrows=25)
#Aqui eu dropei o Id porque ele nao tem muita funcao em nossa tabela, alem da idenftificacao
base_train_twentyfive.drop('Id',axis=1)
import seaborn as sns #essa bibliioteca nos ajudara a trabalhar com a matplot
get_ipython().run_line_magic('matplotlib', 'inline')
#distribuicao das casas por latitude e longitude
sns.pairplot(base_train, x_vars=['longitude'], y_vars=['latitude'], size=13, aspect=1.3, kind='reg')
#relacao entre preço da casa e latitude
sns.pairplot(base_train, x_vars=['median_house_value'], y_vars=['latitude'], size=14, aspect=1.4)
#relacao entre preço da casa e longitude
sns.pairplot(base_train, x_vars=['median_house_value'], y_vars=['longitude'], size=14, aspect=1.4)
#Amostra da base de treino
base_test.head(25)
#dimensoes da base de teste
base_test.shape
#selecionando as features
Xtrain= base_train[['households','population','median_income','latitude','longitude']]
#Target
Ytrain=base_train.median_house_value
#fazendo um match das features selecionadas na base de treino
Xtest=base_test[['households','population','median_income','latitude','longitude']]
#importanto o KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
#atribuindo valores de para o numero de neighbors e cross validacoes
knn=KNeighborsClassifier(n_neighbors=5)
scores=cross_val_score(knn,Xtrain,Ytrain,cv=20)
knn.fit(Xtrain,Ytrain)
scores
#predicao
Ytestpred=knn.predict(Xtest)
Ytestpred
#A acuracia do moledo com knn foi baixissima, irei tentar agora com Naive Bayes
from sklearn.naive_bayes import GaussianNB

features_train = base_train.drop(columns=['median_house_value'])
target_train = base_train['median_house_value']
gnb = GaussianNB()

gnb.fit(features_train, target_train)
#preparando para fazer a validação cruzada
from sklearn.model_selection import cross_val_score
lista1 = []
scores = cross_val_score(gnb, features_train, target_train, cv=40)
scores
