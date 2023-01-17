#Importação das bibliotecas para trabalhar com os dados e iniciar as previsões
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
#Importação dos dados
treino = pd.read_csv("../input/train_data.csv" ,  engine = "python")
teste = pd.read_csv("/../input/test_features.csv" , engine = "python")
#Abaixo a impressão dos dados das bases de teste e de treino para avaliação sobre as features que serão utilizadas
treino.head()
treino.info()
treino.describe()
teste.head()
#Biblioteca auxiliar para apresentação de dados
import seaborn
%matplotlib inline
seaborn.pairplot(treino,x_vars=['char_freq_$','word_freq_remove'],y_vars='ham',size=8, aspect=0.8)
#Interação entre ham e spam
seaborn.pairplot(treino,x_vars=['word_freq_000','char_freq_$'],y_vars='ham',size=8, aspect=0.8)
seaborn.pairplot(treino,x_vars=['word_freq_people','char_freq_!'],y_vars='ham',size=8, aspect=0.8)
#Naive Bayes
from sklearn.naive_bayes import GaussianNB

features_train = treino.drop(columns=['ham'])
target_train = treino['ham']
gnb = GaussianNB()

gnb.fit(features_train, target_train)
from sklearn.model_selection import cross_val_score
lista1 = []
scores = cross_val_score(gnb, features_train, target_train, cv=50)
   
print(scores)
#A impressão dos scores é meramente para criar uma visualização a respeito da qualidade das previsões
#Estabelecendo as previsões

predictions = gnb.predict(teste)
str(predictions)

#Criando um Panda DataFrame

df_entrega = pd.DataFrame(predictions)

#Exportando os resultados numa planilha

df_entrega.to_csv('predictions.csv')
print(predictions)
#Referências para melhor analisar os dados

#'As 100 palavras que podem fazer seu email cair na caixa de spam' - http://www.emailmarketingblog.com.br/100-palavras-consideradas-como-spam/#.W7q-evZRdPY
#'A list of common spam words' - https://emailmarketing.comm100.com/email-marketing-ebook/spam-words.aspx
