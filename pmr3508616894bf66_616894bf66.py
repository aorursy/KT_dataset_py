import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import fbeta_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('../input/train-data/train_data.csv')
train.shape
train.head()
w = train.columns[0:54]
words = []

#dividir a palavra pelo caracter "_" e usar somente o ultimo termo apos a divisao
for i in w:
    x = i.split("_")[2]
    words.append(x)

#renomear as colunas do data set
for i in range(len(train.columns[0:54])):
    train = train.rename(columns = {train.columns[i]: words[i]})
train.head()
freq_words = train.groupby('ham')[train.columns[0:48]].sum()

spam_freq_words = freq_words.T.sort_values(by = 0, ascending = False)[0]
not_spam_freq_words = freq_words.T.sort_values(by = 0, ascending = False)[1]

plt.figure(figsize = (17, 8))
plt.bar(spam_freq_words.index, spam_freq_words, color = 'lightblue')
plt.bar(not_spam_freq_words.index, not_spam_freq_words, color = 'lightgreen', bottom = spam_freq_words)

_ = plt.xticks(rotation = 60)
_ = plt.title('Frequencia de palavras', fontsize = 20)
_ = plt.legend(labels = ["Spam", "Not Spam"], fontsize = 20)
freq_chars = train.groupby('ham')[train.columns[48:54]].sum()

spam_freq_chars = freq_chars.T.sort_values(by = 0, ascending = False)[0]
not_spam_freq_chars = freq_chars.T.sort_values(by = 0, ascending = False)[1]

plt.figure(figsize = (10, 8))
plt.bar(spam_freq_chars.index, spam_freq_chars, color = 'lightblue')
plt.bar(not_spam_freq_chars.index, not_spam_freq_chars, color = "lightgreen", bottom = spam_freq_chars)

_ = plt.title("Frequencia de caracteres", fontsize = 20)
_ = plt.legend(labels = ["Spam", "Not Spam"], fontsize = 20)
l = train.groupby('ham')[train.columns[54:57]].sum()
l_spam = l.T.sort_values(by = 0, ascending = False)[0]
l_not_spam = l.T.sort_values(by = 0, ascending = False)[1]

plt.figure(figsize = (10,8))
plt.bar(l_spam.index, l_spam, color = 'lightblue')
plt.bar(l_not_spam.index, l_not_spam, color = 'lightgreen', bottom = l_spam)

_ = plt.title('Spam email - Outros atributos', fontsize = 20)
_ = plt.legend(labels = ["Spam", "Not Spam"], fontsize = 20)
x = pd.concat([train.iloc[:, 0:57], train['ham']], axis = 1).corr().iloc[57].sort_values()
plt.figure(figsize = (17, 8))
plt.bar(x.index[:-1], x[:-1], color = 'lightblue')
_ = plt.xticks(rotation = 'vertical')
_ = plt.title("Correlação entre Nao Spam e Atributos")
#Seprando os dados em teste e treino
Y = train[["ham"]]
X = train.iloc[:,0:57]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)

#Normalização dos dados
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
#Escolhendo o número de vizinhos para o algoritmo KNN
scores_mean = {}
for i in range(1,15):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train_scaled, Y_train)
    scores_scaled = cross_val_score(knn, X_train_scaled, Y_train, cv=10)
    scores_mean[i] = scores_scaled.mean()
    
pd.DataFrame(pd.Series(scores_mean)).sort_values(by = 0, ascending = False).T
#Treinando o algoritmo KNN
knn_1 = KNeighborsClassifier(n_neighbors = 11)
knn_1.fit(X_train_scaled, Y_train)

#Usando o algoritmo nos dados de teste
Y_predict_knn_1 = knn_1.predict(X_test_scaled)
fbeta_score(Y_test, Y_predict_knn_1, beta=3)
#Identificando as variáveis que possuem correlacao com o alvo > 0.3
high_corr_var = []
x = train.corr()[abs(train.corr()) > 0.3].iloc[57].isna()
for i in range(0,59):
    if x.iloc[i] == False: high_corr_var.append(x.index[i])
high_corr_var = high_corr_var[:-1]
high_corr_var
#Usando uma nova variavel de dados apenas com as colunas detectadas com alta correlaççao
X_train_corr = X_train[high_corr_var]
X_test_corr = X_test[high_corr_var]
#Novamente, escolhendo o número de vizinhos para o algoritmo KNN
scores_mean = {}
for i in range(1,100):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train_corr, Y_train)
    scores_scaled = cross_val_score(knn, X_train_corr, Y_train, cv=10)
    scores_mean[i] = scores_scaled.mean()
    
pd.DataFrame(pd.Series(scores_mean)).sort_values(by = 0, ascending = False).T
#Treinando o algoritmo KNN
k = pd.DataFrame(pd.Series(scores_mean)).sort_values(by = 0, ascending = False).index[0]
knn_2 = KNeighborsClassifier(n_neighbors = k)
knn_2.fit(X_train_corr, Y_train)

#Usando o algoritmo nos dados de teste
Y_predict_knn_2 = knn_2.predict(X_test_corr)
fbeta_score(Y_test, Y_predict_knn_2, beta=3)
train = pd.read_csv('../input/train-data/train_data.csv')
w = train.columns[0:54]
words = []

#dividir a palavra pelo caracter "_" e usar somente o ultimo termo apos a divisao
for i in w:
    x = i.split("_")[2]
    words.append(x)

#renomear as colunas do data set
for i in range(len(train.columns[0:54])):
    train = train.rename(columns = {train.columns[i]: words[i]})
for i in range(0,58):
    #Se o valor observado for menor que a media, substitui-se por 0
    train.iloc[:, i] [train.iloc[:, i] < train.iloc[:, i].mean()] = 0
    #Se for maior, por 1
    train.iloc[:, i] [train.iloc[:, i] >= train.iloc[:, i].mean()] = 1
#Seprando os dados em teste e treino
Y = train[["ham"]]
X = train.iloc[:,0:57]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)

#Normalização dos dados
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
#Novamente, escolhendo o número de vizinhos para o algoritmo KNN
scores_mean = {}
for i in range(1,100):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train_corr, Y_train)
    scores_scaled = cross_val_score(knn, X_train_corr, Y_train, cv=10)
    scores_mean[i] = scores_scaled.mean()
    
pd.DataFrame(pd.Series(scores_mean)).sort_values(by = 0, ascending = False).T
#Treinando o algoritmo KNN
k = pd.DataFrame(pd.Series(scores_mean)).sort_values(by = 0, ascending = False).index[0]
knn_3 = KNeighborsClassifier(n_neighbors = k)
knn_3.fit(X_train_corr, Y_train)

#Usando o algoritmo nos dados de teste
Y_predict_knn_3 = knn_3.predict(X_test_corr)
fbeta_score(Y_test, Y_predict_knn_3, beta=3)
train = pd.read_csv('../input/train-data/train_data.csv')
w = train.columns[0:54]
words = []

#dividir a palavra pelo caracter "_" e usar somente o ultimo termo apos a divisao
for i in w:
    x = i.split("_")[2]
    words.append(x)

#renomear as colunas do data set
for i in range(len(train.columns[0:54])):
    train = train.rename(columns = {train.columns[i]: words[i]})
for i in range(0,58):
    #Se o valor observado for menor que a media, substitui-se por 0
    train.iloc[:, i] [train.iloc[:, i] < train.iloc[:, i].mean()] = 0
    #Se for maior, por 1
    train.iloc[:, i] [train.iloc[:, i] >= train.iloc[:, i].mean()] = 1
#Seprando os dados em teste e treino
Y = train[["ham"]]
X = train.iloc[:,0:57]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)

#Normalização dos dados
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
#Identificando as variáveis que possuem correlacao com o alvo > 0.4
high_corr_var = []
x = train.corr()[abs(train.corr()) > 0.4].iloc[57].isna()
for i in range(0,59):
    if x.iloc[i] == False: high_corr_var.append(x.index[i])
high_corr_var = high_corr_var[:-1]
high_corr_var
#Usando uma nova variavel de dados apenas com as colunas detectadas com alta correlaççao
X_train_corr = X_train[high_corr_var]
X_test_corr = X_test[high_corr_var]
#Novamente, escolhendo o número de vizinhos para o algoritmo KNN
scores_mean = {}
for i in range(1,100):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train_corr, Y_train)
    scores_scaled = cross_val_score(knn, X_train_corr, Y_train, cv=10)
    scores_mean[i] = scores_scaled.mean()

pd.DataFrame(pd.Series(scores_mean)).sort_values(by = 0, ascending = False).T
#Treinando o algoritmo KNN
k = pd.DataFrame(pd.Series(scores_mean)).sort_values(by = 0, ascending = False).index[0]
knn_4 = KNeighborsClassifier(n_neighbors = k)
knn_4.fit(X_train_corr, Y_train)

#Usando o algoritmo nos dados de teste
Y_predict_knn_4 = knn.predict(X_test_corr)
fbeta_score(Y_test, Y_predict_knn_4, beta=3)
train = pd.read_csv('../input/train-data/train_data.csv')
w = train.columns[0:54]
words = []

#dividir a palavra pelo caracter "_" e usar somente o ultimo termo apos a divisao
for i in w:
    x = i.split("_")[2]
    words.append(x)

#renomear as colunas do data set
for i in range(len(train.columns[0:54])):
    train = train.rename(columns = {train.columns[i]: words[i]})
#Seprando os dados em teste e treino
Y = train[["ham"]]
X = train.iloc[:,0:57]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)

#Normalização dos dados
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
#Treinando o modelo
nb_1 = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
nb_1.fit(X_train, Y_train)

#Testando o modelo
Y_predict_nb_1 = nb_1.predict(X_test)
fbeta_score(Y_test, Y_predict_nb_1, beta=3)
train = pd.read_csv('../input/train-data/train_data.csv')
w = train.columns[0:54]
words = []

#dividir a palavra pelo caracter "_" e usar somente o ultimo termo apos a divisao
for i in w:
    x = i.split("_")[2]
    words.append(x)

#renomear as colunas do data set
for i in range(len(train.columns[0:54])):
    train = train.rename(columns = {train.columns[i]: words[i]})
#Identificando as variáveis que possuem correlacao com o alvo > 0.3
high_corr_var = []
x = train.corr()[abs(train.corr()) > 0.3].iloc[57].isna()
for i in range(0,59):
    if x.iloc[i] == False: high_corr_var.append(x.index[i])
high_corr_var = high_corr_var[:-1]
high_corr_var
#Usando uma nova variavel de dados apenas com as colunas detectadas com alta correlaççao
X_train_corr = X_train[high_corr_var]
X_test_corr = X_test[high_corr_var]
#Treinando o modelo
nb_2 = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
nb_2.fit(X_train_corr, Y_train)

#Testando o modelo
Y_predict_nb_2 = nb_2.predict(X_test_corr)
fbeta_score(Y_test, Y_predict_nb_2, beta=3)
train = pd.read_csv('../input/train-data/train_data.csv')
w = train.columns[0:54]
words = []

#dividir a palavra pelo caracter "_" e usar somente o ultimo termo apos a divisao
for i in w:
    x = i.split("_")[2]
    words.append(x)

#renomear as colunas do data set
for i in range(len(train.columns[0:54])):
    train = train.rename(columns = {train.columns[i]: words[i]})
for i in range(0,58):
    #Se o valor observado for menor que a media, substitui-se por 0
    train.iloc[:, i] [train.iloc[:, i] < train.iloc[:, i].mean()] = 0
    #Se for maior, por 1
    train.iloc[:, i] [train.iloc[:, i] >= train.iloc[:, i].mean()] = 1
#Treinando o modelo
nb_3 = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
nb_3.fit(X_train, Y_train)

#Testando o modelo
Y_predict_nb_3 = nb_3.predict(X_test)
fbeta_score(Y_test, Y_predict_nb_3, beta=3)
train = pd.read_csv('../input/train-data/train_data.csv')
w = train.columns[0:54]
words = []

#dividir a palavra pelo caracter "_" e usar somente o ultimo termo apos a divisao
for i in w:
    x = i.split("_")[2]
    words.append(x)

#renomear as colunas do data set
for i in range(len(train.columns[0:54])):
    train = train.rename(columns = {train.columns[i]: words[i]})
for i in range(0,58):
    #Se o valor observado for menor que a media, substitui-se por 0
    train.iloc[:, i] [train.iloc[:, i] < train.iloc[:, i].mean()] = 0
    #Se for maior, por 1
    train.iloc[:, i] [train.iloc[:, i] >= train.iloc[:, i].mean()] = 1
#Seprando os dados em teste e treino
Y = train[["ham"]]
X = train.iloc[:,0:57]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)

#Normalização dos dados
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
#Identificando as variáveis que possuem correlacao com o alvo > 0.4
high_corr_var = []
x = train.corr()[abs(train.corr()) > 0.4].iloc[57].isna()
for i in range(0,59):
    if x.iloc[i] == False: high_corr_var.append(x.index[i])
high_corr_var = high_corr_var[:-1]
high_corr_var
#Usando uma nova variavel de dados apenas com as colunas detectadas com alta correlaççao
X_train_corr = X_train[high_corr_var]
X_test_corr = X_test[high_corr_var]
#Treinando o modelo
nb_4 = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
nb_4.fit(X_train_corr, Y_train)

#Testando o modelo
Y_predict_nb_4 = nb_4.predict(X_test_corr)
fbeta_score(Y_test, Y_predict_nb_4, beta=3)