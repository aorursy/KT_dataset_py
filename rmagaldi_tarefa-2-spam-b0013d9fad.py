#bliblioteca para facilitar as operações vetoriais e matriciais 
import numpy as np

#biblioteca para facilitar a organização do dataset
import pandas as pd

#blibliotecas de visualização gráfica
import matplotlib as plt
import seaborn as sns

#biblioteca de machine learning
import sklearn
data_raw = pd.read_csv("../input/data-spam/train_data.csv")
data_raw.head(10)
data_raw.info()
data_raw.describe()
len(data_raw) == len(data_raw.dropna())
data_raw.columns
data_raw["word_freq_money"].plot.hist(range=[0, 1])
spam = data_raw[data_raw["ham"] == False]
spam["ham"].value_counts()
ham = data_raw[data_raw["ham"] == True]
ham["ham"].value_counts()
money_spam = spam.word_freq_money.mean()
print("A porcentagem média de aparição da palavra money nos emails spam é de %.4f." %(money_spam))
money_ham = ham.word_freq_money.mean()
print("A porcentagem média de aparição da palavra money nos emails ham é de %.4f." %(money_ham))
spam.word_freq_money.plot.hist(range=[0, 1])
ham.word_freq_money.plot.hist(range=[0, 1])
free_spam = spam.word_freq_free.mean()
print("A porcentagem média de aparição da palavra free nos emails spam é de %.4f." %(free_spam))
free_ham = ham.word_freq_free.mean()
print("A porcentagem média de aparição da palavra free nos emails ham é de %.4f." %(free_ham))
george_spam = spam.word_freq_george.mean()
print("A porcentagem média de aparição da palavra george nos emails spam é de %.4f." %(george_spam))
george_ham = ham.word_freq_george.mean()
print("A porcentagem média de aparição da palavra george nos emails ham é de %.4f." %(george_ham))
exclamacao_spam = spam["char_freq_!"].mean()
print("A porcentagem média de aparição do caractere ! nos emails spam é de %.4f." %(exclamacao_spam))
exclamacao_ham = ham["char_freq_!"].mean()
print("A porcentagem média de aparição do caractere ! nos emails ham é de %.4f." %(exclamacao_ham))
sns.barplot(y="char_freq_!", data=data_raw, x="ham")
cifrao_spam = spam["char_freq_$"].mean()
print("A porcentagem média de aparição do caractere $ nos emails spam é de %.4f." %(cifrao_spam))
cifrao_ham = ham["char_freq_$"].mean()
print("A porcentagem média de aparição do caractere $ nos emails ham é de %.4f." %(cifrao_ham))
sns.barplot(y="char_freq_$", data=data_raw, x="ham")
hash_spam = spam["char_freq_#"].mean()
print("A porcentagem média de aparição do caractere # nos emails spam é de %.4f." %(hash_spam))
hash_ham = ham["char_freq_#"].mean()
print("A porcentagem média de aparição do caractere # nos emails ham é de %.4f." %(hash_ham))
sns.barplot(y="char_freq_#", data=data_raw, x="ham")
par_spam = spam["char_freq_("].mean()
print("A porcentagem média de aparição do caractere ( nos emails spam é de %.4f." %(par_spam))
par_ham = ham["char_freq_("].mean()
print("A porcentagem média de aparição do caractere ( nos emails ham é de %.4f." %(par_ham))
sns.barplot(y="char_freq_(", data=data_raw, x="ham")
pev_spam = spam["char_freq_;"].mean()
print("A porcentagem média de aparição do caractere ; nos emails spam é de %.4f." %(pev_spam))
pev_ham = ham["char_freq_;"].mean()
print("A porcentagem média de aparição do caractere ; nos emails ham é de %.4f." %(pev_ham))
sns.barplot(y="char_freq_;", data=data_raw, x="ham")
brackets_spam = spam["char_freq_["].mean()
print("A porcentagem média de aparição do caractere [ nos emails spam é de %.4f." %(brackets_spam))
brackets_ham = ham["char_freq_["].mean()
print("A porcentagem média de aparição do caractere [ nos emails ham é de %.4f." %(brackets_ham))
sns.barplot(y="char_freq_[", data=data_raw, x="ham")
media_maiusc_spam = spam.capital_run_length_average.mean()
std_average_spam = spam.capital_run_length_average.std()
max_average_spam = spam.capital_run_length_average.max()
min_average_spam = spam.capital_run_length_average.min()
print("A média do tamanho médio de sequências de caracteres maiúsculos em spam é %.2f." %(media_maiusc_spam))
print("O desvio padrão é %.2f." %(std_average_spam))
print("O máximo é %.2f." %(max_average_spam))
print("O mínimo é %.2f." %(min_average_spam))
media_maiusc_ham = ham.capital_run_length_average.mean()
std_average_ham = ham.capital_run_length_average.std()
max_average_ham = ham.capital_run_length_average.max()
min_average_ham = ham.capital_run_length_average.min()
print("A média do tamanho médio de sequências de caracteres maiúsculos em ham é %.2f." %(media_maiusc_ham))
print("O desvio padrão é %.2f." %(std_average_ham))
print("O máximo é %.2f." %(max_average_ham))
print("O mínimo é %.2f." %(min_average_ham))
sns.barplot(y="capital_run_length_average", data=data_raw, x="ham")
maior_maiusc_spam = spam.capital_run_length_longest.mean()
std_maior_spam = spam.capital_run_length_longest.std()
max_maior_spam = spam.capital_run_length_longest.max()
min_maior_spam = spam.capital_run_length_longest.min()
print("A média do tamanho da maior sequência de caracteres maiúsculos em spam é %.2f." %(maior_maiusc_spam))
print("O desvio padrão é %.2f." %(std_maior_spam))
print("O máximo é %.2f." %(max_maior_spam))
print("O mínimo é %.2f." %(min_maior_spam))
maior_maiusc_ham = ham.capital_run_length_longest.mean()
std_maior_ham = ham.capital_run_length_longest.std()
max_maior_ham = ham.capital_run_length_longest.max()
min_maior_ham = ham.capital_run_length_longest.min()
print("A média do tamanho da maior sequência de caracteres maiúsculos em ham é %.2f." %(maior_maiusc_ham))
print("O desvio padrão é %.2f." %(std_maior_ham))
print("O máximo é %.2f." %(max_maior_ham))
print("O mínimo é %.2f." %(min_maior_ham))
sns.barplot(y="capital_run_length_longest", data=data_raw, x="ham")
total_maiusc_spam = spam.capital_run_length_total.mean()
std_total_spam = spam.capital_run_length_total.std()
max_total_spam = spam.capital_run_length_total.max()
min_total_spam = spam.capital_run_length_total.min()
print("A média do número total caracteres maiúsculos em spam é %.2f." %(total_maiusc_spam))
print("O desvio padrão é %.2f." %(std_total_spam))
print("O máximo é %.2f." %(max_total_spam))
print("O mínimo é %.2f." %(min_total_spam))
total_maiusc_ham = ham.capital_run_length_total.mean()
std_total_ham = ham.capital_run_length_total.std()
max_total_ham = ham.capital_run_length_total.max()
min_total_ham = ham.capital_run_length_total.min()
print("A média do número total caracteres maiúsculos em spam é %.2f." %(total_maiusc_ham))
print("O desvio padrão é %.2f." %(std_total_ham))
print("O máximo é %.2f." %(max_total_ham))
print("O mínimo é %.2f." %(min_total_ham))
sns.barplot(y="capital_run_length_total", data=data_raw, x="ham")
spam_or_not = data_raw["ham"]
spam_or_not.value_counts()
spam_or_not.value_counts().plot(radius=1.2, kind="pie", autopct='%1.1f%%', pctdistance=0.5, colors=["lightgreen", "red"])
from sklearn.naive_bayes import GaussianNB
train_X = data_raw.drop(labels =["Id", "char_freq_;", "char_freq_(", "ham", "char_freq_#", 'word_freq_415', 'word_freq_85', 'word_freq_857', "word_freq_650"], axis=1)
train_y = data_raw.ham
train_X.head()
train_y.head()
gnb = GaussianNB()

gnb.fit(train_X, train_y)
from sklearn.model_selection import cross_val_score

scores = cross_val_score(gnb, train_X, train_y, cv=15)

print(scores)
print(scores.mean())
from sklearn.neighbors import KNeighborsClassifier
valores_obtidos = []
for i in range(1, 101):
    knn = KNeighborsClassifier(n_neighbors=i)
    scores = cross_val_score(knn, train_X, train_y, cv=10)
    valores_obtidos.append(scores.mean())
print(valores_obtidos)
maximo = 0
index = 1
for j in range(len(valores_obtidos)):
    if valores_obtidos[j] > maximo:
        maximo = valores_obtidos[j]
        index = j
print("O maior valor obtido foi de", maximo, ", relativo ao valor k =", index+1,".")
data_raw.word_freq_order.value_counts()
data_raw[data_raw.word_freq_order == 0].ham.value_counts().plot(kind="pie", autopct='%1.1f%%', pctdistance=0.5, colors=["blue", "orange"])
data_raw[data_raw.word_freq_order > 0].ham.value_counts().plot(kind="pie", autopct='%1.1f%%', pctdistance=0.5, colors=["orange", "blue"])
data_raw.word_freq_money.value_counts()
data_raw["char_freq_!"].value_counts()
data_X_freq_raw = data_raw.drop(["Id", "ham"], axis=1)
data_X_freq = data_X_freq_raw.drop(["capital_run_length_average", "capital_run_length_longest", "capital_run_length_total"], axis=1)
data_X_freq.head()
for index, row in data_X_freq.iterrows():
    for column in data_X_freq.columns:
        if row[column] > 0 :
            data_X_freq.set_value(index, column, 1)
data_X_freq.head(10)
data_X_freq.nunique()
data_freq = data_X_freq.join(data_raw.ham)
data_freq.head(10)
data_freq[data_freq.word_freq_free == 0].ham.value_counts().plot(kind="bar", color=["green", "orange"])
data_freq[data_freq.word_freq_free == 1].ham.value_counts().plot(kind="bar", color=["orange", "green"])
data_X_cap_raw = data_raw.drop(["Id", "ham"], axis=1)
data_X_cap = data_X_cap_raw[["capital_run_length_average", "capital_run_length_longest", "capital_run_length_total"]]
data_X_cap.head(10)
data_cap = data_X_cap.join(data_raw.ham)
data_cap.head()
spam.capital_run_length_average.plot.hist(range=[0,25])
ham.capital_run_length_average.plot.hist(range=[0,25])
for index, row in data_cap.iterrows():
    if row.capital_run_length_average > 7.95:
        data_cap.set_value(index, "capital_run_length_average", 1)
    else:
        data_cap.set_value(index, "capital_run_length_average", 0)
data_cap.sample(10)
data_cap.capital_run_length_average.unique()
for index, row in data_cap.iterrows():
    if row.capital_run_length_longest > 59.92:
        data_cap.set_value(index, "capital_run_length_longest", 1)
    else:
        data_cap.set_value(index, "capital_run_length_longest", 0)
data_cap.sample(10)
data_cap.capital_run_length_longest.value_counts()
for index, row in data_cap.iterrows():
    if row.capital_run_length_total > 200:
        data_cap.set_value(index, "capital_run_length_total", 1)
    else:
        data_cap.set_value(index, "capital_run_length_total", 0)
data_cap.sample(10)
data_cap.capital_run_length_total.value_counts()
binarizado = data_X_freq.join(data_cap)
binarizado["Id"] = data_raw["Id"]
binarizado.sample(10)
sns.countplot(x="word_freq_free", data=binarizado, hue="ham")
sns.countplot(x='word_freq_money', data=binarizado, hue='ham')
from sklearn.naive_bayes import BernoulliNB
train_X = binarizado.drop(labels =["Id", "ham", "capital_run_length_average", "capital_run_length_longest", "capital_run_length_total", 'char_freq_('], axis=1)
train_y = binarizado.ham
train_X.head()
train_y.head()
bnb = BernoulliNB()

bnb.fit(train_X, train_y)
scores = cross_val_score(bnb, train_X, train_y, cv=15)

print(scores)
print(scores.mean())
valores_obtidos = []
for i in range(1, 31):
    knn = KNeighborsClassifier(n_neighbors=i)
    scores = cross_val_score(knn, train_X, train_y, cv=10)
    valores_obtidos.append(scores.mean())
print(valores_obtidos)
maximo = 0
index = 1
for j in range(len(valores_obtidos)):
    if valores_obtidos[j] > maximo:
        maximo = valores_obtidos[j]
        index = j
print("O maior valor obtido foi de", maximo, ", relativo ao valor k =", index+1,".")
def f_beta_score(p, r, b):
    '''
    RETORNA O SCORE DA AVALIAÇÃO F BETA.
    
    p: precisão do classificador
    r: recall do classificador
    b: parâmetro beta do f beta score
    '''
    numerador = p*r
    denominador = r + p * b**2
    fator = 1 + b**2
    
    return fator * numerador / denominador
from sklearn.model_selection import train_test_split
ev_train_raw, ev_test_raw = train_test_split(binarizado, test_size=0.2, random_state=30)
ev_train_X = ev_train_raw.drop(labels =["Id", "ham", "capital_run_length_average", "capital_run_length_longest", "capital_run_length_total", 'char_freq_('], axis=1)
ev_train_y = ev_train_raw.ham

ev_test_X = ev_test_raw.drop(labels =["Id", "ham", "capital_run_length_average", "capital_run_length_longest", "capital_run_length_total", 'char_freq_('], axis=1)
ev_test_y = ev_test_raw.ham
ev_bern = BernoulliNB()

ev_bern.fit(train_X, train_y)
scores = cross_val_score(ev_bern, ev_train_X, ev_train_y, cv=15)

print(scores)
print(scores.mean())
ev_pred = ev_bern.predict(ev_test_X)
from sklearn.metrics import accuracy_score
print("A acurácia foi de", accuracy_score(ev_test_y, ev_pred))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ev_test_y, ev_pred)
tn, fp, fn, tp = confusion_matrix(ev_test_y, ev_pred).ravel()
cm
tn, fp, fn, tp 
precisao = tp/(tp+fp)
print("A precisão foi de %.3f" %(precisao))
recall = tp/(tp+fn)
print("O recall foi de %.3f" %(recall))
f_3 = f_beta_score(precisao, recall, 3)
print("O score F3 foi de %.3f" %(f_3))
test_raw = pd.read_csv("../input/data-spam/test_features.csv")
test_raw.head()
test = test_raw.drop(labels =["Id", "char_freq_(", "capital_run_length_average", "capital_run_length_longest", "capital_run_length_total"], axis=1)
prediction = bnb.predict(test)
envio_raw = pd.DataFrame()
envio_raw['Id'] = test_raw.Id
envio_raw['ham'] = prediction
envio = envio_raw.set_index("Id")
envio.to_csv("prediction.csv")