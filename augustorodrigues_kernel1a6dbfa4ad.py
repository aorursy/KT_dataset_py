import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
%matplotlib inline
df = pd.read_csv('../input/dengue-dataset.csv')
df.head()
df.tail()
df.shape
df.isnull().any()
df2 = df.copy()
df2['ocorrencia-casos'] = list(map(lambda x : 1 if x != 0 else 0, df['casos-confirmados']))
df2.head()
df2.describe()
corr = df2.corr()
fig, ax = plt.subplots(figsize = (13,13))
ax.matshow(corr)
plt.xticks(range(len(corr.columns)), corr.columns)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.show()
corr
plt.figure(figsize=(10,8))
sns.barplot(df2['ocorrencia-casos'], df2['chuva'])
plt.xlabel('Teve caso de dengue', fontsize = 14)
plt.ylabel('Quantidade de chuva', fontsize = 14)
plt.title('Quantidade de chuva quando ocorreu casos de dengue e quando não ocorreu', fontsize = 14)
plt.show()
df2.head()
df2['temperatura-media-arredondada'] = df2['temperatura-media'].round()
df2.head()
fig, ax = plt.subplots(figsize = (10,8))
sns.barplot(x='temperatura-media-arredondada', y='chuva', hue='ocorrencia-casos', palette="husl", data=df2)
ax.set_ylabel('Chuva média', fontdict={'size':14})
ax.set_xlabel('temperatura média arredondada', fontdict={'size':14})
ax.set_title('Temperatura média arredondada por Chuva média e Ocorrencia de casos', fontdict={'size':14})
ax.legend()
plt.show()
df2 = pd.concat([pd.DataFrame([x[:2] for x in df2['data'].str.split('-').values.tolist()], columns=['Ano', 'Mes']), df2], axis=1)
df2.head()
fig, ax = plt.subplots(figsize=(10,8))
sns.barplot(x = 'Ano', y = 'chuva', hue = 'ocorrencia-casos', data=df2, palette='husl')
ax.set_xlabel('Ano', fontdict = {'size' : 14})
ax.set_ylabel('Chuva Média', fontdict = {'size' : 14})
ax.set_title('Ano por chuva média e ocorrencia de casos de dengue', fontdict = {'size' : 14})
ax.legend()
plt.show()
fig, ax = plt.subplots(figsize = (10,8))
sns.barplot(x = 'Mes', y = 'chuva', hue = 'ocorrencia-casos', data = df2, palette='husl')
ax.set_xlabel('Mes', fontdict = {'size' : 14})
ax.set_ylabel('Chuva Média', fontdict = {'size' : 14})
ax.set_title('Mês por chuva média e ocorrencia de casos de dengue', fontdict = {'size' : 14})
ax.legend()
plt.show()
num_true = len(df2.loc[df2['ocorrencia-casos'] == 1])
num_false = len(df2.loc[df2['ocorrencia-casos'] == 0])
print("Número de Casos Verdadeiros: {0} ({1:2.2f}%)".format(num_true, (num_true/ (num_true + num_false)) * 100))
print("Número de Casos Falsos     : {0} ({1:2.2f}%)".format(num_false, (num_false/ (num_true + num_false)) * 100))
from sklearn.model_selection import train_test_split
atributos = ['chuva', 'temperatura-media', 'temperatura-mininima', 'temperatura-maxima']
atrib_prev = ['ocorrencia-casos']
split_test_size = 0.30
X = df2[atributos].values
Y = df2[atrib_prev].values
X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size = split_test_size, random_state = 42)
print("{0:0.2f}% nos dados de treino".format((len(X_treino)/len(df.index)) * 100))
print("{0:0.2f}% nos dados de teste".format((len(X_teste)/len(df.index)) * 100))
print("Original True : {0} ({1:0.2f}%)".format(len(df2.loc[df2['ocorrencia-casos'] == 1]), 
                                               (len(df2.loc[df2['ocorrencia-casos'] ==1])/len(df.index) * 100)))

print("Original False : {0} ({1:0.2f}%)".format(len(df2.loc[df2['ocorrencia-casos'] == 0]), 
                                               (len(df2.loc[df2['ocorrencia-casos'] == 0])/len(df.index) * 100)))
print("")
print("Training True : {0} ({1:0.2f}%)".format(len(Y_treino[Y_treino[:] == 1]), 
                                               (len(Y_treino[Y_treino[:] == 1])/len(Y_treino) * 100)))

print("Training False : {0} ({1:0.2f}%)".format(len(Y_treino[Y_treino[:] == 0]), 
                                               (len(Y_treino[Y_treino[:] == 0])/len(Y_treino) * 100)))
print("")
print("Test True : {0} ({1:0.2f}%)".format(len(Y_teste[Y_teste[:] == 1]), 
                                               (len(Y_teste[Y_teste[:] == 1])/len(Y_teste) * 100)))

print("Test False : {0} ({1:0.2f}%)".format(len(Y_teste[Y_teste[:] == 0]), 
                                               (len(Y_teste[Y_teste[:] == 0])/len(Y_teste) * 100)))
from sklearn.preprocessing import Imputer
preenche_0 = Imputer(missing_values = 0, strategy = "mean", axis = 0)

# Substituindo os valores iguais a zero, pela média dos dados
X_treino = preenche_0.fit_transform(X_treino)
X_teste = preenche_0.fit_transform(X_teste)
from sklearn.naive_bayes import GaussianNB
modelo_v1 = GaussianNB()
modelo_v1.fit(X_treino, Y_treino.ravel())
from sklearn import metrics
nb_predict_train = modelo_v1.predict(X_treino)
print("Exatidão (Accuracy): {0:.4f}".format(metrics.accuracy_score(Y_treino, nb_predict_train)))
print()
nb_predict_test = modelo_v1.predict(X_teste)
print("Exatidão (Accuracy): {0:.4f}".format(metrics.accuracy_score(Y_teste, nb_predict_test)))