import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
Dados = pd.read_csv('../input/amostra-dados-enem-2019/MICRODADOS_ENEM_2019_SAMPLE_43278.csv')

Dados.head()
Dados.shape
Dados["SG_UF_RESIDENCIA"].value_counts()
Dados["NU_IDADE"].value_counts()
Dados["NU_IDADE"].hist(bins = 20, figsize = (8,6))

plt.title('Idades que mais aparecem entre os candidatos')

plt.ylabel('Quantidade')

plt.xlabel('Idades')

Proporcao = Dados['NU_IDADE'].value_counts()

Lista = []

for i in Proporcao:

  a = (i * 100) / 127379

  b = round(a,2)

  Lista.append(b)

Porcentagem = pd.DataFrame(Proporcao)

Porcentagem['Porcentagem'] = Lista

Porcentagem  
Dados.query("NU_IDADE==13 or NU_IDADE==14")
total = len(Dados.query("NU_IDADE<18"))

Dados.query("NU_IDADE<18").SG_UF_RESIDENCIA.value_counts()/total*100
Dados.query("IN_TREINEIRO == 1")["NU_IDADE"].value_counts().sort_index()
Dados.query("NU_IDADE==13 or NU_IDADE==14")
N = pd.DataFrame(Dados.query('IN_TREINEIRO == 1')['NU_IDADE'])

S = pd.DataFrame(Dados.query('IN_TREINEIRO == 0')['NU_IDADE'])

N.hist(bins = 20)

plt.title('Treineiros')

plt.xlabel('Idade')

plt.ylabel('Quantidade por idade')



S.hist(bins = 20)

plt.title('Não treineiros')

plt.xlabel('Idade')

plt.ylabel('Quantidade por idade')
Dados['Q025'].value_counts()

plt.figure(figsize=(10,6))

Dados['Q025'].value_counts().plot.pie(autopct = '%1.1f%%')

plt.title('Alunos com acesso a internet')

medias = [Dados['NU_NOTA_CH'].mean(),Dados['NU_NOTA_CN'].mean(),Dados['NU_NOTA_MT'].mean(),Dados['NU_NOTA_LC'].mean(),Dados['NU_NOTA_REDACAO'].mean()]

media = pd.DataFrame(medias, index = 'Ciencias_humanas Ciencias_natureza Matematica Linguagens_codigo Redação'.split())

media.round(0)

Dados['TP_LINGUA'].hist()
ingles = Dados.query('TP_LINGUA == 0')

espanhol = Dados.query('TP_LINGUA == 1')
ingles['NU_NOTA_LC'].hist(bins=50, figsize=(9,6))

espanhol['NU_NOTA_LC'].hist(bins=50, alpha = 0.4,color = 'green', figsize=(9,6))

plt.legend(['Prova ingles', 'Prova espanhol'])

plt.title('Desempenho provas linguagens(ingles e espanhol)')

provas = ['NU_NOTA_CN','NU_NOTA_CH','NU_NOTA_LC','NU_NOTA_MT', 'NU_NOTA_REDACAO']

Dados["NU_NOTA_TOTAL"] = Dados[provas].sum(axis=1)

Dados[provas].describe()

dados_diferente_zero = Dados[Dados['NU_NOTA_TOTAL'] != 0]

dadoszero = dados_diferente_zero.query("NU_NOTA_LC != 0 & NU_NOTA_CH !=0 & NU_NOTA_CN != 0 & NU_NOTA_MT !=0")

dadoszero.dropna()

plt.figure(figsize=(10,6))

sns.boxplot(x = 'TP_COR_RACA', y = 'NU_NOTA_TOTAL', data = dadoszero)

d = Dados[Dados['NO_MUNICIPIO_RESIDENCIA'] == 'Bagé']

d
mediasb = [d['NU_NOTA_CH'].mean(),d['NU_NOTA_CN'].mean(),d['NU_NOTA_MT'].mean(),d['NU_NOTA_LC'].mean(),d['NU_NOTA_REDACAO'].mean()]

mediab = pd.DataFrame(mediasb, index = 'Ciencias_humanas Ciencias_natureza Matematica Linguagens_codigo Redação'.split())

mediab.round(0)
Menor18 = Dados[Dados['NU_IDADE'] < 18]

Menores = pd.DataFrame(Menor18[['NU_IDADE', 'SG_UF_RESIDENCIA']])

plt.figure(figsize=(12,6))

sns.barplot(x = 'NU_IDADE', y = 'SG_UF_RESIDENCIA', data = Menores)
renda_ordenada = Dados["Q006"].unique()

renda_ordenada.sort()
plt.figure(figsize=(10,6))

sns.boxplot(x="Q006", y = "NU_NOTA_TOTAL", data = dadoszero,order = renda_ordenada)
def boxplot(x,y ,data):

    sns.boxplot(x = x, y = y, data = data)
plt.figure(figsize=(10,6))

boxplot("Q006","NU_NOTA_TOTAL",dadoszero)
presenca = ['TP_PRESENCA_CN', 'TP_PRESENCA_CH', 'TP_PRESENCA_LC','TP_PRESENCA_MT']
zero = Dados[Dados['NU_NOTA_TOTAL'] == 0]
dados_status_zero = {

    i: zero[i].value_counts() for i in presenca

}

dados_status_zero = pd.DataFrame.from_dict(dados_status_zero)
dados_status_zero
plt.figure(figsize=(12,8))

axes = sns.boxplot(x = 'Q006', y = 'NU_NOTA_TOTAL',hue = 'TP_ESCOLA',palette = 'rainbow', data = dadoszero, order = renda_ordenada)

axes.legend(loc = 1)

plt.title('Grafico das notas totais pela renda, com matiz no tipo de escola do ensino medio')





dados_status_zero.index.names = ['0: faltou  1: presente 2: eliminado' ]
dados_status_zero
eliminados = Dados.query("TP_PRESENCA_CN == 2 or TP_PRESENCA_CH == 2 or TP_PRESENCA_LC == 2 or TP_PRESENCA_MT == 2")

eliminados[provas]
var = Dados[['Q006', 'IN_TREINEIRO']]

rt = pd.DataFrame(var)

rt
dados_diferente_zero
dados_geral_treineiros = dados_diferente_zero['Q006'].value_counts()

dados_treineiros = dados_diferente_zero.query('IN_TREINEIRO == 1')['Q006'].value_counts()/dados_geral_treineiros

dados_nao_treineiros = dados_diferente_zero.query('IN_TREINEIRO == 0')['Q006'].value_counts()/dados_geral_treineiros



fig, ax = plt.subplots(figsize=(10,8))



ax.bar(renda_ordenada, dados_treineiros, label='Treineiros')

ax.bar(renda_ordenada, dados_nao_treineiros, bottom=dados_treineiros, label='Não Treineiros')



ax.set_ylabel('Quantidade')

ax.set_title('Quantidade de treineiros e não treineiros por classe social')

ax.legend()



plt.show()
plt.figure(figsize=(12,8))

ax = sns.boxplot(x="Q025", y = "NU_NOTA_TOTAL",hue = 'IN_TREINEIRO', data = dados_diferente_zero)

ax.set_xticklabels(['Não', 'Sim'])

plt.title("Boxplot das notas de total pelo acesso à internet")
estudo_ordenado = Dados["Q001"].unique()

estudo_ordenado.sort()
axes = sns.boxplot(x = 'Q001', y = 'NU_NOTA_TOTAL', data = dados_diferente_zero, order = estudo_ordenado)

plt.title("Notas dos alunos baseado na escolaridade do pai")
axes2 = sns.boxplot(x = 'Q002', y = 'NU_NOTA_TOTAL', data = dados_diferente_zero, order = estudo_ordenado)

plt.title("Notas dos alunos baseado na escolaridade do mãe")
internet_ordenado = dados_diferente_zero["Q003"].unique()

internet_ordenado.sort()

internet_ordenado2 = dados_diferente_zero["Q004"].unique()

internet_ordenado2.sort()

axes3 = sns.boxplot(x = 'Q003', y = 'NU_NOTA_TOTAL', data = dados_diferente_zero, order = internet_ordenado)

plt.title("Notas dos alunos baseado na profissão do pai")

import math

import matplotlib.gridspec as gridspec
axes4 = sns.boxplot(x = 'Q004', y = 'NU_NOTA_TOTAL', data = dados_diferente_zero, order = internet_ordenado2)

plt.title("Notas dos alunos baseado na profissão da mãe")





correlacao = dadoszero[provas].corr()

correlacao



mask = np.triu(np.ones_like(correlacao, dtype=bool))



cmap = sns.diverging_palette(230, 20, as_cmap=True)



sns.heatmap(data=correlacao, mask=mask, cmap=cmap, center= 0.45, vmax=0.8, square=True, linewidths=.1, cbar_kws={"shrink": 0.4}, annot=True)
dadosRS = dadoszero.query("SG_UF_RESIDENCIA == 'RS'")

dadosRS.dropna()



correlacao = dadosRS[provas].corr()

correlacao



mask = np.triu(np.ones_like(correlacao, dtype=bool))



cmap = sns.diverging_palette(230, 20, as_cmap=True)



sns.heatmap(data=correlacao, mask=mask, cmap=cmap, center= 0.45, vmax=0.8, square=True, linewidths=.1, cbar_kws={"shrink": 0.4}, annot=True)
sns.scatterplot(data = dadoszero, x="NU_NOTA_LC", y="NU_NOTA_MT")

plt.xlim((-50, 1050))

plt.ylim((-50, 1050))

plt.title("Correlação entre Matemática e Linguages")
provas_entrada = ["NU_NOTA_CH","NU_NOTA_LC", "NU_NOTA_CN","NU_NOTA_REDACAO"]

prova_saida = "NU_NOTA_MT"

dados_diferente_zero = dados_diferente_zero[provas].dropna()

x = dados_diferente_zero[provas_entrada]

y = dados_diferente_zero[prova_saida]
from sklearn.model_selection import train_test_split
SEED = 1001



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30,random_state=SEED)
from sklearn.svm import LinearSVR



modelo = LinearSVR(random_state = SEED)

modelo.fit(x_train, y_train)
predicoes_matematica = modelo.predict(x_test)
y_test[:5]
plt.figure(figsize=(8, 6))

sns.scatterplot(x=predicoes_matematica, y=y_test)

plt.xlim((-50, 1050))

plt.ylim((-50, 1050))
plt.figure(figsize=(8, 6))

sns.scatterplot(x=y_test, y=y_test - predicoes_matematica)

resultados = pd.DataFrame()

resultados["Real"] = y_test

resultados["Previsao"] = predicoes_matematica

resultados["diferenca"] = resultados["Real"] - resultados["Previsao"]

resultados["quadrado_diferenca"] = (resultados["Real"] - resultados["Previsao"])**2
resultados
resultados["quadrado_diferenca"].mean()
resultados["quadrado_diferenca"].mean()**(1/2)
from sklearn.dummy import DummyRegressor



modelo_dummy = DummyRegressor()

modelo_dummy.fit(x_train, y_train)

dummy_predicoes = modelo_dummy.predict(x_test)
from sklearn.metrics import *



mean_squared_error(y_test, dummy_predicoes)
mean_squared_error(y_test, predicoes_matematica)
from sklearn.linear_model import LinearRegression

modelo_lr = LinearRegression()

modelo_lr.fit(x_train, y_train)

predicao_lr = modelo_lr.predict(x_test)



resultados_lr = pd.DataFrame()

resultados_lr["Real"] = y_test

resultados_lr["Previsao"] = predicao_lr

resultados_lr["diferenca"] = resultados_lr["Real"] - resultados_lr["Previsao"]

resultados_lr["quadrado_diferenca"] = (resultados_lr["Real"] - resultados_lr["Previsao"])**2

resultados_lr
from sklearn.linear_model import SGDRegressor

from sklearn import preprocessing



scaler = preprocessing.StandardScaler().fit(x_train)

x_sgd_train = scaler.transform(x_train)

x_sgd_test = scaler.transform(x_test)



n_iter=100

modelo_sgd = SGDRegressor(max_iter=n_iter)

modelo_sgd.fit(x_sgd_train, y_train)

predicao_sgd = modelo_sgd.predict(x_sgd_test)



resultados_sgd = pd.DataFrame()

resultados_sgd["Real"] = y_test

resultados_sgd["Previsao"] = predicao_sgd

resultados_sgd["diferenca"] = resultados_sgd["Real"] - resultados_sgd["Previsao"]

resultados_sgd["quadrado_diferenca"] = (resultados_sgd["Real"] - resultados_sgd["Previsao"])**2

resultados_sgd
from sklearn.ensemble import RandomForestRegressor



regressor = RandomForestRegressor(n_estimators = 100, random_state = SEED) 

regressor.fit(x_train, y_train)

predicao_rf = regressor.predict(x_test) 



resultados_rf = pd.DataFrame()

resultados_rf["Real"] = y_test

resultados_rf["Previsao"] = predicao_rf

resultados_rf["diferenca"] = resultados_rf["Real"] - resultados_rf["Previsao"]

resultados_rf["quadrado_diferenca"] = (resultados_rf["Real"] - resultados_rf["Previsao"])**2

resultados_rf
sns.set()

fig, axes = plt.subplots(2,2, figsize=(20, 15)) 

axes[0,0].set_title("LinearSVR")

axes[0,1].set_title("LinearRegression")

axes[1,0].set_title("SGDRegression")

axes[1,1].set_title("RandomForest")



sns.scatterplot(x=y_test, y=predicoes_matematica, ax=axes[0,0])

sns.scatterplot(x=y_test, y=predicao_lr, ax=axes[0,1])

sns.scatterplot(x=y_test, y=predicao_sgd, ax=axes[1,0])

sns.scatterplot(x=y_test, y=predicao_rf, ax=axes[1,1])



plt.show()