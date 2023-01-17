import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score, accuracy_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

import math

import numpy as np

from statsmodels.stats.weightstats import DescrStatsW

from scipy.stats import t as t_student
tabela = pd.read_csv('../input/campeonato-braileiro-20092018/tabelas/Tabela_Clubes.csv')
tabela.head()
tabela.drop(['Unnamed: 13', 'Unnamed: 14', 'Unnamed: 15', 'Unnamed: 16'], axis=1, inplace=True)
tabela.shape
tabela.dtypes
tabela.describe().transpose()
tabela.isnull().sum()
# Algumas colunas estavam com o nome errado, por exemplo a coluna de empates estava como derrota e vice versa. Por isso na linha abaixo estou renomeando as colunas com o nome correto.

tabela.columns = ['Ano', 'Posição', 'Clubes', 'Vitorias', 'Empates', 'Derrotas','GolsF/S', 'Saldo', 'Qtd_Jogadores','Idade_Media', 'Estrangeiros', 'Valor_total', 'Media_Valor']
# O ano dos campeonatos esta errado, por isso vou adicionar 1 aos anos.

tabela['Ano'] = tabela['Ano']+1
tabela.Clubes.value_counts()
#transformando a variável Idade_Media em um float

tabela.Idade_Media = tabela.Idade_Media.apply(lambda x: x.replace(',', '.'))

tabela.Idade_Media = tabela.Idade_Media.astype(float)
# Agora estou criando uma coluna com os pontos dos times

tabela["Pontos"] = (tabela['Vitorias']*3)+(tabela["Empates"])
tabela.head()
#Um filtro com apenas os campeões.

campeoes = tabela.query('Posição == 1').sort_values('Pontos', ascending=False)
#Para efeito gráfico, vou colocar o ano junto ao nome do Clube.

new_campeao = campeoes["Ano"].copy()

new_campeao = new_campeao.astype(str)

campeoes["Clubes"]= campeoes["Clubes"].str.cat(new_campeao, sep =" - ") 
# Estatística de Pontos dos campeões

campeoes['Pontos'].describe()
plt.figure(figsize=(18,8))

sns.barplot(x="Clubes", y="Pontos", data=campeoes)
campeoes
sns.distplot(campeoes['Pontos'], bins=5)
campeoes.shape
print (campeoes["Pontos"].mean())

print (campeoes["Pontos"].std())
sign = 0.05

conf = 1 - sign

n =10

liber = n - 1
t_alpha = t_student.ppf(conf, liber)

t_alpha *(-1)
t = (75.5-67) / (4.927248499698161 / np.sqrt(10))

t
p_valor = t_student.sf(5.4552475104671165, df = 9)

p_valor
test=DescrStatsW(campeoes["Pontos"])
test.ttest_mean(value=67, alternative="smaller")

p_valor >=sign
desc = DescrStatsW(campeoes["Pontos"])

desc.tconfint_mean()
# Estatística de Vitórias dos campeões

campeoes['Vitorias'].describe()
plt.figure(figsize=(18,6))

sns.barplot(x="Clubes", y="Vitorias", data=campeoes)
# Estatística de Derrotas dos campeões

campeoes['Derrotas'].describe()
plt.figure(figsize=(18,8))

sns.barplot(x="Clubes", y="Derrotas", data=campeoes)
vice_campeoes = tabela.query('Posição == 2')
vice_campeoes = vice_campeoes.sort_values('Pontos', ascending=False)
#Para efeito gráfico, vou colocar o ano junto ao nome do Clube.

new = vice_campeoes["Ano"].copy()

new = new.astype(str)

vice_campeoes["Clubes"]= vice_campeoes["Clubes"].str.cat(new, sep =" - ") 
# Estatística de pontos dos Vice-campeões

vice_campeoes['Pontos'].describe()
plt.figure(figsize=(18,8))

sns.barplot(x="Clubes", y="Pontos", data=vice_campeoes)
# Estatística de vitórias dos vice_campeões

vice_campeoes['Vitorias'].describe()
# Estatística de Derrotas dos Vice_campeões

vice_campeoes['Derrotas'].describe()
plt.figure(figsize=(18,8))

sns.barplot(x="Clubes", y="Derrotas", data=vice_campeoes)
libertadores = tabela.query('Posição == 4').sort_values('Pontos', ascending=False)
#Para efeito gráfico, vou colocar o ano junto ao nome do Clube.

new_liberta = libertadores["Ano"].copy()

new_liberta = new_liberta.astype(str)

libertadores["Clubes"]= libertadores["Clubes"].str.cat(new_liberta, sep =" - ") 
# Estatística de pontos Libertadores

libertadores['Pontos'].describe()
plt.figure(figsize=(16,6))

sns.barplot(x="Clubes", y="Pontos", data=libertadores)
pre_libertadores = tabela.query('Posição in [5,6]').sort_values('Pontos', ascending=False)
pre_libertadores['Pontos'].describe()
X = tabela.drop(['Ano', 'Posição', 'Clubes', 'Vitorias', 'Derrotas', 'Empates', 'GolsF/S', 'Saldo', 'Pontos'], axis=1)

y= tabela['Posição']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
modelo_regressao = LinearRegression()
modelo_arvore= DecisionTreeClassifier()
modelo_svc = SVC()
modelo_regressao.fit(X_train,y_train)
modelo_arvore.fit(X_train,y_train)
modelo_svc.fit(X_train,y_train)
predict_regressao = modelo_regressao.predict(X_test)
predict_arvore = modelo_arvore.predict(X_test)
predict_svc= modelo_svc.predict(X_test)
regressao = r2_score(y_test,predict_regressao)*100

regressao = round(regressao, 2)
decision_tree = accuracy_score(y_test,predict_arvore)*100

decision_tree = round(decision_tree, 2)
svc = accuracy_score(y_test,predict_svc)*100

svc = round(svc, 2)
print('Usando regressão linear as minhas previsões acertaram', regressao, '%')

print('Usando arvore de decisão as minhas previsões acertaram', decision_tree, '%')

print('Usando SVC as minhas previsões acertaram', svc, '%')