import matplotlib.pyplot as plt

%matplotlib inline



import pandas as pd

import numpy as np
import warnings

warnings.filterwarnings('ignore')
dados = pd.read_csv("../input/consumo-de-cerveja/Consumo_cerveja.csv", sep = ";")
dados
dados.shape
dados.describe().round(2)
dados.corr().round(4)
fig, ax = plt.subplots(figsize = (30,6))



ax.set_title("Consumo de Cerveja", fontsize = 30)

ax.set_ylabel('Litros', fontsize = 22)

ax.set_xlabel("Dias", fontsize = 22)

ax = dados['consumo'].plot()

import seaborn as sns
dados.boxplot(["consumo"])



consumo = dados["consumo"]

ax = sns.boxplot(data=dados["consumo"], orient='v', width=0.2)

ax.figure.set_size_inches(12,6)

ax.set_title("Consumo de Cerveja", fontsize = 22)

ax.set_ylabel("Litros", fontsize = 22)
ax = sns.distplot(dados["consumo"])

ax.figure.set_size_inches(12,6)

ax.set_title("Distribuição de Frequências", fontsize = 22)

ax.set_ylabel("Consumo de Cerveja (Litros)", fontsize = 22)

ax
ax.figure.set_size_inches(12,6)

ax.set_title("Consumo de Cerveja", fontsize = 22)

ax.set_ylabel("Litros", fontsize = 22)

ax = sns.pairplot(dados)
ax = sns.pairplot(dados, y_vars="consumo", x_vars=['temp_min', 'temp_media', 'temp_max', 'chuva', 'fds'])

ax.fig.suptitle("Dispersão entre as Variáveis", fontsize = 20, y=1.1)

ax
ax = sns.pairplot(dados, y_vars="consumo", x_vars=['temp_min', 'temp_media', 'temp_max', 'chuva', 'fds'], kind = 'reg')

ax.fig.suptitle("Dispersão entre as Variáveis", fontsize = 20, y=1.1)

ax
ax = sns.jointplot(x="temp_max", y='consumo', data=dados, kind = 'reg')
ax = sns.lmplot(x='temp_max', y='consumo', data=dados)
ax = sns.lmplot(x='temp_max', y='consumo', data=dados, hue = 'fds', markers=['o','*'], legend = False)

ax.add_legend(title="Final de Semana")

for item, legenda in zip(ax._legend.texts, ['Não', 'Sim']): 

    item.set_text(legenda)

ax
ax = sns.lmplot(x='temp_max', y='consumo', data=dados, col = 'fds')
from sklearn.model_selection import train_test_split
y = dados['consumo']
X = dados[['temp_max','chuva','fds'] ]
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=2811)
X_train.shape
X_test.shape
X_train.shape[0]+X_test.shape[0]
X.shape[0]*0.3
X.shape[0]*0.7
from sklearn.linear_model import LinearRegression

from sklearn import metrics
modelo = LinearRegression()
modelo.fit(X_train, y_train)
print("R² = {}".format(modelo.score(X_train, y_train).round(2)))
y_previsto = modelo.predict(X_test)
print('R² = %s' % metrics.r2_score(y_test, y_previsto).round(2))
entrada = X_test[0:1]

entrada
modelo.predict(entrada)[0]
temp_max=14

chuva=10

fds=1

entrada=[[temp_max, chuva, fds]]



print('{0:.2f} litros'.format(modelo.predict(entrada)[0]))
modelo.intercept_
type(modelo.intercept_)
modelo.coef_
type(modelo.coef_)
X.columns
index=['Intercepto', 'Temperatura Máxima', 'Chuva (mm)', 'Final de Semana']
pd.DataFrame(data=np.append(modelo.intercept_, modelo.coef_), index=index, columns=['Parâmetros'])
y_previsto_train = modelo.predict(X_train)
ax = sns.scatterplot(x = y_previsto_train, y = y_train)

ax.figure.set_size_inches(12,6)

ax.set_title('Previsão X Real', fontsize = 18)

ax.set_xlabel('Consumo de Cerveja (litros) - Previsão', fontsize = 14)

ax.set_ylabel('Consumo de Cerveja (litros) - Real', fontsize = 14)

ax
residuo = y_train - y_previsto_train

residuo
ax = sns.scatterplot(x = y_previsto_train, y = residuo, s = 110)

ax.figure.set_size_inches(20,8)

ax.set_title('Residuo X Previsão', fontsize = 18)

ax.set_xlabel('Consumo de Cerveja (litros) - Previsão', fontsize = 14)

ax.set_ylabel('Residuo', fontsize = 14)

ax
ax = sns.scatterplot(x = y_previsto_train, y = residuo**2, s = 110)

ax.figure.set_size_inches(20,8)

ax.set_title('Residuo X Previsão', fontsize = 18)

ax.set_xlabel('Consumo de Cerveja (litros) - Previsão', fontsize = 14)

ax.set_ylabel('Residuo', fontsize = 14)

ax
ax = sns.distplot(residuo, bins=10)

ax.figure.set_size_inches(12,6)

ax.set_title('Distribuição de Frequência dos Residuos', fontsize = 18)

ax.set_xlabel('Lítros', fontsize = 18)

ax
X2 = dados[['temp_media', 'chuva', 'fds']]
X2_train, X2_test, y2_train, y2_test = train_test_split( X2, y, test_size=0.3, random_state=2811)
modelo_2 = LinearRegression()
modelo_2.fit(X2_train, y2_train)
print("R² = {} modelo antigo".format(modelo.score(X_train, y_train).round(2)))
print("R² = {}".format(modelo_2.score(X2_train, y2_train).round(2)))
y_previsto = modelo.predict(X_test)

y_previsto_2 = modelo_2.predict(X2_test)
print('R² = {}'.format(metrics.r2_score(y2_test, y_previsto_2).round(2)))
print('R² = {}'.format(metrics.r2_score(y_test, y_previsto).round(2)))
EQM_2 = metrics.mean_squared_error(y2_test, y_previsto_2).round(2)

REQM_2 = np.sqrt(metrics.mean_squared_error(y2_test, y_previsto_2)).round(2)

R2_2 = metrics.r2_score(y2_test, y_previsto_2).round(2)



pd.DataFrame([EQM_2, REQM_2, R2_2], ['EQM', 'REQM', 'R²'], columns = ['Metricas'])
EQM= metrics.mean_squared_error(y_test, y_previsto).round(2)

REQM = np.sqrt(metrics.mean_squared_error(y_test, y_previsto)).round(2)

R2 = metrics.r2_score(y_test, y_previsto).round(2)



pd.DataFrame([EQM, REQM, R2], ['EQM', 'REQM', 'R²'], columns = ['Metricas'])
X_test[0:1]
entrada = X_test[0:1]
modelo.predict(entrada)[0]
temp_max=14

chuva=10

fds=1

entrada=[[temp_max, chuva, fds]]



print('{0:.2f} litros'.format(modelo.predict(entrada)[0]))
import pickle
output = open('modelo_consumo_cerveja', 'wb')

pickle.dump(modelo, output)

output.close()