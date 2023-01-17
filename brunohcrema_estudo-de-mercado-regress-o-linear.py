import matplotlib.pyplot as plt

%matplotlib inline

import pandas as pd

import numpy as np
dados = pd.read_csv('../input/analise-casas/HousePrices_HalfMil.csv', sep = ';')
dados.head(10)
dados.shape
dados.describe()
dados.corr().round(4)
import seaborn as sns
ax = sns.boxplot(data=dados["precos"], orient='v', width=0.2)

ax.figure.set_size_inches(12,6)

ax.set_title("Valor do Imóvel", fontsize = 22)

ax.set_ylabel("Valor em Reais", fontsize = 22)

ax
ax = sns.boxplot(data=dados, y="precos", x="garagem", orient='v', width=0.2)

ax.figure.set_size_inches(12,6)

ax.set_title("Valor do Imóvel X Garagem", fontsize = 22)

ax.set_ylabel("Quantidade de Garagens", fontsize = 22)

ax
ax = sns.boxplot(y="precos", x="banheiros", data=dados, orient='v', width=0.2)

ax.figure.set_size_inches(12,6)

ax.set_title("Valor do Imóvel X Banheiro", fontsize = 22)

ax.set_ylabel("Quantidade de Banheiros", fontsize = 22)

ax

ax = sns.boxplot(y="precos", x="lareira", data=dados, orient='v', width=0.2)

ax.figure.set_size_inches(12,6)

ax.set_title("Valor do Imóvel X Lareira", fontsize = 22)

ax.set_ylabel("Quantidade de Lareiras", fontsize = 22)

ax
ax = sns.boxplot(y="precos", x="marmore", data=dados, orient='v', width=0.2, hue = "marmore")

ax.figure.set_size_inches(12,6)

ax.set_title("Valor do Imóvel X Mármore", fontsize = 22)

ax.set_ylabel("Com ou sem Mármore", fontsize = 22)

ax
ax = sns.boxplot(y="precos", x="andares", data=dados, orient='v', width=0.2)

ax.figure.set_size_inches(12,6)

ax.set_title("Valor do Imóvel X Andares", fontsize = 22)

ax.set_ylabel("Número de andares", fontsize = 22)

ax
x = pd.DataFrame(dados["precos"])

ax = sns.distplot(x)

ax.figure.set_size_inches(12,6)

ax.set_title("Distribuição de Frequências", fontsize = 22)

ax.set_ylabel('Frequências', fontsize=16)

ax
ax = sns.pairplot(dados)
ax = sns.pairplot(dados, y_vars="precos", x_vars=['area', 'garagem', 'banheiros', 'lareira', 'marmore', 'andares'], kind = "reg")

ax.fig.suptitle("Dispersão entre as Variáveis", fontsize = 20, y=1.1)

ax
dados.corr().round(4)
from sklearn.model_selection import train_test_split
y = dados['precos']
X = dados[['area', 'garagem', 'banheiros', 'lareira', 'marmore', 'andares']]
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=2811)
from sklearn.linear_model import LinearRegression

from sklearn import metrics
modelo = LinearRegression()
modelo.fit(X_train, y_train)
print("R² = {}".format(modelo.score(X_train, y_train).round(2)))
y_previsto = modelo.predict(X_test)
print('R² = %s' % metrics.r2_score(y_test, y_previsto).round(2))
area = 130

garagem = 2

banheiros = 3

lareira= 1

marmore= 1

andares= 1

entrada=[[area, garagem, banheiros, lareira, marmore, andares]]



print('{0:.2f} Reais'.format(modelo.predict(entrada)[0]))
EQM = metrics.mean_squared_error(y_test, y_previsto).round(2)

REQM = np.sqrt(metrics.mean_squared_error(y_test, y_previsto)).round(2)

R2 = metrics.r2_score(y_test, y_previsto).round(2)



pd.DataFrame([EQM, REQM, R2], ['EQM', 'REQM', 'R²'], columns=['Métricas'])
import pickle
output = open('modelo_preço', 'wb')

pickle.dump(modelo, output)

output.close()