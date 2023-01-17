# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
dados = pd.read_csv('/kaggle/input/covid19/brazil_covid19.csv')
dados.head()
dados.shape
dados.dtypes
dados.isnull().sum()
dados.describe()
a = dados.date.value_counts().sort_index()

print('Primeira data da base:',a.index[0])

print('Última data da base:',a.index[-1])
dados.tail()
ultimos_dados = dados[dados.date == a.index[-1]]

ultimos_dados = ultimos_dados.groupby('state').sum()

ultimos_dados = ultimos_dados.drop(['suspects','refuses'], axis=1)

ultimos_dados.head()
# Criando a ordenando o plot

bar =ultimos_dados.sort_values(by='cases', 

                          ascending=True).plot(kind='bar',

                          figsize=(20,10),

                          color = ['yellow','red'],

                          width = 1,

                          rot=1)



sns.set(style='whitegrid')



for i in bar.patches:

    bar.annotate('{:.0f}'.format(i.get_height()), (i.get_x()+0.4, 

                                                   i.get_height()),ha='center', 

                                                   va='bottom',color= 'black')



# Definindo a legenda e os paramentros

plt.title('Panorama gráfico de casos por estados', size=20)

plt.ylabel('Quantidade', size=15)

plt.yticks(size=10)

plt.xlabel('Estados', size=15)

plt.xticks(size=10, rotation=45)

plt.legend(bbox_to_anchor=(0.95,0.95), # coordenadas

                          frameon = True,

                          fontsize = 10,

                          ncol = 2,

                          fancybox = True,

                          framealpha = 0.95,

                          shadow = True,

                          borderpad = 1);
agregado_brasil = dados.groupby('date').sum()

agregado_brasil.head()
agregado_brasil.shape
agregado_brasil = agregado_brasil.reset_index(drop=False)

agregado_brasil.head()
agregado_brasil.dtypes
agregado_brasil.loc[agregado_brasil['cases'] == 1]
agregado_brasil_inicio_dos_casos = agregado_brasil.loc[agregado_brasil['date'] >= '2020-02-26']

print(agregado_brasil_inicio_dos_casos.shape)

agregado_brasil_inicio_dos_casos.head()
plt.figure(figsize = (10,10));

sns.scatterplot(x='date', s = 100, y='cases', 

                data=agregado_brasil_inicio_dos_casos, color = 'blue')

plt.xticks(rotation=45);
agregado_brasil_inicio_dos_casos['dias'] = range(1, len(agregado_brasil_inicio_dos_casos)+1, 1)

agregado_brasil_inicio_dos_casos.head()
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split 

from sklearn.preprocessing import PolynomialFeatures
agregado_brasil_inicio_dos_casos.columns
X = agregado_brasil_inicio_dos_casos[['dias']]

y = agregado_brasil_inicio_dos_casos[['deaths']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Fitando uma regressão polinomial

poly_reg = PolynomialFeatures(degree=2, include_bias = False)

X_poly = poly_reg.fit_transform(X_train)

pol_reg = LinearRegression()

pol_reg.fit(X_poly, y_train)
y_predict = pol_reg.predict(poly_reg.fit_transform(X_train))
# Visualizing the Polymonial Regression results

def viz_polymonial():

    plt.scatter(X, y, color='red')

    plt.plot(X_test, pol_reg.predict(poly_reg.fit_transform(X_test)), color='blue')

    plt.title('Regression')

    plt.xlabel('Dias')

    plt.ylabel('Casos')

    plt.show()

    return

viz_polymonial()
plt.plot(X_train,y_predict, c= 'r')

plt.scatter(X_train, y_train, s = 32)



plt.plot(X_test, pol_reg.predict(poly_reg.fit_transform(X_test)), c = 'y')

plt.scatter(X_test, y_test, s=30, c='g')
plt.plot(X_train,y_predict, c= 'r')

plt.scatter(X_train, y_train, s = 32)
plt.plot(X_test, pol_reg.predict(poly_reg.fit_transform(X_test)), c = 'y')

plt.scatter(X_test, y_test, s=30, c='g')
from sklearn.metrics import mean_squared_error, r2_score

print("Erro médio quadrático: ",mean_squared_error(y_test,pol_reg.predict(poly_reg.fit_transform(X_test))))

print("R^2: ", r2_score(y_test,pol_reg.predict(poly_reg.fit_transform(X_test))))
from sklearn.tree import DecisionTreeRegressor 

from sklearn.metrics import accuracy_score

from sklearn.metrics import mean_squared_error
tree = DecisionTreeRegressor(random_state=42)
tree.fit(X_train, y_train)
y_tree_treino = tree.predict(X_train)

y_tree_teste = tree.predict(X_test)
plt.figure()

plt.scatter(X, y, s=20, edgecolor="black",

            c="darkorange", label="Dados")



plt.plot(X_train, y_tree_treino, color="cornflowerblue",

         label="Treino", linewidth=2)



plt.plot(X_test, y_tree_teste, color="yellowgreen", label="Teste", linewidth=2)

plt.xlabel("Dias")

plt.ylabel("target")

plt.title("Fit da árvore de decisão")

plt.legend()
acc_tree_treino = round(accuracy_score(y_train, y_tree_treino)*100, 2)

print('Acurácia de: ', acc_tree_treino)
print("Erro médio quadrático: ", np.sqrt(mean_squared_error(y_train, y_tree_treino)))

print("R^2: ", r2_score(y_train,pol_reg.predict(poly_reg.fit_transform(X_train))))
acc_tree_teste = round(accuracy_score(y_test, y_tree_teste)*100, 2)

print('Acurácia de: ', acc_tree_teste)
print("Erro médio quadrático: ", np.sqrt(mean_squared_error(y_test, y_tree_teste)))

print("R^2: ", r2_score(y_test,pol_reg.predict(poly_reg.fit_transform(X_test))))
tree.feature_importances_
!pip install pydotplus
from sklearn.externals.six import StringIO

from IPython.display import Image

from sklearn.tree import export_graphviz

import pydotplus
plot = X

tar = agregado_brasil_inicio_dos_casos[['deaths']]



dot_data = StringIO()

export_graphviz(tree, out_file=dot_data,

                filled=True, rounded=True,

                special_characters=True,

                feature_names=X.columns,

                class_names=tar)



graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

Image(graph.create_png())
plt.figure(figsize = (15,10));

sns.set(style='whitegrid')

sns.distplot(dados['cases'], rug=True, hist=True, bins=5);
plt.figure(figsize = (15,10));

sns.set(style='whitegrid');

sns.jointplot(x='cases', y='deaths', data=dados);
dados.head()
dados.drop(['date', 'hour'], axis=1).groupby('state').max().sort_values(by ='cases', 

                                                                        ascending=False).style.background_gradient(cmap='Reds')
dados[dados.date == '2020-03-30'].cases.sum()
dados.columns
dados_sao_paulo = dados[dados.state == 'São Paulo']

dados_sao_paulo = dados_sao_paulo.drop(labels=['hour'], axis=1)

dados_sao_paulo.head()
dados_sao_paulo.shape
dados_sao_paulo.head()
plt.figure(figsize = (20,10));

sns.set(style='whitegrid')



sns.lineplot(x='date', y='deaths', data=dados_sao_paulo, color='r')



plt.legend(labels=['Casos fatais'])

plt.title('Quantidade de casos em São Paulo')

plt.xticks(rotation=45)

plt.xlabel('Data')

plt.ylabel('Casos')

plt.show()
plt.figure(figsize = (20,10));

sns.set(style='whitegrid')

sns.lineplot(x='date', y='deaths', data=dados_sao_paulo, color = 'red')

sns.lineplot(x='date', y='cases', data=dados_sao_paulo, color = 'orange')

plt.legend(labels=['Casos fatais', 'Infectados'])

plt.title('Quantidade de casos fatais e infectados ao longo do tempo')

plt.xticks(rotation=45)

plt.xlabel('Data')

plt.ylabel('Casos')

plt.show()
c = dados_sao_paulo.corr()

c.style.background_gradient(cmap='coolwarm')
plt.figure(figsize = (10,10));

sns.set(style='whitegrid')

sns.scatterplot(x='cases', s = 100, y='deaths', data=dados_sao_paulo, color = 'green')

plt.title('Infectados x Vítimas fatais em São Paulo')

plt.xticks(rotation=45)

plt.xlabel('Casos fatais')

plt.ylabel('Infectados')

plt.show()