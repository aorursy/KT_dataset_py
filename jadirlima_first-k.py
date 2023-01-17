# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error

%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
#sl.__version__


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
dados=pd.read_csv('/kaggle/input/car-price-prediction/CarPrice_Assignment.csv')
dados.head(3)
#id do carro nao tem relevancia
dados = dados.drop('car_ID',axis=1)
# converter risco_seguro pois é uma variável categoria
dados["symboling"] = dados["symboling"].astype(str)
# separar Fabricante do nome do carro
dados["fabricante"] = dados['CarName'].str.split(' ', expand=True)[0]
dados.groupby('fabricante').size()
# corrigindo erros nos nomes dos fabricantes
dados["fabricante"] = dados["fabricante"].replace({"maxda":"mazda",
                               "Nissan":"nissan",
                               "porcshce":"porsche",
                               "toyouta":"toyota",
                               "vokswagen":"volkswagen",
                               "vw":"volkswagen"})
# Valores unicos de Fabricantes
dados["fabricante"].unique()
# o nome do carro nao tem relevancia
dados.drop(columns="CarName", inplace=True)
# Dividindo carros em categoria popular e superior
dados["categoria_preco"] = dados["price"].apply(lambda x: "popular" if x <= 18500 else "superior")
colunas_numerica = list(dados.select_dtypes(exclude="object"))
colunas_categoricas = list(dados.select_dtypes(include="object"))
# visualizar distribuição por Fabricante
plt.figure(figsize=(15,6))
dados["fabricante"].value_counts().sort_values(ascending=False).plot.bar()
plt.xticks(rotation=90)
plt.xlabel("Fabricante", fontweight="bold")
plt.ylabel("Qtde", fontweight="bold")
plt.title("Quantidade de carros por Fabricante", fontweight="bold")
plt.show()
# Observando variaveis categoricas
plt.figure(figsize=(15,20))
for i,col in enumerate(colunas_categoricas[:-2], start=1):
    plt.subplot(5,2,i)
    sns.countplot(dados[col])
    plt.xlabel(col, fontweight="bold")
plt.show()
# mapa de calor para visualizar a correlação de pearson entre preço e outras variáveis numéricas
plt.figure(figsize=(12,8))
sns.heatmap(dados.corr(), annot=True, cmap="RdYlGn", square=True, mask=np.triu(dados.corr(), k=1))
plt.show()
variables=dados.drop(columns=["price"])
# converter variáveis categoricas em numéricas
le = LabelEncoder()
df_encoded=dados
df_encoded[colunas_categoricas] = df_encoded[colunas_categoricas].apply(lambda col: le.fit_transform(col))
df_encoded.head()
# Coletando x e y
X = df_encoded.drop(columns=["price"])
y = df_encoded['price']
# Criando um Correlation Plot
def visualize_correlation_matrix(data, hurdle = 0.0):
    R = np.corrcoef(data, rowvar = 0)
    R[np.where(np.abs(R) < hurdle)] = 0.0
    heatmap = plt.pcolor(R, cmap = mpl.cm.coolwarm, alpha = 0.8)
    heatmap.axes.set_frame_on(False)
    heatmap.axes.set_yticks(np.arange(R.shape[0]) + 0.5, minor = False)
    heatmap.axes.set_xticks(np.arange(R.shape[1]) + 0.5, minor = False)
    heatmap.axes.set_xticklabels(variables, minor = False)
    plt.xticks(rotation=90)
    heatmap.axes.set_yticklabels(variables, minor = False)
    plt.tick_params(axis = 'both', which = 'both', bottom = 'off', top = 'off', left = 'off', right = 'off') 
    plt.colorbar()
    plt.show()
# Visualizando o Plot
plt.figure(figsize=(12,8))
visualize_correlation_matrix(X, hurdle = 0.5)
# Gerando os dados
observations = len(df_encoded)
variables = df_encoded.columns
# Aplicando Padronização
standardization = StandardScaler()
Xst = standardization.fit_transform(X)
original_means = standardization.mean_
originanal_stds = standardization.scale_
# Gerando X e Y
Xst = np.column_stack((Xst,np.ones(observations)))
y  = df_encoded['price'].values
from sklearn.linear_model import LinearRegression
modelo = LinearRegression(normalize = False, fit_intercept = True)
def r2_est(X,y):
    return r2_score(y, modelo.fit(X,y).predict(X))
# Gera o impacto de cada atributo no R2
r2_impact = list()
for j in range(X.shape[1]):
    selection = [i for i in range(X.shape[1]) if i!=j]
    r2_impact.append(((r2_est(X,y) - r2_est(X.values[:,selection],y)), X.columns[j]))
    
for imp, varname in sorted(r2_impact, reverse = True):
    print ('%6.3f %s' %  (imp, varname))
XX=X[['fabricante','enginelocation','enginesize','stroke','carbody',
      'horsepower','drivewheel','carlength','highwaympg','citympg']]
X_train, X_test, Y_train, Y_test  = train_test_split(XX, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestRegressor
# Criando o modelo
modelo = RandomForestRegressor()

# Treinando o modelo
modelo.fit(X_train, Y_train)

# Fazendo previsões
Y_pred = modelo.predict(X_test)

# Resultado
Y_pred = modelo.predict(X_test)
print("R-squared:", r2_score(Y_pred, Y_test))
