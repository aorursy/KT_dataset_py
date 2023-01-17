# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
np.random.seed(2)

#

pd.set_option('display.max_columns', None)

#pd.set_option('display.max_rows', 100)

import warnings

warnings.filterwarnings('ignore')

#

import matplotlib.pyplot as plt

%matplotlib inline
dados_treino = pd.read_csv("../input/dataset_treino.csv", parse_dates=['Date'])

dados_teste = pd.read_csv("../input/dataset_teste.csv", parse_dates=['Date'])

lojas = pd.read_csv("../input/lojas.csv")
lojas.isnull().sum()
lojas[lojas['CompetitionDistance'].isnull()]
media = lojas['CompetitionDistance'].mean()

lojas.loc[lojas['CompetitionDistance'].isnull(), 'CompetitionDistance'] = media
lojas.fillna(0, inplace=True)
dados_teste.isnull().sum()
dados_teste[dados_teste.Open.isnull()]
dados_teste.loc[(dados_teste.Store == 622) & (dados_teste.Promo == 1), 'Open'] = 1
dados_teste[dados_teste.Open.isnull()]
dados_teste.fillna(value=0, inplace=True)
dados_treino.isnull().sum()
df_treino = pd.merge(dados_treino, lojas, on='Store')

df_teste = pd.merge(dados_teste, lojas, on='Store')
df_teste.head().append(df_teste.tail())
df_treino.head().append(df_treino.tail())
df_treino.shape
df_teste.shape
df_treino = df_treino.sort_values(['Date'], ascending=False)

df_teste = df_teste.sort_values(['Date'], ascending=False)
treino_stats = df_treino.describe()

treino_stats = treino_stats.transpose()

treino_stats
teste_stats = df_teste.describe()

teste_stats = teste_stats.transpose()

teste_stats
df_treino.Sales.describe()
df_treino.Sales.hist()
df_treino.shape
treino = df_treino.loc[(df_treino.Sales > 0) & (df_treino.Sales <= 13000)]

treino.Sales.hist()
df_treino = treino.copy()

df_treino.shape
df_treino.CompetitionDistance.describe()
df_treino.CompetitionDistance.hist()
treino = df_treino.loc[(df_treino.CompetitionDistance >= 10000) & (df_treino.CompetitionDistance <= 35000)]

treino.CompetitionDistance.hist()

treino.shape
df_treino = treino.copy()

df_treino.shape
def processamento(dados, isTeste=False):

    #

    levels = {'0':0, 'a':1, 'b':2, 'c':3, 'd':4}

    dados.StoreType.replace(levels, inplace=True)

    dados.Assortment.replace(levels, inplace=True)

    dados.StateHoliday.replace(levels, inplace=True)

    #

    levels2 = {'0':0, 'Feb,May,Aug,Nov':1, 'Jan,Apr,Jul,Oct':2, 'Mar,Jun,Sept,Dec':3}

    dados.PromoInterval.replace(levels2, inplace=True)

    

    # Separar os dados da data

    dados['Ano'] = dados.Date.dt.year

    dados['Mes'] = dados.Date.dt.month

    dados['Dia'] = dados.Date.dt.day

    dados['SemanaDoAno'] = dados.Date.dt.weekofyear

    

    # Calcular tempo aberto da concorrência em meses

    dados['CompetitionOpen'] = 12 * (dados.Mes - dados.CompetitionOpenSinceYear) + (dados.Mes - dados.CompetitionOpenSinceMonth)

    dados['CompetitionOpen'] = dados['CompetitionOpen'].apply(lambda x: x if x > 0 else 0)

    

    # Vezes em que o Promo2 esteve aberta nos meses

    dados['PromoOpen'] = 12 * (dados.Ano - dados.Promo2SinceYear) + (dados.SemanaDoAno - dados.Promo2SinceYear) / 4.0

    dados['PromoOpen'] = dados['PromoOpen'].apply(lambda x: x if x > 0 else 0)

    

    # Mes para String

    mesString = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 

                 7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}

    dados['Mes_String'] = dados.Mes.map(mesString)

    

    def check(linha):

        if isinstance(linha['PromoInterval'], str) and linha['Mes_String'] in linha['PromoInterval']:

            return 1

        else:

            return 0

    

    dados['MesPromocao'] = dados.apply(lambda linha: check(linha), axis=1)

    

    

    dados.PromoInterval.replace({0:'0'}, inplace=True)

    

    # Remover o atributo Date que já foi tratada e separada

    # e o atributo StateHoliday que possui muitos valores 0

    # O dataset de teste não possui o atributo Customers

    features = ['CompetitionDistance', 'PromoOpen', 'StoreType', 'Assortment', 'Mes', 'Dia', 'Ano', 

                'SemanaDoAno', 'DayOfWeek', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 

                'CompetitionOpen', 'MesPromocao', 'StateHoliday', 'PromoInterval', 'Open', 'Promo', 

                'Promo2', 'SchoolHoliday']

    #

    if not isTeste:

        features.append('Sales')

    

    dados = dados[features]

    

    return dados
df_treino = processamento(df_treino)
df_treino.shape
df_teste = processamento(df_teste, isTeste=True)
df_teste.shape
X = df_treino.drop('Sales', axis=1).copy()

y = np.log1p(df_treino['Sales'])
X.shape
y.shape
X.head()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)
X_normalized[0]
X_normalized.shape
X_teste_normalized = scaler.fit_transform(df_teste)
X_teste_normalized[0]
X_teste_normalized.shape
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_normalized, y, test_size=10000, random_state=2)
x_train.shape
y_test.shape
from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.wrappers.scikit_learn import KerasRegressor

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from keras.optimizers import Adam

import tensorflow as tf

tf.set_random_seed(2)
def build_regressor():

    modelo = Sequential()

    modelo.add(Dense(units=512, activation='relu', kernel_initializer='normal', input_dim=x_train.shape[1]))

    modelo.add(Dropout(0.40))

    modelo.add(Dense(units=256, activation='relu', kernel_initializer='normal'))

    modelo.add(Dropout(0.20))

    modelo.add(Dense(units=128, activation='relu', kernel_initializer='normal'))

    modelo.add(Dropout(0.20))

    modelo.add(Dense(units=1, activation='linear'))

    optimizer = Adam(lr=1e-3, decay=1e-3 / 200, clipvalue = 0.5)

    modelo.compile(loss="mae", optimizer=optimizer, metrics=['mse'])

    return modelo
rlr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 10, verbose = 1)
modelo = KerasRegressor(build_fn=build_regressor, batch_size=768, epochs=500)
modelo.fit(x_train, y_train, validation_data=(x_test, y_test), callbacks = [rlr])
modelo.score(x_test, y_test)
y_pred = modelo.predict(x_test)
from sklearn.metrics import mean_squared_error, mean_absolute_error
mean_squared_error(y_test, y_pred)
mean_absolute_error(y_test, y_pred)
fig, ax = plt.subplots()

ax.scatter(y_test, y_pred)

ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)

ax.set_xlabel('Measured')

ax.set_ylabel('Predicted')

plt.show()
def correction():

    weights = np.arange(0.95, 1.05, 0.005)

    errors = []

    for w in weights:

        error = mean_absolute_error(y_test, y_pred*w)

        errors.append(error)

        

    # make line plot

    plt.plot(weights, errors)

    plt.xlabel('pesos')

    plt.ylabel('MAE')

    plt.title('Curva MAE')

    # print min error

    idx = errors.index(min(errors))

    print('O melhor peso é {}, MAE é {:.4f}'.format(weights[idx], min(errors)))
correction()
X_teste_normalized[0]
previsoes = modelo.predict(X_teste_normalized)
resultado = pd.DataFrame({"Id": dados_teste['Id'], 'Sales': np.round(np.expm1(previsoes)).astype(int)})
resultado.head()
idx = dados_teste.loc[dados_teste['Open'] == 0].index

resultado.loc[resultado.iloc[idx].index.values, 'Sales'] = 0

resultado.iloc[idx].head()
resultado.to_csv("submission_v16.csv", index=False)