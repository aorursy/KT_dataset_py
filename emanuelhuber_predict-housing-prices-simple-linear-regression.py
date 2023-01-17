# Importando as bibliotecas necessárias

import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

from scipy.stats import norm

from math import sqrt



!pip install -U mxnet-cu101mkl==1.6.0  # updating mxnet to at least v1.6

!pip install d2l==0.13.2 -f https://d2l.ai/whl.html # installing d2l





# Código padrão do kaggle para exibir as bases de dados presentes no ambiente atual

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# Carregando o dataset em memória com base no retorno da última célula

df_house = pd.read_csv('../input/kc_house_data.csv')
# Vizualisando o dataset

df_house.head()
# Número total de amostras e atributos

df_house.shape
# Exibindo os tipos de cada atributo

df_house.dtypes
from sklearn.model_selection import train_test_split



X = df_house.drop(['price'],axis =1)

y = df_house['price']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(f"Número total de amostras: {X_train.shape[0]}")

print(f"Número total de ids únicos: {X_train['id'].unique().shape[0]}")

print(f"Existem {X_train.shape[0] - X_train['id'].unique().shape[0]} amostras com ids duplicados")
duplicated_sample = X_train[ X_train['id'].duplicated() ].iloc[0]

df_house[ df_house['id'] == duplicated_sample['id']]
y_train = pd.DataFrame(y_train).drop(pd.DataFrame(y_train)[pd.DataFrame(X_train)['id'].duplicated()].index)

X_train = pd.DataFrame(X_train).drop_duplicates(subset='id')



X_train = X_train.drop(['id','date'],axis =1)

X_train.head()
# Verificando se há valores vazios

df_house.isnull().sum().sort_values(ascending = False)
min_ = min(y_train['price'])

max_ = max(y_train['price'])

x = np.linspace(min_,max_,100)

mean = np.mean(y_train['price'])

std = np.std(y_train['price'])



# For Histogram

plt.hist(y_train['price'], bins=20, density=True, alpha=0.3, color='b')

y = norm.pdf(x,mean,std)



# For normal curve

plt.plot(x,y, color='red')



plt.show()
correlation_matrix = df_house.corr()

print(correlation_matrix)
sns.heatmap(correlation_matrix)
# Price & Sqft Living

df_house.plot(x='sqft_living',y='price',style = 'o')

plt.title('Sqft_Living Vs Price')
from d2l import mxnet as d2l

from mxnet import autograd, gluon, np, npx

npx.set_np()



def load_array(data_arrays, batch_size, is_train=True):  #@save

    """Construct a Gluon data loader."""

    dataset = gluon.data.ArrayDataset(*data_arrays)

    return gluon.data.DataLoader(dataset, batch_size, shuffle=is_train)



# Normalização do atributo sqft_living

X_train_simple = X_train['sqft_living'].apply(

    lambda x: (x - X_train['sqft_living'].mean()) / (X_train['sqft_living'].std()))



# Convertendo para o numpy do mxnet

X_train_simple = np.array(X_train_simple.values.astype('float32'))

y_train_simple = np.array(y_train.values.astype('float32'))



# Normalização do atributo sqft_living

X_test_simple = X_test['sqft_living'].apply(

    lambda x: (x - X_train['sqft_living'].mean()) / (X_train['sqft_living'].std()))



# Convertendo para o numpy do mxnet

X_test_simple = X_test_simple.values.reshape(-1,1)

y_test_simple = y_test.values.reshape(-1,1)



X_test_simple = np.array(X_test_simple.astype('float32'))

y_test_simple = np.array(y_test_simple.astype('float32'))



# Converter para o numpy do mxnet



batch_size = 10

data_iter = load_array((X_train_simple, y_train_simple), batch_size)
# Definindo o modelo

from mxnet.gluon import nn

net = nn.Sequential()

net.add(nn.Dense(1))
# Inicializando os pesos

from mxnet import init

net.initialize(init.Normal(sigma=0.01))
# Definindo a função de custo

from mxnet.gluon import loss as gloss

loss = gloss.L2Loss()
# Definindo o algoritmo de otimização

from mxnet import gluon

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.0003})
# Treinamento da rede neural



num_epochs = 10

for epoch in range(1, num_epochs + 1):

    for X, y in data_iter:

        with autograd.record():

            l = loss(net(X), y)

        l.backward()

        trainer.step(batch_size)

    l = loss(net(X_train_simple), y_train_simple)

    print('epoch %d, loss: %f' % (epoch, l.mean().asnumpy()))
# Exibir os coeficientes finais



w = net[0].weight.data()

print('w', w[0])

b = net[0].bias.data()

print('b', b[0])
# Testando o modelo com uma amostra 

x_sample = np.array([X_train_simple[1]])

y_sample = y_train_simple[1]

print(f"Área livre: {X_train.iloc[1]['sqft_living']}")

print(f"Valor estimado pelo modelo: {net(x_sample)[0][0]}, valor real: {y_sample[0]}")
# plotar os coeficientes com as features

pred = net(X_train_simple)

plt.plot(X_train['sqft_living'].values, y_train['price'].values, 'g^', X_train['sqft_living'].values, pred, 'r--')
# Normalização do atributo sqft_living



def standard_normalization(df, column):

    return df[column].apply(

        lambda x: (x - df[column].mean()) / (df[column].std())

    )



columns = ['sqft_living', 'sqft_living15', 'sqft_above', 'bedrooms', 'bathrooms', 'grade']



X_train_simple = X_train[columns]



for column in columns:

    X_train_simple[column] = standard_normalization(X_train_simple, column)



# Convertendo para o numpy do mxnet

X_train_simple = np.array(X_train_simple.values.astype('float32'))

y_train_simple = np.array(y_train.values.astype('float32'))



X_test_simple = X_test[columns]



for column in columns:

    X_test_simple[column] = standard_normalization(X_test_simple, column)



X_test_simple = np.array(X_test_simple.values.astype('float32'))

y_test_simple = np.array(y_test.values.astype('float32'))



# Converter para o numpy do mxnet



batch_size = 10

data_iter = load_array((X_train_simple, y_train_simple), batch_size)
# Treinamento da rede neural



net = nn.Sequential()

net.add(nn.Dense(1))

net.initialize(init.Normal(sigma=0.01))

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.0003})



num_epochs = 20

for epoch in range(1, num_epochs + 1):

    for X, y in data_iter:

        with autograd.record():

            l = loss(net(X), y)

        l.backward()

        trainer.step(batch_size)

    l = loss(net(X_train_simple), y_train_simple)

    print('epoch %d, loss: %f' % (epoch, l.mean().asnumpy()))
print(f"Train final loss: {loss(net(X_train_simple), y_train_simple).mean().asnumpy()}")

print(f"Test final loss: {loss(net(X_test_simple), y_test_simple).mean().asnumpy()}")