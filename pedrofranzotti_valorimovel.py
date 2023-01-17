import numpy as np

import pandas as pd



import seaborn as sns

import os

print(os.listdir("../input"))



# O arquivo ripar.py contém o código utilizado para capturar as informações no site
#  Importando os dados

df= pd.read_csv("../input/bancodedados/bd.csv")

df.describe()
#  Limpando os dados

dflimpo = df[df.Metragem > 20]

dflimpo = dflimpo[dflimpo.Metragem<12000]

dflimpo = dflimpo[dflimpo.Quarto>0]

dflimpo = dflimpo[dflimpo.Banheiro>0]

dflimpo = dflimpo[dflimpo.VagaGaragem>=0]



dflimpo.describe()
#  Teste base com Metragem e Preço

dfMetragemPreco = dflimpo[['Metragem','Preco']]

dfMetragemPreco.describe()
#  Gráfico com Metragem e Preço

sns.pairplot(data=dfMetragemPreco, kind="reg")
#  Importando biblioteca para as regressões

from sklearn import linear_model

from sklearn import preprocessing # equaliza a dimensão das variaveis

le = preprocessing.LabelEncoder() # permite alterar o nome para numero

regressao = linear_model.LinearRegression()
#  Calculando a regressão com todo o banco de dados

x = np.array(dfMetragemPreco[['Metragem']])

y = np.array(dfMetragemPreco[['Preco']])

regressao.fit(x, y)
#  Calculando os valores com a base toda

tamanho = 100

print('Valor: ',regressao.predict(np.array(tamanho).reshape(-1, 1)))

tamanho = 200

print('Valor: ',regressao.predict(np.array(tamanho).reshape(-1, 1)))

tamanho = 500

print('Valor: ',regressao.predict(np.array(tamanho).reshape(-1, 1)))

tamanho = 1000

print('Valor: ',regressao.predict(np.array(tamanho).reshape(-1, 1)))
#  Criando array com todos os bairros

nome = []

for index, row in dflimpo.iterrows():

    if row["Bairro"] not in nome:

        nome.append(row["Bairro"])
# Calculando a regressão para cada bairro

# E estimando o preço de uma casa de 100m² com 2 quartos, 2 banheiros e 2 vagas de garagem



for bairro in nome:

    dfMetragemPreco = dflimpo[dflimpo.Bairro==bairro]

    dfMetragemPreco = dfMetragemPreco[['Metragem','Quarto','Banheiro','VagaGaragem','Preco']]

    #dfMetragemPreco.describe()

    x = np.array(dfMetragemPreco[['Metragem','Quarto','Banheiro','VagaGaragem']])

    y = np.array(dfMetragemPreco[['Preco']])

    reg = regressao.fit(x, y)

    print("Para o bairro ", bairro, " temos os indices", reg.coef_, reg.intercept_)

    print("100m², 2 quartos, 2 banheiros e 2 vagas: Preço R$ ",reg.predict([[100,2,2,2]])[0][0])

    