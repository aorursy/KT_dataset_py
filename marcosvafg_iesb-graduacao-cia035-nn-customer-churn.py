# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Carregando os dados

df = pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')



df.shape
# Olhando os dados

df.head().T
# % de clientes que deixaram a operadora

df['Churn'].value_counts(normalize=True)
# Verificando os tipos dos dados e os tamanhos

df.info()
# Para corrigir o TotalCharges vamos trocar o espaço em branco

# pelo valor ZERO e forçar a conversão para float

df['TotalCharges'] = df['TotalCharges'].str.replace(' ', '0').astype(float)
# Convertendo as colunas categórias em colunas numéricas

for col in df.columns:

    if df[col].dtype == 'object':

        df[col] = df[col].astype('category').cat.codes
# Verificando os tipos dos dados e os tamanhos

df.info()
# Importando o PyTorch

import torch

import torch.nn as nn
# Importando o train_test_split

from sklearn.model_selection import train_test_split



# Preparar e separar os dataframes

X = df.drop(['customerID', 'Churn'], axis = 1).values

y = df['Churn'].values



# Separando os dados

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Transformando os dados em tensores

X_train = torch.FloatTensor(X_train)

X_test = torch.FloatTensor(X_test)

y_train = torch.LongTensor(y_train)

y_test = torch.LongTensor(y_test)
# Visualizando os dados de treino

X_train[0:5]
# Visualindos as resposta de treino

y_train[0:5]
# Definindo uma rede neural (fully connected)

# Input layer: 19 entradas (variáveis de entrada) -> 16 saídas (arbitrário/definido por nós)

# Hideen layer: 16 entradas (combinando com as saídas da camada anterior) -> 12 saídas (arbitrário/definido por nós)

# Hideen layer: 12 entradas (combinando com as saídas da camada anterior) -> 6 saídas (arbitrário/definido por nós)

# Output layer: 6 entradas (combinando com as saídas da camada anterior) - 2 saídas (Churn 1 ou Churn 0)



# Criando a rede neural usando nn.Sequential

model = nn.Sequential(nn.Linear(19, 16),

                      nn.ReLU(),

                      nn.Linear(16, 12),

                      nn.ReLU(),

                      nn.Linear(12, 6),

                      nn.ReLU(),

                      nn.Linear(6, 2))



model
# Temos que definir a função de erro e o otimizador (qu vai alterar os pesos dos perceptrons)

error_function = nn.CrossEntropyLoss() # criterion

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# Treinamento do modelo



# Definindo o número de épocas

epochs = 5000



# Erros

running_loss = []



# For para rodar o número de épocas

for i in range(epochs):

    # Treinamento

    

    # Foward Propagation (passando os dados de treino pela rede)

    outputs = model.forward(X_train)

    # Calculando o erro

    loss = error_function(outputs, y_train)

    # Guadando o erro

    running_loss.append(loss)

    

    # Exibindo o erro de 5 em 5 épocas

    if i % 5 == 0:

        print(f'Epoch: {i} / Loss: {loss}')

        

    # Back Propagation

    # Limpar os parametros do otimizador (zerar o Gradiente Descendent)

    optimizer.zero_grad()

    # Calcular os novos pesos

    loss.backward()

    # Executar o optimizador (efetivamente fazer o back propagation mudando os pesos)

    optimizer.step()
# Verificando os 5 últimos valores da função de erro

running_loss[-5:]
# Previsões para os dados de teste



# Lista das previsões

preds = []



# Colocar a rede em modo de execução/previsão / tirar do modo de treinamento

with torch.no_grad():

    for val in X_test:

        predict = model.forward(val)

        preds.append(predict.argmax().item())
# Criando um dataframe com os resultados

df_result = pd.DataFrame({'Y': y_test, 'YHat': preds})

df_result['Correct'] = [1 if corr == pred else 0 for corr, pred in zip(df_result['Y'], df_result['YHat'])]



df_result.head()
# Medindo a acurácia

df_result['Correct'].sum() / len(df_result)