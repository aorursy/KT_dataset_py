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
# Carregando o dataset

df = pd.read_csv('/kaggle/input/iris/Iris.csv')



df.shape
# Verificando os dados

df.info()
# Olhando os dados

df.head()
# Olhando a coluna Species

df['Species'].value_counts()
# Convertendo a coluna Species

df['Species'] = df['Species'].map({'Iris-versicolor': 0, 'Iris-setosa': 1, 'Iris-virginica': 2})



df.info()
# Importando o PyTorch

import torch

import torch.nn as nn
# Importando o train_test_split

from sklearn.model_selection import train_test_split



# Preparar e separar os dataframes

X = df.drop(['Id', 'Species'], axis = 1).values

y = df['Species'].values



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

# Input layer: 4 entradas (variáveis de entrada) -> 16 saídas (arbitrário/definido por nós)

# Hideen layer: 16 entradas (combinando com as saídas da camada anterior) -> 12 saídas (arbitrário/definido por nós)

# Output layer: 12 entradas (combinando com as saídas da camada anterior) - 3 saídas (são 3 espécies a serem previstas)

# Vamos usar ReLu como função de ativação



# Criando a rede neural usando nn.Sequential

model = nn.Sequential(nn.Linear(4, 16),

                      nn.ReLU(),

                      nn.Linear(16, 12),

                      nn.ReLU(),

                      nn.Linear(12, 3))



model
# Temos que definir a função de erro e o otimizador (qu vai alterar os pesos dos perceptrons)

error_function = nn.CrossEntropyLoss() # criterion

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# Treinamento do modelo



# Definindo o número de épocas

epochs = 100



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

    

    # Exibindo o erro de 10 em 10 épocas

    if i % 10 == 0:

        print(f'Epoch: {i} / Loss: {loss}')

        

    # Back Propagation

    # Limpar os parametros do otimizador (zerar o Gradiente Descendent)

    optimizer.zero_grad()

    # Calcular os novos pesos

    loss.backward()

    # Executar o optimizador (efetivamente fazer o back propagation mudando os pesos)

    optimizer.step()
# Previsões para os dados de teste



# Lista das previsões

preds = []



# Colocar a rede em modo de execução/previsão / tirar do modo de treinamento

with torch.no_grad():

    for val in X_test:

        predict = model.forward(val)

        preds.append(predict.argmax().item())

        

preds
# Verificando os valores reais

y_test
# Vamos criar um Rede Neural no Pytorch usando uma subclasse de nn.Module



# Importando a biblioteca funcional do PyTorch

import torch.nn.functional as F



# Definindo a classe

class RedeNeural(nn.Module):

    

    # Função de inicialização da rede

    def __init__(self):

        # Chamada ao método __init__ da classe mãe

        super().__init__()

        # Vamos definir as camadas

        self.fc1 = nn.Linear(4, 16)

        self.fc2 = nn.Linear(16,12)

        self.output = nn.Linear(12, 3)

        

    # Função para executar o Feed Forward (Forward Propagation)

    def forward(self, x):

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.output(x)

        

        return x

    

model2 = RedeNeural()



model2
# Treinamento do modelo model2



# Redefinindo o optimizer

optimizer = torch.optim.Adam(model2.parameters(), lr=0.01)



# Definindo o número de épocas

epochs = 100



# Erros

running_loss = []



# For para rodar o número de épocas

for i in range(epochs):

    # Treinamento

    

    # Foward Propagation (passando os dados de treino pela rede)

    outputs = model2.forward(X_train)

    # Calculando o erro

    loss = error_function(outputs, y_train)

    # Guadando o erro

    running_loss.append(loss)

    

    # Exibindo o erro de 10 em 10 épocas

    if i % 10 == 0:

        print(f'Epoch: {i} / Loss: {loss}')

        

    # Back Propagation

    # Limpar os parametros do otimizador (zerar o Gradiente Descendent)

    optimizer.zero_grad()

    # Calcular os novos pesos

    loss.backward()

    # Executar o optimizador (efetivamente fazer o back propagation mudando os pesos)

    optimizer.step()
# Previsões para os dados de teste usando o model2



# Lista das previsões

preds = []



# Colocar a rede em modo de execução/previsão / tirar do modo de treinamento

with torch.no_grad():

    for val in X_test:

        predict = model2.forward(val)

        preds.append(predict.argmax().item())

        

preds