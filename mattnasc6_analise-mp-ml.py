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
# Importação das bibliotecas a serem utilizadas.



import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import h5py



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras import callbacks as C
# Importação das bases de dados a serem utilizadas na análise.



cust_mp = pd.read_csv("../input/cust_mp.csv")

faturamento_jan19 = pd.read_csv("../input/faturamento_jan19.csv")
# Verificando a quantidade de dados/variáveis de cada dataset.



print(cust_mp.shape)

print(faturamento_jan19.shape)
# Pegando 10 valores aleatórios da base para ter uma noção de como são as

# variáveis que estamos lidando. Basicamente, desejamos ter uma noção inicial

# de quais variáveis são quantitativas e quais são qualitativas.



cust_mp.sample(10)
faturamento_jan19.sample(10)
# Shape do banco de dados 'faturamento_jan19' antes do agrupamento.



faturamento_jan19.shape
# Deletando a variável 'date' que não será utilizada e agrupando os valores de 

# faturamento por cliente.



faturamento_jan19 = faturamento_jan19.drop(['date'], axis=1)

faturamento_jan19 = faturamento_jan19.groupby('id').sum()
# Shape do banco de dados 'faturamento_jan19' depois do agrupamento.



faturamento_jan19.shape
# Verificando a proporção de cada classe da variável 'classificacao_vendedor'.



cust_mp['classificacao_vendedor'].value_counts()
# Criando as colunas de variável dummy para a variável 'classificacao_vendedor'.



d1 = pd.get_dummies(cust_mp['classificacao_vendedor'], prefix='d1')

d1.head()
# Criando as colunas de variável dummy para a variável 'produto_utilizado_30'.



d2 = pd.get_dummies(cust_mp['produto_utilizado_30'], prefix='d2')

d2.head()
# Deletando as variáveis 'classificacao_vendedor' e 'produto_utilizado_30'

# e adicionando as dummies correspondentes.



cust_mp = cust_mp.drop(['classificacao_vendedor', 'produto_utilizado_30'], axis=1)

cust_mp = pd.concat([cust_mp, d1, d2], axis=1)

cust_mp.head()
# Transformando o index (id) do banco de dados 'faturamento_jan19' em coluna

# para a posterior realização do merge.



faturamento_jan19 = faturamento_jan19.reset_index(level=0)

faturamento_jan19.head()
# Merge das tabela pela chave 'id'



df = pd.merge(cust_mp, faturamento_jan19)

df = df.reset_index(drop=True)

df.head(15)
# Frequência absoluta de cada nível da variável resposta.



df['ativo_90'].value_counts()
# Visualização gráfica da frequência absoluta de cada nível da variável

# resposta.



sns.countplot(x='ativo_90', data=df, palette='hls')

plt.show()
# Agrupamento dos níveis da variável resposta através da média.



df.groupby('ativo_90').mean()
# Definição das variáveis independentes e da variável dependente.



columns_X = ['qtd_devolucao_30', 'maquina_ativa', 'cartao', 'idade_dias',

          'perc_boleto', 'perc_cancelamento', 'valor_cancelado', 'd1_1',

          'd1_2', 'd1_3', 'd1_4', 'd1_5', 'd1_6', 'd2_OFF', 'd2_ON',

          'd2_QR', 'faturamento_30']

columns_y = ['ativo_90']



X = df[columns_X]

y = df[columns_y]
# Divisão entre base de treino e teste. A base de treino conterá as únicas informações que o modelo verá.

# Dessa forma, acompanharemos a acurácia do modelo através da base de teste, a qual ele nunca viu.



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# Padronização das variáveis quantitativas, uma vez que elas estão em

# escalas diferentes.



quant_variables = ['qtd_devolucao_30', 'perc_boleto', 'perc_cancelamento',

                  'valor_cancelado', 'idade_dias', 'faturamento_30']

scaler = StandardScaler()

X_train[quant_variables] = scaler.fit_transform(X_train[quant_variables])

X_test[quant_variables] = scaler.transform(X_test[quant_variables])
# Callbacks de treinamento



callbacks_list = [C.EarlyStopping(monitor='val_loss', patience=1),

                  C.ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)

                 ]
# Construção do modelo



model = Sequential()

model.add(Dense(1, activation='sigmoid', input_dim=X_train.shape[1]))
# Compilando o modelo

model.compile(optimizer='adam', loss='binary_crossentropy',

             metrics=['acc'])

epochs_hist = model.fit(X_train, y_train, epochs=1000, batch_size=64,

                       validation_data=(X_test, y_test), callbacks=callbacks_list)
predicted = model.predict(X_test)

predicted



true_predicted = []

for item in predicted:

    if item <= 0.5:

        true_predicted.append(0)

    else:

        true_predicted.append(1)
confusionmatrix = confusion_matrix(y_test, true_predicted)

confusionmatrix
from tensorflow.keras.models import load_model



model = load_model('best_model.h5')
w = model.get_weights()

w