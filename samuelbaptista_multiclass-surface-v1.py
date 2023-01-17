import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



# import quaternion



# Houve um problema com a importação desse pacote.

# Vou comentar as linhas em que era utilizado... Vamos perder duas variáveis no processo.

# Talvez algumas informações da documentação fiquem incorretas. 



from matplotlib import rc_file_defaults



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import GradientBoostingClassifier



from keras.models import Sequential

from keras.layers import Dense, BatchNormalization, Dropout, LSTM, Conv1D, MaxPool1D, Flatten, TimeDistributed

from keras.callbacks import EarlyStopping



import warnings



rc_file_defaults()

plt.style.use('seaborn-notebook')

sns.set_style('whitegrid')



warnings.filterwarnings('ignore')
# Importando os datasets

treino = pd.read_csv('../input/competicao-dsa-machine-learning-sep-2019/X_treino.csv')

teste = pd.read_csv('../input/competicao-dsa-machine-learning-sep-2019/X_teste.csv')

target = pd.read_csv('../input/competicao-dsa-machine-learning-sep-2019/y_treino.csv')
# Definindo uma coluna de identificação para que possa explorar os dados de forma consolidada.

treino['tipo'] = 'treino'

teste['tipo'] = 'teste'
# Juntando datasets de treino e teste para análise exploratória

dados = pd.concat([treino, teste], axis=0)
# Verificando colunas e seus tipos

dados.info()
# Definindo o objeto dos quartênios para facilitar o trabalho

# dados = dados.assign(quaternion=lambda x: quaternion.from_float_array(x.loc[:, 'orientation_X': 'orientation_W']))
# Validando a primeira etapa de transformações

dados.head()
# Validando a primeira etapa de transformações

dados.shape
# Utilizando os objetos para coletar algumas informações

# q = dados.quaternion.values



# Coletando angulo do quartênio

# angle = []

# for i in range(len(q)):

#    angle.append(q[i].angle())

    

# Coletando a variação entre os angulos a cada timestep   

# rotation = [0]

# for i in range(len(angle)-1):

#    rotation.append(angle[i+1]-angle[i])
# Atibuindo colunas com os novos dados.

# dados['angle'] = angle

# dados['rotation'] = rotation
# Removendo colunas que não consideramos importantes

dados.drop(['row_id'], axis=1, inplace=True)

# dados.drop(['quaternion'], axis=1, inplace=True)
# Organizando o indice

dados.reset_index(drop=True, inplace=True)
# Validando a segunda etapa de transformações

dados.head(10)
# Validando a segunda etapa de transformações

dados.tail(10)
# Explorando as dataset com as informações alvo

target.shape
# Explorando as dataset com as informações alvo

target.head()
# Explorando a variável presente no dataset

target.group_id.value_counts().count()
# Explorando a variável presente no dataset

target.groupby('group_id').surface.value_counts().sort_values().head(10)
plt.figure(figsize=(15,5))

sns.countplot(x="group_id", data=target, order = target['group_id'].value_counts().index)

plt.tight_layout()

plt.show()
mean = int(target.groupby('series_id').group_id.value_counts().unstack().sum().mean())

print(f'Temos uma média de {mean} séries de medidas por grupo')
plt.figure(figsize=(10,5))

sns.countplot(y='surface', data = target, order = target['surface'].value_counts().index)

plt.show()
# Verificando o balanceamento das classes

target.surface.value_counts(normalize=True)
# Verificando se as orientações dos quartênios estão normalizadas

dados = dados.assign(orientation_norm=lambda x: round((x.orientation_X ** 2 + x.orientation_Y ** 2 + x.orientation_Z ** 2 + x.orientation_W ** 2)))



print(f'As colunas de orientação estão normalizadas? {sum(dados.orientation_norm) == len(dados.orientation_norm)}')

dados.drop('orientation_norm', axis=1, inplace=True)
# Análise estatística das colunas numéricas

dados.describe()
# Removendo o efeito da aceleração da gravidade no eixo Z

dados.loc[: , 'linear_acceleration_Z'] = dados.loc[: , 'linear_acceleration_Z'] + 9.8
# Verificando quantas medidas temos em cada série

print(f'Temos {treino.series_id.value_counts()[0]} medidas por série')
# Separando as medições de uma série de treino e uma de teste. 

series_0_treino = dados.loc[(dados.series_id == 0) & (dados.tipo == 'treino'), :]

series_0_teste = dados.loc[(dados.series_id == 0) & (dados.tipo == 'teste'), :]
# Resentando as configuração de gráfico e defindo um estilo.

rc_file_defaults()

plt.style.use('seaborn-notebook')



plt.figure(figsize=(10,15))



for i, col in enumerate(series_0_treino.drop(['series_id', 'measurement_number', 'tipo'], axis=1).columns):

    plt.subplot(6, 2, i + 1)

    plt.plot(range(len(series_0_treino)), series_0_treino[col])

    plt.title(col)



plt.tight_layout()

plt.show()
# Verificando a distribuição dos dados de velocidade em cada eixo



# Lista para facilitar a construção dos gráficos

coord = ["X", "Y", "Z"]



# Definindo o tamanho da figura para melhor visualização 

plt.figure(figsize=(9.75,2.5))



# Loop para construção dos gráficos

for i in range(len(coord)):

    plt.subplot(131+i)

    dados[f'angular_velocity_{coord[i]}'].hist(edgecolor='black', log=True, grid=False)

    plt.title(f'Velocidade Eixo {coord[i]}')



# Ajustando o layout

plt.tight_layout()



# Usando o comando 'show' para evitar o texto do objeto criado.

plt.show()
# Mesma configuração da celular anterior, mas avaliando as variáveis de aceleração.

coord = ["X", "Y", "Z"]



plt.figure(figsize=(9.75,2.5))

for i in range(len(coord)):

    plt.subplot(131+i)

    dados[f'linear_acceleration_{coord[i]}'].hist(edgecolor='black', log=True, grid=False)

    plt.title(f'Aceleração Eixo {coord[i]}')



plt.tight_layout()

plt.show()
# Resentando a configuração dos gráficos

rc_file_defaults()



# Definindo a lista para facilitar a construção dos gráficos

coord = ["X", "Y", "Z"]



# Comparando a distribuição dos dados entre treino e teste. 

# Verificando a presença de outliers.

for i in range(len(coord)):

    dados.loc[:, [f'angular_velocity_{coord[i]}', 'tipo']].boxplot(by='tipo', grid=False)

    plt.title(f'Distribuição da Velocidade Angular no Eixo {coord[i]}')

    plt.xlabel('Tipo')



# Usando o comando 'show' para evitar o texto do objeto criado.

plt.show()
# Mesma configuração da celular anterior, mas avaliando as variáveis de aceleração.

rc_file_defaults()

coord = ["X", "Y", "Z"]





for i in range(len(coord)):

    dados.loc[:, [f'linear_acceleration_{coord[i]}', 'tipo']].boxplot(by='tipo', grid=False)

    plt.title(f'Distribuição da Aceleração Linear no Eixo {coord[i]}')

    plt.xlabel('Tipo')





plt.show()
# Definindo a lista para facilitar a construção dos gráficos

# lista = ['angle', 'rotation']



# Definindo o tamanho da figura para melhor visualização 

# plt.figure(figsize=(9.75,5.75))



# Construindo gráfico para verificar a distribuição das variáveis criadas a partir dos quartênios

# for i in range(len(lista)):

#     plt.subplot(221+i)

#     dados[lista[i]].hist(edgecolor='black', log=True, grid=False)

#     plt.title(f'Distribuição dos dados: {lista[i]}')



# plt.tight_layout()

# plt.show()
# A partir dos gráficos boxplot, conseguimos identificar alguns limites da distribuição.

# Removendo outliers, velocidade eixo X

dados.loc[dados.angular_velocity_X > 0.5, 'angular_velocity_X'] = 0.5

dados.loc[dados.angular_velocity_X < -0.5, 'angular_velocity_X'] = -0.5



# Removendo outliers, velocidade eixo Y

dados.loc[dados.angular_velocity_Y > 0.5, 'angular_velocity_Y'] = 0.5

dados.loc[dados.angular_velocity_Y < -0.5, 'angular_velocity_Y'] = -0.5



# Removendo outliers, velocidade eixo Z

dados.loc[dados.angular_velocity_Z > 0.5, 'angular_velocity_Z'] = 0.5

dados.loc[dados.angular_velocity_Z < -0.5, 'angular_velocity_Z'] = -0.5
# Removendo outliers, aceleração eixo X

dados.loc[dados.linear_acceleration_X > 20, 'linear_acceleration_X'] = 20

dados.loc[dados.linear_acceleration_X < -20, 'linear_acceleration_X'] = -20



# Removendo outliers, aceleração eixo Y

dados.loc[dados.linear_acceleration_Y > 20, 'linear_acceleration_Y'] = 20

dados.loc[dados.linear_acceleration_Y < -20, 'linear_acceleration_Y'] = -20



# Removendo outliers, aceleração eixo Z

dados.loc[dados.linear_acceleration_Z > 20, 'linear_acceleration_Z'] = 20

dados.loc[dados.linear_acceleration_Z < -20, 'linear_acceleration_Z'] = -20
# Haja vista que deeplearning é capaz de inativar os neurônios que não são úteis ao modelo,

# vamos acrescentar o máximo de variáveis possíveis aos dados



dados = dados.assign(total_velocity=lambda x: (x.angular_velocity_X ** 2 + x.angular_velocity_Y ** 2 + x.angular_velocity_Z ** 2) ** 0.5)

dados = dados.assign(total_acceleration=lambda x: (x.linear_acceleration_X ** 2 + x.linear_acceleration_Y ** 2 + x.linear_acceleration_Z ** 2) ** 0.5)

dados = dados.assign(total_coords=lambda x: (x.orientation_X ** 2 + x.orientation_Y ** 2 + x.orientation_Z ** 2) ** 0.5)

dados = dados.assign(acceleration_velocity_ratio=lambda x: x.total_velocity / x.total_acceleration)
# Verificando os dados das primeiras séries de medidas para cada dataset(treino e teste) e avaliando se os valores estão coerentes.

dados_median = pd.DataFrame(dados.drop('measurement_number', axis=1).groupby(['series_id', 'tipo'], as_index=False).median())

dados_median.head(6)
# Com os dados de orientação, o modelo superou 90% de acurácia nos testes, mas não generalizou bem.

# Por isso, vamos remover essas variáveis buscando melhorar a generalização do modelo.

dados.drop(['orientation_X', 'orientation_Y', 'orientation_Z', 'orientation_W'], axis=1, inplace=True)
# Vamos escalar os dados a partir do score Z uma vez que percebemos que existe uma variação grande nos valores,

# e a escala MinMax fica prejudicada por isso.



# Criando o objeto StandardScaler

scaler = StandardScaler()



# Escalando os dados originais do dataset

dados.loc[:, 'angular_velocity_X':'linear_acceleration_Z'] = scaler.fit_transform(dados.loc[:, 'angular_velocity_X':'linear_acceleration_Z'])



# Escalando os dados inseridos no dataset

dados.loc[:, 'total_velocity':'acceleration_velocity_ratio'] = scaler.fit_transform(dados.loc[:, 'total_velocity':'acceleration_velocity_ratio'])
# Verificando se as alterações foram feitas corretamente

dados.describe()
# Primeiras linhas do dataset

dados.head()
# Verificando o formato dos dados para a preparação com foco no treinamento dos modelos.

dados.shape
# Separando os dados de treinamento

treino = dados.loc[dados['tipo'] == 'treino', :]
# Verificando as primeiras linhas para verificar se a ordem ainda está correta.

treino.head()
# Verificando as ultimas linhas para validar se o slicing foi feito de maneira correta.

treino.tail()
# Revomendo as colunas não numéricas

treino.drop(['tipo', 'series_id', 'measurement_number'], axis=1, inplace=True)
# Criando um array com os valores para poder redefinir os formatos

treino_v = treino.values
# Verificando o formato

treino_v.shape
# Primeira tranformação:

# Sabemos que são 128 medidas por série, dessa forma, vamos alinhar os dados que representam cada série de medidas.

treino_v = treino_v.reshape(int(len(treino_v)/128), 128*treino_v.shape[1])
# Verificando se a transformação ocorreu como esperado

treino_v.shape
# Transformando em dataframe para facilitar a visualização

treino_df = pd.DataFrame(treino_v)
# Visualizando as primeiras linhas após transformação

treino_df.head()
# Retornando os id´s de cada linha 

treino_df['series_id'] = range(3810)
# Trazendo os dados dos labels

treino_df = treino_df.merge(target, on='series_id')
# Removendo as colunas que não representam informações úteis

treino_df.drop(['group_id', 'series_id'], axis = 1, inplace=True)
# Verificando o dataset após alterações

treino_df.head()
# Verificando o dataset após alterações

treino_df.tail()
# Verificando novamente o balanceamento dos dados

treino_df.surface.value_counts()
# Criando os arrays para inicio da preparação do dataset com foco na criação dos modelos

x = treino_df.drop('surface', axis=1).values

y = treino_df.loc[:, 'surface'].values
# Verificando o formato das variáveis preditoras

x.shape
# Verificando o formato dos labels

y.shape
# Utilizando a técnica de smooting para balancear as classes

from imblearn.over_sampling import SMOTE

x_smt, y_smt = SMOTE().fit_sample(x, y)
# Verificando o formato das variáveis preditoras após transformação

x_smt.shape
# Verificando o formato dos labels após transformação

y_smt.shape
# Verificando o balanceamento dos dados após balanceamento

pd.Series(y_smt).value_counts()
# Definindo os timesteps

timesteps = 128



# Transformando os dados para treinamento do modelo 

x_smt_timesteps = x_smt.reshape(x_smt.shape[0], timesteps, int(x_smt.shape[1]/timesteps))
# Verificando a transformação

x_smt_timesteps.shape
# Verificando o array dos dados de treino

y_smt
# Colocando os dados target na formatação onehotenconding para que o modelo possa ser treinado utilizando a entropia cruzada

# como função de custo



# Criando o objeto

encoder = OneHotEncoder(sparse=False, categories='auto')



# Aplicando aos dados target

y_target = encoder.fit_transform(y_smt.reshape(len(y_smt), 1))
# Verificando a transformação

y_target
# Separando os dados do primeiro modelo em treino e teste para que possamos avaliar o resultado posteriormente

X_train, X_test, y_train, y_test = train_test_split(x_smt, y_target, test_size=0.3)



# Separando os dados para treinamento do modelo GBC. A única diferença está no formato das variáveis target. 

X_train_GBC, X_test_GBC, y_train_GBC, y_test_GBC = train_test_split(x_smt, y_smt, test_size=0.3)
# Verificando a separação 

X_train_GBC.shape
# Validando que variáveis preditoras e target tem o mesmo número de observações

y_train_GBC.shape
# Instanciando o objeto do GradientBoostingClassifier com os respectivos parametros. 

model_GBC = GradientBoostingClassifier(n_estimators=1000,

                                       subsample=0.5,

                                       tol=0.01,

                                       n_iter_no_change=5,

                                       verbose=1,

                                       validation_fraction=0.2,

                                       learning_rate=0.1)



# Treinando o modelo

model_GBC.fit(X_train_GBC, y_train_GBC)
# Criando um objeto do modelo sequencial

model = Sequential()



# Definindo parada antecipada para evitar overfitting do modelo

# Usando como monitor a entropia cruzada nos dados de validação

# Encerramento do treinamento após 25 epochs sem evolução de ao menos 0.001.

stop = EarlyStopping(monitor='val_loss', patience=25, min_delta=0.001)



# Adicionando a primeira camada densa, 1024 neurônios, ativação 'relu'

model.add(Dense(1024, activation='relu'))



# Camada de Dropout para desativar 50% dos neurônios aleatóriamente e buscar melhores resultados na generalização

model.add(Dropout(0.5))



# Segunda camada, 512 neurônios, ativação 'relu'

model.add(Dense(512, activation='relu'))



# Terceira camada, 256 neurônios, ativação 'relu'

model.add(Dense(256, activation='relu'))



# Novo Dropout para ajudar no controle de overfitting

model.add(Dropout(0.5))



# Camada de output com ativação softmax para calcular as probabilidades de cada uma das 9 classes

model.add(Dense(9, activation='softmax'))



# Compilando o modelo utilizando entropia cruzada como função de custo,

# otimizador Adam e Acurácia como métrica de acompanhamento.

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



# Treinamento do modelo, atualizando os pesos a cada 32 passadas, usando 30% dos dados como validação

model.fit(X_train, y_train, epochs=250, callbacks=[stop], verbose=1, validation_split=0.3, batch_size=32)
# Separando os dados, em um segundo formato, para posterior avaliação.

X_train2, X_test2, y_train2, y_test2 = train_test_split(x_smt_timesteps, y_target, test_size=0.3)
# Criando um objeto do modelo sequencial

model2 = Sequential()



# Definindo parada antecipada para evitar overfitting do modelo

# Usando como monitor a entropia cruzada nos dados de validação

# Encerramento do treinamento após 10 epochs sem evolução de ao menos 0.001.

stop = EarlyStopping(monitor='val_loss', patience=10, min_delta=0.001)



# Primeira camada LSTM, ativação 'relu'

model2.add(LSTM(100, activation='relu'))



# Camada de Dropout para desativar 50% dos neurônios aleatóriamente e buscar melhores resultados na generalização

model2.add(Dropout(0.5))



# Adicionando a primeira camada densa, 100 neurônios, ativação 'relu'

model2.add(Dense(100, activation='relu'))



# Adicionando a segunda camada densa, 50 neurônios, ativação 'relu'

model2.add(Dense(50, activation='relu'))



# Camada de output com ativação softmax para calcular as probabilidades de cada uma das 9 classes

model2.add(Dense(9, activation='softmax'))



# Compilando o modelo utilizando entropia cruzada como função de custo,

# otimizador Adadelta (para evitar degradação do gradiente) e Acurácia como métrica de acompanhamento.

model2.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])



# Treinamento do modelo, atualizando os pesos a cada 64 passadas, usando 30% dos dados como validação

model2.fit(X_train2, y_train2, epochs=200, callbacks=[stop], verbose=1, validation_split=0.3, batch_size=64)
# Criando um objeto do modelo sequencial

model3 = Sequential()



# Definindo parada antecipada para evitar overfitting do modelo

# Usando como monitor a entropia cruzada nos dados de validação

# Encerramento do treinamento após 10 epochs sem evolução de ao menos 0.001.

stop = EarlyStopping(monitor='val_loss', patience=10, min_delta=0.001)



# Primeira camada de convolução, 128 filtros, 4x4.

model3.add(Conv1D(128, kernel_size=4, activation='relu'))



# Segunda camada de convolução, 128 filtros, 2x2.

model3.add(Conv1D(128, kernel_size=2, activation='relu'))



# Camada de Dropout para desativar 50% dos neurônios aleatóriamente e buscar melhores resultados na generalização

model3.add(Dropout(0.5))



# Camada de pooling para reduzir a dimensão dos dados. Foi usado o valor máximo de cada pool. 

model3.add(MaxPool1D(pool_size=8))



# Camada que muda o shape dos dados para input nas camadas densas.

model3.add(Flatten())



# Primeira camada densa, 128 

model3.add(Dense(128, activation='relu'))



# Camada de Dropout para desativar 50% dos neurônios aleatóriamente e buscar melhores resultados na generalização

model3.add(Dropout(0.5))



# Camada de output com ativação softmax para calcular as probabilidades de cada uma das 9 classes

model3.add(Dense(9, activation='softmax'))



# Compilando o modelo utilizando entropia cruzada como função de custo,

# otimizador Adamax e Acurácia como métrica de acompanhamento.

model3.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['accuracy'])



# Treinamento do modelo, atualizando os pesos a cada 64 passadas, usando 30% dos dados como validação

model3.fit(X_train2, y_train2, epochs=250, callbacks=[stop], verbose=1, validation_split=0.3, batch_size=64)
# Transformando os dados para treinamento do modelo 

x_smt_timesteps2 = x_smt.reshape(x_smt.shape[0], 4, 32, int(x_smt.shape[1]/timesteps))



# Verificando as alterações

x_smt_timesteps2.shape
# Separando os dados, em um terceiro formato, para posterior avaliação.

X_train3, X_test3, y_train3, y_test3 = train_test_split(x_smt_timesteps2, y_target, test_size=0.3)
# Criando um objeto do modelo sequencial

model4 = Sequential()



# Definindo parada antecipada para evitar overfitting do modelo

# Usando como monitor a entropia cruzada nos dados de validação

# Encerramento do treinamento após 10 epochs sem evolução de ao menos 0.001.

stop = EarlyStopping(monitor='val_loss', patience=10, min_delta=0.001)



# Criando a primeira de convulação considerando o fator tempo.

model4.add(TimeDistributed(Conv1D(filters=128, kernel_size=4, activation='relu')))



# Criando a segunda de convulação considerando o fator tempo.

model4.add(TimeDistributed(Conv1D(filters=64, kernel_size=2, activation='relu')))



# Criando uma camada de dropout considerando o fator tempo.

model4.add(TimeDistributed(Dropout(0.5)))



# Camada de pooling para reduzir a dimensão dos dados considerando o fator tempo. Foi usado o valor máximo de cada pool. 

model4.add(TimeDistributed(MaxPool1D(pool_size=4)))



# Camada que muda o shape dos dados para input na camada LSTM. (Reduz de 4 para 3 dimensões)

model4.add(TimeDistributed(Flatten()))



# Primeira camada LSTM 

model4.add(LSTM(100, activation='relu'))



# Camada de Dropout para desativar 50% dos neurônios aleatóriamente e buscar melhores resultados na generalização

model4.add(Dropout(0.5))



# Camada de output com ativação softmax para calcular as probabilidades de cada uma das 9 classes

model4.add(Dense(9, activation='softmax'))



# Compilando o modelo utilizando entropia cruzada como função de custo,

# otimizador Adamax e Acurácia como métrica de acompanhamento.

model4.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['accuracy'])



# Treinamento do modelo, atualizando os pesos a cada 64 passadas, usando 30% dos dados como validação

model4.fit(X_train3, y_train3, epochs=250, callbacks=[stop], verbose=1, validation_split=0.3, batch_size=64)
# Avaliando os modelos nos dados de teste

eval1 = model.evaluate(X_test, y_test, batch_size=32, verbose=2)

eval2 = model2.evaluate(X_test2, y_test2, batch_size=64, verbose=2)

eval3 = model3.evaluate(X_test2, y_test2, batch_size=64, verbose=2)

eval4 = model4.evaluate(X_test3, y_test3, batch_size=64, verbose=2)



# Imprimindo os resultados de forma a facilitar a leitura

print(f'\nModelo GBC teve {model_GBC.score(X_test_GBC, y_test_GBC):0.2f}% de Acurácia')

print(f'Modelo MLPerceptron teve {eval1[1]:0.2f}% de Acurácia')

print(f'Modelo LSTM teve {eval2[1]:0.2f}% de Acurácia')

print(f'Modelo CNN teve {eval3[1]:0.2f}% de Acurácia')

print(f'Modelo CNN LSTM teve {eval4[1]:0.2f}% de Acurácia')
# Selecionando os dados normalizados

teste = dados.loc[dados['tipo'] == 'teste', :]
# Removendo colunas que não são importantes

teste.drop(['tipo', 'series_id', 'measurement_number'], axis=1, inplace=True)
# Criando o array para manipulação dos dados

x_teste = teste.values
# Verificando o formato

x_teste.shape
# Mudando o shape para alinhar com o formato esperado pelo modelo MLP

x_teste = x_teste.reshape(int(len(x_teste)/128), 128*x_teste.shape[1])
# Verificando a alteração

x_teste.shape
# Transformando em dataframe para facilitar a visualização.

teste_df = pd.DataFrame(x_teste)

teste_df['series_id'] = range(3816)
# Visualizando as primeiras linhas

teste_df.head()
# Visualizando as ultimas linhas

teste_df.tail()
# Prevendo as classes para o modelo MLP

prev1 = model.predict(x_teste)



# Revertendo a transformação para que a resposta esteja com o nome da superfície.

classes_teste1 = encoder.inverse_transform(prev1)

classes_teste1 = pd.Series(classes_teste1.reshape(3816,))



# Avaliando as previsões. 

print(classes_teste1.value_counts(), end="\n\n")

print(classes_teste1.value_counts(normalize=True))
# Redefinindo para 128 timesteps

x_teste_timesteps = x_teste.reshape(x_teste.shape[0], timesteps, int(x_teste.shape[1]/timesteps))
# Verificando as alterações

x_teste_timesteps.shape
# Prevendo as classes para o modelo LSTM

prev2 = model2.predict(x_teste_timesteps)



# Revertendo a transformação para que a resposta esteja com o nome da superfície.

classes_teste2 = encoder.inverse_transform(prev2)

classes_teste2 = pd.Series(classes_teste2.reshape(3816,))



# Avaliando as previsões. 

print(classes_teste2.value_counts(), end="\n\n")

print(classes_teste2.value_counts(normalize=True))
# Prevendo as classes para o modelo CNN

prev3 = model3.predict(x_teste_timesteps)



# Revertendo a transformação para que a resposta esteja com o nome da superfície.

classes_teste3 = encoder.inverse_transform(prev3)

classes_teste3 = pd.Series(classes_teste3.reshape(3816,))



# Avaliando as previsões. 

print(classes_teste3.value_counts(), end="\n\n")

print(classes_teste3.value_counts(normalize=True))
# Redefinindo para 4 passos que serão realizados 32 vezes.

x_teste_timesteps2 = x_teste.reshape(x_teste.shape[0], 4, 32, int(x_teste.shape[1]/timesteps))



# Verificando as alterações.

x_teste_timesteps2.shape
# Prevendo as classes para o modelo CNN-LSTM

prev4 = model4.predict(x_teste_timesteps2)



# Revertendo a transformação para que a resposta esteja com o nome da superfície.

classes_teste4 = encoder.inverse_transform(prev4)

classes_teste4 = pd.Series(classes_teste4.reshape(3816,))



# Avaliando as previsões.

print(classes_teste4.value_counts(), end="\n\n")

print(classes_teste4.value_counts(normalize=True))
# Prevendo as classes para o modelo GBC

prev5 = model_GBC.predict(x_teste)



classes_teste5 = pd.Series(prev5)



# Avaliando as previsões.

print(classes_teste5.value_counts(), end="\n\n")

print(classes_teste5.value_counts(normalize=True))
# Criando o dataframe para formatação da resposta do modelo MLP

resposta1 = pd.DataFrame({'series_id': range(3816)})

resposta1['surface'] = classes_teste1



# Verificando o dataframe

resposta1.head()
# Criando o dataframe para formatação da resposta do modelo LSTM

resposta2 = pd.DataFrame({'series_id': range(3816)})

resposta2['surface'] = classes_teste2



# Verificando o dataframe

resposta2.head()
# Criando o dataframe para formatação da resposta CNN

resposta3 = pd.DataFrame({'series_id': range(3816)})

resposta3['surface'] = classes_teste3



# Verificando o dataframe

resposta3.head()
# Criando o dataframe para formatação da resposta do modelo CNN-LSTM

resposta4 = pd.DataFrame({'series_id': range(3816)})

resposta4['surface'] = classes_teste4



# Verificando o dataframe

resposta4.head()
# Criando o dataframe para formatação da resposta do modelo GBC

resposta5 = pd.DataFrame({'series_id': range(3816)})

resposta5['surface'] = classes_teste5



# Verificando o dataframe

resposta5.head()
print('-=' * 35)

print('\n', resposta5.surface.head().values, '\n', resposta4.surface.head().values,'\n', resposta3.surface.head().values,

     '\n', resposta2.surface.head().values, '\n', resposta1.surface.head().values, '\n')

print('-=' * 35)
# Gerando os arquivos para submissão das respostas

resposta1.to_csv('resposta_MLP.csv', index=False)

resposta2.to_csv('resposta_LSTM.csv', index=False)

resposta3.to_csv('resposta_CNN.csv', index=False)

resposta4.to_csv('resposta_CNN_LSTM.csv', index=False)

resposta5.to_csv('resposta_GBC.csv', index=False)