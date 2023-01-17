# Selecionando e carregando os dados
import pandas as pd
colunas_selecionadas = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long']
dados = pd.read_csv('../input/kc_house_data.csv', usecols=colunas_selecionadas)
print('Quantidade de linhas: ', dados.shape[0])
print('Quantidade de colunas: ', dados.shape[1])
dados.head()
# Fazer o escalonamento dos valores

   # é possível fazer a padronização ou normalização (escala entre zero e um)
from sklearn.preprocessing import MinMaxScaler

# Escaler para as variaveis do eixo X
scaler_x = MinMaxScaler()
features_selected = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long']
dados[features_selected] = scaler_x.fit_transform(dados[features_selected])
dados.head()
# Escaler para as variaveis do eixo Y (Preço)
scaler_y = MinMaxScaler()
dados[['price']]  = scaler_y.fit_transform(dados[['price']])
dados.head() # Exibe os dados escalonados
# Separa as colunas que serão utilizadas para a análise preditiva
feature_columns = dados.drop('price', axis=1)
feature_columns.head()

# separa a coluna que contém o resultado conhecido e que podemos utilizar para validar o modelo (variavel dependente)
price_column = dados.price
price_column.head()
# A seguir iremos montar a logica de predição com TensorFlow para chegarmos à predição

# Criando as Features Columns do tipo Numeric no TensorFlow para cada coluna preditora
import tensorflow as tf
tf_columns = [tf.feature_column.numeric_column(key = c) for c in features_selected]
print('Lista de features TensorFlow:\r')
for c in tf_columns:
    print(c)
# fazendo a divisão dos dados entre treinamento e teste
from sklearn.model_selection import train_test_split
features_train, features_test, price_train, price_test = train_test_split(feature_columns, price_column, test_size = 0.3)
print('Quantidade de registros de treinamento: ' + str(features_train.shape[0]))
print('Quantidade de registros de teste: ' + str(features_test.shape[0]))
# definindo as funções de treinamento e teste usando Estimator do TensorFlow
train_function = tf.estimator.inputs.pandas_input_fn(x = features_train, y = price_train, batch_size = 32, num_epochs = None, shuffle = True)
test_function = tf.estimator.inputs.pandas_input_fn(x = features_test, y = price_test, batch_size = 32, num_epochs = 10000, shuffle = True)
regressor = tf.estimator.LinearRegressor(feature_columns = tf_columns)
# executar o treinamento do modelo
regressor.train(input_fn=train_function, steps=10000)
# obtendo as métricas do treinamento
train_metrics = regressor.evaluate(input_fn=train_function, steps=10000)
train_metrics
# obtendo as metricas do teste
test_metrics = regressor.evaluate(input_fn=test_function, steps=10000)
test_metrics
# criando a previsão
predict_function = tf.estimator.inputs.pandas_input_fn(x = features_test, shuffle=False)
predictions = regressor.predict(input_fn=predict_function)
list(predictions)[:10]
prediction_values = []
for p in regressor.predict(input_fn=predict_function):
    prediction_values.append(p['predictions'])
prediction_values[:10]
# Transformado os dados de volta para a escala original dos valores
import numpy as np
prediction_values = np.asarray(prediction_values).reshape(-1,1) # coloca em formato de matriz
prediction_values = scaler_y.inverse_transform(prediction_values)
prediction_values[:10]
# Preparando os dados para comparar
price_test_matrix = price_test.values.reshape(-1,1) # transformando o array para matriz
price_test_matrix = scaler_y.inverse_transform(price_test_matrix) # voltando os valores para a escala original
price_test_matrix[:10]
# medindo a qualidade do algoritmo treinado
from sklearn.metrics import mean_absolute_error
mean = mean_absolute_error(price_test_matrix, prediction_values)
mean

# o algoritmo treinado está "errando" em cerca de $134234 para cima ou para baixo o preço da casa
# considerando múltiplas features em um algoritmo de Regressão Linear Múltipla com o Estimators do TensorFlow.