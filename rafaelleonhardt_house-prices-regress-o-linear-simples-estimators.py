# Importaçao dos dados conhecidos
import pandas as pd
dados = pd.read_csv('../input/kc_house_data.csv')
print('Quantidade de linhas: ', dados.shape[0])
print('Quantidade de colunas: ', dados.shape[1])
dados.head()
# Extração das features e valores a serem previstos


# obtem-se os valores da coluna de precos dos imoveis
#    para não precisar fazer o reshape, informa-se a ':coluna+1'
#    caso contrário, o precos.shape seria (qtde_linhas,nulo) ao  invés de (qtde_linhas,uma_coluna)
coluna_precos_indice = 2
precos = dados.iloc[:,coluna_precos_indice:coluna_precos_indice+1].values
print('Precos: \n', precos)

coluna_metragem_indice = 5
metragens = dados.iloc[:,coluna_metragem_indice:coluna_metragem_indice+1].values
print('\nMetragens: \n', metragens)

# exibir os dados graficamente
import matplotlib.pyplot as plt
%matplotlib inline
plt.scatter(metragens, precos)
plt.title('House Prices in King County, USA')
plt.xlabel('Metragens')
plt.ylabel('Preços')



# Escalonamento dos dados
#   Para trabalhar com dados em uma mesma proporção é preciso aplicar uma escala neles.

from sklearn.preprocessing import StandardScaler
scaler_precos = StandardScaler()
precos_escalonados = scaler_precos.fit_transform(precos)
print('Precos escalonados: \n', precos_escalonados)

scaler_metragem = StandardScaler()
metragens_escalonadas = scaler_metragem.fit_transform(metragens)
print('\nMetragens escalonadas: \n', metragens_escalonadas)

# exibir os dados graficamente
plt.scatter(metragens_escalonadas, precos_escalonados)
plt.title('House Prices in King County, USA (em escala)')
plt.xlabel('Metragem')
plt.ylabel('Preços')
# Calcula-se os valores iniciais (pesos iniciais) para b0 e b1
#   com a Gradient Descendent esses valores são alterados pelo treinamento até se chegar ao valor ideal
import numpy as np
np.random.seed(0) # força para fins de estudo que sempre gere os mesmos valores randômicos
valores_aleatorios = np.random.random(2) # gera dois numeros aleatorios
print('Valores aleatorios iniciais: ', valores_aleatorios)
b0_inicial = valores_aleatorios[0]
print('b0: ', b0_inicial)
b1_inicial = valores_aleatorios[1]
print('b1: ', b1_inicial)
from sklearn.model_selection import train_test_split
metragens_treino, metragens_teste, precos_treino, precos_teste \
    = train_test_split(metragens_escalonadas, precos_escalonados, test_size=0.3)

print('Quantidade de dados de treino: ', metragens_treino.shape[0])
print('Quantidade de dados de teste: ', metragens_teste.shape[0])
import tensorflow as tf

# Preparação dos dados para os Estimators de Regressão Linear do TensorFlow
colunas = [tf.feature_column.numeric_column('metragens', shape=[1])]

# Cria o objeto de Regressão Linear usando os Estimators (High Level API do TensorFlow)
regressor = tf.estimator.LinearRegressor(feature_columns=colunas)
# Cria a funcão de treinamento usando Estimator
#  os dados de entrada estão em um formato numpy
#  o processamento ocorrerá em lotes de 32 amostras
#  os valores de amostra serão sorteados
#  a quantiadade de épocas no treinamento é definido depois.
funcao_treino = tf.estimator.inputs.numpy_input_fn({'metragens': metragens_treino}, precos_treino, batch_size=32, num_epochs=None, shuffle=True)
funcao_teste  = tf.estimator.inputs.numpy_input_fn({'metragens': metragens_teste}, precos_teste, batch_size=32, num_epochs=1000, shuffle=False)

# aumentar o nivel de log exibido pelo TensorFlow
tf.logging.set_verbosity('INFO')

# executa-se o treinamento
#   em uma época, é percorrido todos os valores
#   em um step é do gradient descent
regressor.train(input_fn= funcao_treino, steps=10000)
# Métricas

metricas_treino = regressor.evaluate(input_fn=funcao_treino, steps=10000)
print('Métricas de treino:\n', metricas_treino)
metricas_teste = regressor.evaluate(input_fn=funcao_teste, steps=10000)
print('Métricas de teste:\n', metricas_teste)
# Obtendo previsão para um novo valor
import numpy as np
novas_casas = np.array([[800],[900],[1000]])
novas_casas_escalonadas = scaler_metragem.transform(novas_casas)

funcao_previsao = tf.estimator.inputs.numpy_input_fn({'metragens': novas_casas_escalonadas}, shuffle=False)
previsoes = regressor.predict(input_fn=funcao_previsao)

# visualizando os valores
novos_precos = np.array([])
print('Previsão de preço para as novas casas:')
for p in regressor.predict(input_fn=funcao_previsao):
    novo_valor = scaler_precos.inverse_transform(p['predictions'])
    np.append(novos_precos, novo_valor)
    print(novo_valor)