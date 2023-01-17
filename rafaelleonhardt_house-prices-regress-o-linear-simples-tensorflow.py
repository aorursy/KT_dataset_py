# Importaçao dos dados conhecidos
import pandas as pd
dados = pd.read_csv('../input/kc_house_data.csv')
print('Quantidade de linhas: ', dados.shape[0])
print('Quantidade de colunas: ', dados.shape[1])
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
import tensorflow as tf

# cria as variaveis inicias que serão alteradas pelo treinamento
b0 = tf.Variable(b0_inicial)
b1 = tf.Variable(b1_inicial)
b1
# Cria-se o placeholders para armazenar os valores manipulados no treinamento
#   em caso de grande volume de dados é comum dividir a carga em pedaços (batch)
batch_size = 32 # irá processar os valores em pedaços de 32 amostras
metragens_ph = tf.placeholder(tf.float64, [batch_size, 1])
precos_ph = tf.placeholder(tf.float64, [batch_size, 1])
# Função de cálculo do erro
#  neste caso, será utilizado MSE para ajustar os parämetros
equacao_regressao = b0 + b1 * metragens_ph
funcao_de_erro = tf.losses.mean_squared_error(precos_ph, equacao_regressao)

# defini-se a taxa de aprendizagem (quanto é avancado na linha do Gradient Descent)
otimizador = tf.train.GradientDescentOptimizer(learning_rate = 0.001)

# defini-se a funcao de treinamento para minimizar o erro
treinamento = otimizador.minimize(funcao_de_erro)

# obrigatório inicializar as variaveis do TensorFlow
init = tf.global_variables_initializer()

# Execução do TREINAMENTO

with tf.Session() as sess:
    sess.run(init)
    for i in range(10000): # quantidade de épocas para treinar
        
        # como faremos o treinamento em pedaços (batch), é preciso criar as amostras nesses tamanhos
        indices = np.random.randint(len(metragens_escalonadas), size = batch_size)
        # alimenta-se os valores para os placeholders
        feed = { metragens_ph: metragens_escalonadas[indices], precos_ph: precos_escalonados[indices]}
        # executa o treinamento passando os valores. O treinamento irá altertar os valores de b0 e b1
        sess.run(treinamento, feed_dict= feed)

    # ao final do treinamento, obte-se os melhor valor encontrado para b0 e b1
    #   esses valores então são utilizados para prever um resultado para novas amostragens
    b0_final, b1_final = sess.run([b0, b1])
    print('b0 final: ', b0_final)
    print('b1 final: ', b1_final)
# Previsões considernado as metragens já conhecidas
# com os melhores valores de b0 e b1, podemos fazer o cálculo de previsão para novos valores
# y = b0 + b1 * x1
# porém, ao fazer a previsão, o resultado final estará com valor em escala. 
previsoes_escalonadas = b0_final + b1_final * metragens_escalonadas
plt.plot(metragens_escalonadas, precos_escalonados, 'o')
plt.plot(metragens_escalonadas, previsoes_escalonadas, color='red')
plt.title('Previsões em escala')
plt.xlabel('Metragens')
plt.ylabel('Preços')
# Prever o preço de um imóvel para um novo valor de metragem
# Neste caso é necessario:
#  transformar a metragem para a escala usada no treinamento
#  transformar a previsao para a escala inversa
nova_metragem = 1000
preco_previsto = b0_final + b1_final * scaler_metragem.transform([[nova_metragem]])
print('Preço escalonado: ', preco_previsto)
preco_previsto_final = scaler_precos.inverse_transform(preco_previsto)
print('Preço para metragem 1000: ', preco_previsto_final)
plt.plot(metragens, precos, 'o')
#plt.plot(metragens, (nova_metragem,preco_previsto_final), '*', color='red')
#point = plt.plot(NaN, NaN, 'r*');
#set(point,'Xdata', nova_metragem, 'YData', preco_previsto_final);
plt.plot(nova_metragem, preco_previsto_final, marker='o', markersize=5, color="red")
plt.title('Previsão para imovel de metragem 1000')
plt.xlabel('Metragens')
plt.ylabel('Preços')
# Calcular a precisao da previsao
from sklearn.metrics import mean_absolute_error, mean_squared_error

# desfaz a escala sobre os valores da previsao
previsoes = scaler_precos.inverse_transform(previsoes_escalonadas)

# Calcula a taxa média de erro para as previsões em relação aos valores conhecidos
print('Taxas médias de erro das previsões sobre os valores conhecidos')
mae = mean_absolute_error(precos, previsoes)
print('MAE: ', mae)
mse = mean_squared_error(precos, previsoes)
print('MSE: ', mse)

# Conclusão
print('\nCONCLUSÃO')
print('O algoritmo está errando em média R$ ', mae, ' para mais ou para menos no valor do imóvel.')

# dá para perceber que a solução encontrada não é tão boa, pois tem uma margem de erro ainda grande.
# além disto, não foi feito qualquer separação quanto a dados de treinamento e dados de teste.