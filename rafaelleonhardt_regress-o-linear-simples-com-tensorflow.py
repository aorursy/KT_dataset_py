import numpy as np
import matplotlib.pyplot as plt

# Valores conhecidos
idades = np.array([[18], [23], [28], [33], [38], [43], [48], [53], [58], [63]])
custo_plano_saude = np.array([[871], [1132], [1042], [1356], [1488], [1638], [1569], [1754], [1866], [1900]])

%matplotlib inline
plt.scatter(idades, custo_plano_saude)
plt.title('Custo do Plano de Saúde X Idade')
plt.xlabel('Idade')
plt.ylabel('Custo')
# Ajustar a escala dos valores
# É importante que seja normalizado a escala de valores que estão sendo trabalhados.

from sklearn.preprocessing import StandardScaler

scaler_idades = StandardScaler()
idades = scaler_idades.fit_transform(idades)

scaler_custo = StandardScaler()
custo_plano_saude = scaler_custo.fit_transform(custo_plano_saude)

# Exibe o gráfico com valores em escala diferentes, mas com posições iguais ao gráfico anterior
%matplotlib inline
plt.scatter(idades, custo_plano_saude)
plt.title('Custo do Plano de Saúde X Idade')
plt.xlabel('Idade (em escala)')
plt.ylabel('Custo (em escala)')

# Calcula-se os valores iniciais para b0 e b1
#   com a Gradient Descendent esses valores são alterados pelo treinamento até se chegar ao valor ideal

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

# Função de cálculo do erro
#  neste caso, será utilizado MSE para ajustar os parämetros
funcao_mse = b0 + b1 * idades
funcao_de_erro = tf.losses.mean_squared_error(custo_plano_saude, funcao_mse)

# defini-se a taxa de aprendizagem (quanto é avancado na linha do Gradient Descent)
otimizador = tf.train.GradientDescentOptimizer(learning_rate = 0.001)

# defini-se a funcao de treinamento para minimizar o erro
treinamento = otimizador.minimize(funcao_de_erro)

# obrigatório inicializar as variaveis do TensorFlow
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000): # quantidade de épocas para treinar
        sess.run(treinamento)

    # ao final do treinamento, obte-se os melhor valor encontrado para b0 e b1
    b0_final, b1_final = sess.run([b0, b1])

# com os melhores valores de b0 e b1, podemos fazer o cálculo de previsão para novos valores
# y = b0 + b1 * x1
# porém, ao fazer a previsão, o resultado final estará com valor em escala. 
previsoes = b0_final + b1_final * idades
plt.plot(idades, custo_plano_saude, 'o')
plt.plot(idades, previsoes, '*', color='red')
plt.title('Previsões em escala')
plt.xlabel('Idade')
plt.ylabel('Custo')
# Prever o valor de custo para uma nova idade (40)
# Neste caso é necessario:
#  transformar a idade para a escala usada no treinamento
#  transformar a previsao para a escala inversa
idade = 40
custo_previsto = b0_final + b1_final * scaler_idades.transform([[idade]])
print('Custo escalonado: ', custo_previsto)
custo_previsto_final = scaler_custo.inverse_transform(custo_previsto)
print('Custo para 40 anos: ', custo_previsto_final)
# Prever o valor de custo de todas as idades da matriz

# Desfaz o escalonamento dos valores de entrada e previsoes
idades = scaler_idades.inverse_transform(idades)
custo_plano_saude = scaler_custo.inverse_transform(custo_plano_saude)

previsoes = scaler_custo.inverse_transform(previsoes)
plt.plot(idades, custo_plano_saude, 'o')
plt.plot(idades, previsoes, '*', color='red')
plt.title('Previsões finais')
plt.xlabel('Idade')
plt.ylabel('Custo')
# Calcular a precisao da previsao
from sklearn.metrics import mean_absolute_error, mean_squared_error

# informacoes utilizadas
print('Idades conhecidas: ', idades)
print('Valores conhecidos: ', custo_plano_saude)
print('Valores previstos: ', previsoes)
print('\n')

# Calcula a taxa média de erro para as previsões em relação aos valores conhecidos
print('Taxas médias de erro das previsões sobre os valores conhecidos')
mae = mean_absolute_error(custo_plano_saude, previsoes)
print('MAE: ', mae)
mse = mean_squared_error(custo_plano_saude, previsoes)
print('MSE: ', mse)

# Conclusão
print('\nCONSCLUSÃO')
print('O algoritmo está errando em média R$ ', mae, ' para mais ou para menos.')