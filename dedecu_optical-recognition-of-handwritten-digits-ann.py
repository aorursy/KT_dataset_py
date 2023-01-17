# DESCOBERTA DE CONHECIMENTO EM BANCOS DE DADOS E MINERAÇÃO DE DADOS

# Este é um exemplo bem simples de uma REDE NEURAL para reconhecimento de dígitos.

# Problema de classificação

# DATASET: 64 campos que formam a imagem de 1 dígito, ou seja, um número de 0 a 9.

# O objetivo será classificar cada imagem em um das 10 classes: 0 até 9.



from sklearn import datasets
d = datasets.load_digits()
print(d.keys())
print(d['DESCR']) # Informações sobre o DATASET

print(d['data'][1]) # Primeira imagem. Exemplo de algumas imagens do DATASET.
import matplotlib.pyplot as plot #Plotando a imagem do DATASET da posição acima. 
plot.figure(1, figsize=(3, 3))

plot.imshow(d.images[1], cmap=plot.cm.gray_r, interpolation='nearest')

plot.show()
import math
dados = d['data']

print(len(dados)) # Número de instâncias
idx_treinamento = math.floor(len(dados)*0.7)
print (idx_treinamento) # Separando 70% dos dados para treinamento

cjtreinamento = dados [:idx_treinamento] # Slice do array da posição 0 até 1257
idx_teste = len(dados) - idx_treinamento # Separando 30% para testes
print (idx_teste) 
cjteste = dados [idx_treinamento:]  #Slice do array da posição 1257 até final do array
print (idx_treinamento + idx_teste) # total
from sklearn.neural_network import MLPClassifier
# usei a função de ativação logistic (sigmoid function) e taxa de aprendizado = 0.1



c = MLPClassifier(activation='logistic', learning_rate_init=0.1, verbose=True)
print(cjtreinamento)
print (d['target'])

len(d['target'])
rotulos = d['target']

rotulos_treinamento = rotulos[:idx_treinamento] # Slice dos rotulos da posição 0 até 1257

len(rotulos_treinamento)
# Treinando ...

c.fit(cjtreinamento,rotulos_treinamento)
# Com a rede treinada, hora de testar!!

resultado = c.predict(cjteste)
rotulos_teste = rotulos[idx_treinamento:] # Slice dos rotulos da posição 1257 até o final do array



acuracia = c.score(cjteste, rotulos_teste) # Medindo a perfomance: a acurácia foi:

print ('Acurária',acuracia)
# Alterando alguns parâmetros para melhorar a acurácia. Apenas alterei o default do ajustes de pessos para 

# sgd (stochastic gradient descent) e alterei a taxa de aprendizado



c = MLPClassifier(solver='sgd',activation='logistic', learning_rate_init=0.12, verbose=True)
c.fit(cjtreinamento,rotulos_treinamento)

resultado = c.predict(cjteste)

acuracia = c.score(cjteste, rotulos_teste)

print ('Acurária',acuracia)