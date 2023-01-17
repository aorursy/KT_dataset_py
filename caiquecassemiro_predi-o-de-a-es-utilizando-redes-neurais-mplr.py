#importando as bibliotecas necessárias para o estudo
from sklearn.neural_network import MLPRegressor
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import num2date, date2num
import matplotlib.dates
import datetime
#lendo a base
dataset = pd.read_csv('../input/base_petr.csv', sep=',')
#ordenando e 'limpando' o dataset
dataset = dataset[::-1]
dataset = dataset[dataset.columns[0:5]]
dataset = dataset.applymap(lambda x: x.replace(',', '.'))
print(dataset.head())
def inclui_proximo_fechamento(dataset):
    # inclui a coluna proximo_fechamento pegando o fechamento da (data_atual+1)
    proximo_fechamento_temp = dataset['fechamento'].values
    proximo_fechamento = [x for x in proximo_fechamento_temp[1:]]
    proximo_fechamento.append('6000')
    dataset['proximo_fechamento'] = proximo_fechamento

    return dataset

dados = inclui_proximo_fechamento(dataset)
print(dados.head(10))
#separando dados de treino e teste
Y = dados['proximo_fechamento']
Y = pd.to_numeric(Y)

X = dados.drop(['proximo_fechamento', 'data', 'abertura'], axis=1)
X = X.applymap(pd.to_numeric)

qtd = 5900
X_train = X[:qtd]
X_test = X[qtd:-1]
Y_train = Y[:qtd]
Y_test = Y[qtd:-1]
print('Tamanho Treino: {}\nTamanho Teste: {}'.format(len(X_train), len(X_test)))
print(X.head())
#instanciando objeto
mlp = MLPRegressor(solver='adam', alpha=0.0001, hidden_layer_sizes=(5,5), random_state=1, 
                   learning_rate='constant', learning_rate_init=0.01, max_iter=1000, 
                   activation='logistic', momentum=0.9, verbose=False, tol=0.00001)


#treinando a rede
mlp.fit(X_train, Y_train)

#predizendo os valores
saidas = mlp.predict(X_test)
saidas = [round(x, 2) for x in saidas]
taxa_de_acerto = round(mlp.score(X_test, Y_test), 2) * 100
print(f'{taxa_de_acerto}%')
print(f'{list(Y_test[:10].values)} (Valores reais)')
print(f'{saidas[:10]} (Valores obtidos através da rede neural)')
plt.plot(saidas, 'r', Y_test.values, 'b')
plt.show()
