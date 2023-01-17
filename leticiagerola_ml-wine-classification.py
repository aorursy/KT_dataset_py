import pandas as pd

arquivo = pd.read_csv('../input/wine-dataset/wine_dataset.csv')
arquivo.head()
# atribuindo valor numérico à última coluna 'red x white'

arquivo['style'] = arquivo['style'].replace('red', 0)
arquivo['style'] = arquivo['style'].replace('white', 1)
# separando as variáveis

y = arquivo['style'] # armazena coluna 'style' na variável y (variável alvo)
x = arquivo.drop('style', axis = 1) # armazena todo o restante na variável x (variáveis preditoras)
# conjunto de dados para treino e teste

from sklearn.model_selection import train_test_split
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = 0.3) # test_size: quantos % p/ treino e p/ teste
# criação do modelo

from sklearn.ensemble import ExtraTreesClassifier
modelo = ExtraTreesClassifier() # algoritmo que cria várias árvores de decisão
modelo.fit(x_treino, y_treino)
resultado = modelo.score(x_teste, y_teste)
print(f'Acurácia: {resultado}')
y_teste[205:210] # filtro de 5 amostras aleatóreas que o modelo não teve contato
x_teste[205:210] # filtro de 5 amostras aleatóreas que o modelo não teve contato
previsoes = modelo.predict(x_teste[205:210]) # criação de variável para checar se o modelo treinado acerta
print(previsoes) # modelo previu corretamente