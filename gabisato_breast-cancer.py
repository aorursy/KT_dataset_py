import pandas

# le o arquivo csv

dataset = pandas.read_csv("../input/data.csv")



dataset.head()
import seaborn

import matplotlib.pyplot as matplot



graph = seaborn.FacetGrid(dataset, col='diagnosis')

graph.map(matplot.hist, 'id', bins=10)
graph = seaborn.FacetGrid(dataset, col='diagnosis')

graph.map(matplot.hist, 'radius_mean', bins=20)

# radius mean: média das distâncias do centro para os pontos no perímetro

# parece que quanto maior o radius mean maior a possibilidade de ser maligno
graph = seaborn.FacetGrid(dataset, col='diagnosis')

graph.map(matplot.hist, 'texture_mean', bins=20)

# texture_mean: desvio padrão dos valores da escala de cinza

# entre os valores 10 e 20 parece ter maior probabilidade de ser benigno

# entre os valores de 20 a 30 parece ter maior probabilidade de ser maligno
graph = seaborn.FacetGrid(dataset, col='diagnosis')

graph.map(matplot.hist, 'perimeter_mean', bins=20)

# perimeter_mean: tamanho médio do tumor do núcleo

# a probabilidade de ser benigno parece estar entre o tamanho 50 a 100

# já a de ser maligno estar entre 75 adiante, tendo pico entre 100 e 125
graph = seaborn.FacetGrid(dataset, col='diagnosis')

graph.map(matplot.hist, 'area_mean', bins=20)

# para médias da área maiores que 500 parece ter maior probabilidade de ser maligno

# para menores de 500 a probabilidade é maior de ser benigno
graph = seaborn.FacetGrid(dataset, col='diagnosis')

graph.map(matplot.hist, 'smoothness_mean', bins=20)

# smoothness_mean: média da variação local em comprimentos de raio

# não vejo uam interpretação sobre esses dados em especifico
graph = seaborn.FacetGrid(dataset, col='diagnosis')

graph.map(matplot.hist, 'compactness_mean', bins=20)

# média do perímetro ^ 2 / área - 1,0

# para valores menores de 0.1 a probabilidade de ser benigno é maior

# e para maiores de 0.1 a probabilidade para ser maior para maligno
graph = seaborn.FacetGrid(dataset, col='diagnosis')

graph.map(matplot.hist, 'concavity_mean', bins=20)

# média de severidade das porções côncavas do contorno

# para valores menores de 0.1 tem maior probabilidade de ser benicno

# e para dados acima, com pico entre 0.1 e 0.2 tem maior probabilidade de ser maligno
graph = seaborn.FacetGrid(dataset, col='diagnosis')

graph.map(matplot.hist, 'concave points_mean', bins=20)

# concave points_mean: média para o número de porções côncavas do contorno

# menores de 0.005 parecem ter maior probabilidade de serem benignos 

# para maiores de 0.05 parecem ter maior probabilidade de ser maligno
graph = seaborn.FacetGrid(dataset, col='diagnosis')

graph.map(matplot.hist, 'symmetry_mean', bins=20)

# não entendi a relação da média de simetria
graph = seaborn.FacetGrid(dataset, col='diagnosis')

graph.map(matplot.hist, 'fractal_dimension_mean', bins=20)

# fractal_dimension_mean: média para "aproximação costeira" - 1
graph = seaborn.FacetGrid(dataset, col='diagnosis')

graph.map(matplot.hist, 'radius_se', bins=15)

# radius_se: erro padrão para a média das distâncias do centro para os pontos no perímetro
graph = seaborn.FacetGrid(dataset, col='diagnosis')

graph.map(matplot.hist, 'texture_se', bins=20)

# texture_se: erro padrão para desvio padrão dos valores da escala de cinza
graph = seaborn.FacetGrid(dataset, col='diagnosis')

graph.map(matplot.hist, 'perimeter_se', bins=20)
graph = seaborn.FacetGrid(dataset, col='diagnosis')

graph.map(matplot.hist, 'area_se', bins=20)
graph = seaborn.FacetGrid(dataset, col='diagnosis')

graph.map(matplot.hist, 'smoothness_se', bins=20)

# erro padrão para variação local em comprimentos de raio
graph = seaborn.FacetGrid(dataset, col='diagnosis')

graph.map(matplot.hist, 'compactness_se', bins=20)

# erro padrão para perímetro ^ 2 / area - 1.0
graph = seaborn.FacetGrid(dataset, col='diagnosis')

graph.map(matplot.hist, 'concavity_se', bins=20)

# erro padrão para a gravidade das porções côncavas do contorno
graph = seaborn.FacetGrid(dataset, col='diagnosis')

graph.map(matplot.hist, 'concave points_se', bins=20)

# erro padrão para o número de porções côncavas do contorno
graph = seaborn.FacetGrid(dataset, col='diagnosis')

graph.map(matplot.hist, 'symmetry_se', bins=20)
graph = seaborn.FacetGrid(dataset, col='diagnosis')

graph.map(matplot.hist, 'fractal_dimension_se', bins=20)
graph = seaborn.FacetGrid(dataset, col='diagnosis')

graph.map(matplot.hist, 'radius_worst', bins=20)

# "pior" ou maior valor médio para a média das distâncias do centro para os pontos no perímetro

# benigno entre 10 a 20

# maligno entre 15 a acima
graph = seaborn.FacetGrid(dataset, col='diagnosis')

graph.map(matplot.hist, 'texture_worst', bins=20)

# "pior" ou maior valor médio para desvio padrão dos valores da escala de cinza
graph = seaborn.FacetGrid(dataset, col='diagnosis')

graph.map(matplot.hist, 'perimeter_worst', bins=20)

# valores menor que 100 parecem ter maior probabilidade de serem benignos

# para maiores que 100 parecem ser malignos
graph = seaborn.FacetGrid(dataset, col='diagnosis')

graph.map(matplot.hist, 'area_worst', bins=20)

# valores menores que 1000 há maior probabilidade de serem benignos

# já para maiores há maior probabilidade de serem malignos
graph = seaborn.FacetGrid(dataset, col='diagnosis')

graph.map(matplot.hist, 'smoothness_worst', bins=20)

# "pior" ou maior valor médio para variação local em comprimentos de raio
graph = seaborn.FacetGrid(dataset, col='diagnosis')

graph.map(matplot.hist, 'compactness_worst', bins=20)

# "pior" ou maior valor médio para perímetro ^ 2 / area - 1,0
graph = seaborn.FacetGrid(dataset, col='diagnosis')

graph.map(matplot.hist, 'concavity_worst', bins=20)
graph = seaborn.FacetGrid(dataset, col='diagnosis')

graph.map(matplot.hist, 'concave points_worst', bins=20)
graph = seaborn.FacetGrid(dataset, col='diagnosis')

graph.map(matplot.hist, 'symmetry_worst', bins=20)
graph = seaborn.FacetGrid(dataset, col='diagnosis')

graph.map(matplot.hist, 'fractal_dimension_worst', bins=20)
# descreve os dados numericos de cada coluna

dataset.describe()
# descreve os dados categoricos

dataset.describe(include = ('O'))
# removendo as colunas de dados que nao serao usadas

dataset.drop(['id', 'smoothness_mean', 'symmetry_mean', 'fractal_dimension_mean', 'smoothness_se',

              'symmetry_se', 'fractal_dimension_se', 'smoothness_worst', 'symmetry_worst', 

              'fractal_dimension_worst','Unnamed: 32'], axis=1, inplace=True)
dataset.head()
from sklearn.preprocessing import LabelEncoder 

# transforma os dados da coluna diagnosis em dados numericos

labelencoder = LabelEncoder()

dataset['diagnosis'] = labelencoder.fit_transform(dataset['diagnosis'])

dataset.head()
# separa os dados de classe e de atributos

classe = dataset['diagnosis']

atributos = dataset.drop('diagnosis', axis=1)
classe.head()
atributos.head()
from sklearn.model_selection import train_test_split

# separa dados de treinamento e de teste

atributos_train, atributos_test, classe_train, classe_test = train_test_split(atributos, classe, test_size = 0.25)



atributos_train.describe()
from sklearn.tree import DecisionTreeClassifier

# cria a arvore

tree = DecisionTreeClassifier(criterion='gini', splitter='random', max_depth=3, random_state=0)

#cria o modelo

model = tree.fit(atributos_train, classe_train)
from sklearn.metrics import accuracy_score

# tenta predicao dos dados de teste 

classe_pred = model.predict(atributos_test)

classe_pred
# probabilidade de acerto

acc = accuracy_score(classe_test, classe_pred)

print("A probabilidade de acerto é: ", format(acc))