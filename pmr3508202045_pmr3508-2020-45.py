import numpy as np 

import pandas as pd #biblioteca utilizada para tratar dos dados

import matplotlib.pyplot as plt #biblioteca utilizada para plotagem dos gráficos

import seaborn as sns #biblioteca utilizada para plotagem dos gráficos

import sklearn #biblioteca com algoritmos de aprendizado de máquina
Dados_treino = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv", index_col=['Id'], na_values="?") 
Dados_treino.head()
Dados_treino = Dados_treino.drop(['education', 'relationship'], axis=1) #retirando as colunas redundantes
Dados_treino.head()
Dados_treino.describe() #analisando os dados da tabela
#plotando os gráficos para análise dos dados numéricos

sns.catplot(x="income", y="age", kind="violin", data=Dados_treino, palette="Set2") #plotando o gráfico income x age
#plotando os gráficos para análise dos dados numéricos

sns.catplot(x="income", y="age", kind="boxen", data=Dados_treino, palette="Set2") #plotando o gráfico income x age
sns.catplot(x="income", y="fnlwgt", kind="violin", data=Dados_treino, palette="Set2") #plotando o gráfico income x fnlwgt
sns.catplot(x="income", y="fnlwgt", kind="boxen", data=Dados_treino, palette="Set2") #plotando o gráfico income x fnlwgt
sns.catplot(x="income", y="education.num", kind="violin", data=Dados_treino, palette="Set2") #plotando o gráfico income x education.num
sns.catplot(x="income", y="education.num", kind="boxen", data=Dados_treino, palette="Set2") #plotando o gráfico income x education.num
sns.catplot(x="income", y="capital.gain", kind="violin", data=Dados_treino, palette="Set2") #plotando o gráfico income x capital.gain
sns.catplot(x="income", y="capital.gain", kind="boxen", data=Dados_treino, palette="Set2") #plotando o gráfico income x capital.gain
sns.catplot(x="income", y="capital.loss", kind="violin", data=Dados_treino, palette="Set2") #plotando o gráfico income x capital.loss
sns.catplot(x="income", y="capital.loss", kind="boxen", data=Dados_treino, palette="Set2") #plotando o gráfico income x capital.loss
sns.catplot(x="income", y="hours.per.week", kind="violin", data=Dados_treino, palette="Set2") #plotando o gráfico income x hours.per.week
sns.catplot(x="income", y="hours.per.week", kind="boxen", data=Dados_treino, palette="Set2") #plotando o gráfico income x hours.per.week
#plotando os gráficos para análise dos dados de categoria (atributos não numéricos)

sns.catplot(x="income", hue="workclass", kind="count", data=Dados_treino, palette="Set2") #plotando o gráfico income x workclass
sns.catplot(x="income", hue="marital.status", kind="count", data=Dados_treino, palette="Set2") #plotando o gráfico income x marital.status
sns.catplot(x="income", hue="occupation", kind="count", data=Dados_treino, palette="Set2") #plotando o gráfico income x occupation
sns.catplot(x="income", hue="race", kind="count", data=Dados_treino, palette="Set2") #plotando o gráfico income x race
sns.catplot(x="income", hue="sex", kind="count", data=Dados_treino, palette="Set2") #plotando o gráfico income x sex
sns.catplot(x="native.country", hue="income", kind="count", data=Dados_treino, palette="Set2", aspect=2, height=8) #plotando o gráfico income x native.country

plt.xticks(rotation=80) #rotação da legenda
#Tratando dos dados faltantes

Dados_treino.info() # Verificação por categoria de quantos dados tem
Dados_treino.isnull().sum() # Verificação de quem tem e quantos são dados faltantes
Dados_treino['workclass'].describe() #top é a moda e freq a frequência que a moda aparece
Dados_treino['occupation'].describe() #top é a moda e freq a frequência que a moda aparece
Dados_treino['native.country'].describe() #top é a moda e freq a frequência que a moda aparece
#Substituindo os valores NaN pelos valores da moda de cada categoria (estão apresntados nas tabelas acima)

Dados_treino['workclass'] = Dados_treino['workclass'].fillna(Dados_treino['workclass'].describe().top)

Dados_treino['occupation'] = Dados_treino['occupation'].fillna(Dados_treino['occupation'].describe().top)

Dados_treino['native.country'] = Dados_treino['native.country'].fillna(Dados_treino['native.country'].describe().top)
Dados_treino.isnull().sum() #analisando se sobraram dados faltantes
dados_de_categoria = ['workclass', 'marital.status', 'occupation', 'race', 'sex', 'native.country']
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder() #instanciando o LabelEncoder
#Transformando as variáveis de categorias em numéricas

for column in dados_de_categoria:

    Dados_treino[column] = le.fit_transform(Dados_treino[column])

#Transformando a variável income em 0 ou 1    

Dados_treino['income'] = le.fit_transform(Dados_treino['income'])
Dados_treino.head() #verificação das variáveis se são somente numéricas
Dados_treino_correlacao = Dados_treino.corr() # Cálculo da matriz de correlação
mascara = np.triu(np.ones_like(Dados_treino_correlacao, dtype=bool)) # Geração de uma máscara para o triângulo superior
f, ax = plt.subplots(figsize=(10, 10)) #plotando o gráfico com o matplot

sns.heatmap(Dados_treino_correlacao, mask=mascara, cmap="Greens", vmax=0.5, vmin = -0.5, center=0, annot=True, square=True, linewidths=.5, cbar_kws={"shrink": .5})
atributos_relevantes = ["age", "education.num", "sex", "capital.gain", "capital.loss", "hours.per.week", "income"]

Dados_treino_atributos = Dados_treino[atributos_relevantes].copy() #faz uma tabela com os atributos analisados como relevantes
Dados_treino_atributos.head()
Dados_treino_eixo_y = Dados_treino_atributos.pop('income') #separando a coluna 'income' no eixo y (atributo de interesse)

Dados_treino_eixo_x = Dados_treino_atributos
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import RobustScaler

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline
variaveis_numericas = list(Dados_treino_eixo_x.select_dtypes(include=[np.number]).columns.values) #separando as variáveis numéricas
variaveis_numericas.remove('capital.gain')

variaveis_numericas.remove('capital.loss')

variaveis_esparsas = ['capital.gain', 'capital.loss'] #separando as variáveis esparsas
numericas_pipeline = Pipeline(steps = [('scaler', StandardScaler())]) #utilizando o método StandardScaler para os dados numéricos normais
esparsas_pipeline = Pipeline(steps = [('scaler', RobustScaler())]) #utilizando o método RobustScaler para os dados numéricos esparsos
#preprocessamento

preprocessador = ColumnTransformer(transformers = [('num', numericas_pipeline, variaveis_numericas),('spr', esparsas_pipeline, variaveis_esparsas),])
Dados_treino_eixo_x  = preprocessador.fit_transform(Dados_treino_eixo_x)
#Treinando o classificador

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score
knn = KNeighborsClassifier(n_neighbors=15) #15 vizinhos 
score = cross_val_score(knn, Dados_treino_eixo_x, Dados_treino_eixo_y, cv = 5, scoring="accuracy")

print("Acurácia com validação cruzada:", score.mean())
#Otimização dos hiperparâmetros

from sklearn.model_selection import RandomizedSearchCV #importando o Random Search com validação cruzada
#Definindo o Random Search CV. Fornecendo o argumento n_iter (quantas configurações de hiperparametros vai testar):

random_search_cv = RandomizedSearchCV(estimator = KNeighborsClassifier(), param_distributions = {'n_neighbors': range(1,50)}, scoring='accuracy', cv = 5, n_iter = 12)
random_search_cv.fit(Dados_treino_eixo_x, Dados_treino_eixo_y) #otimizando por GridSearch
print('Melhor número de vizinhos: {}'.format(random_search_cv.best_params_['n_neighbors']))

print('Melhor acurácia: {}'.format(round(random_search_cv.best_score_,5)))
#Criando um classificador KNN com o valor achado com a otimização dos hiperparâmetros

knn = KNeighborsClassifier(n_neighbors=34)
knn.fit(Dados_treino_eixo_x, Dados_treino_eixo_y)
#Testando o classificador proposto

Dados_teste= pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv", index_col=['Id'], na_values="?")
Dados_teste.head()
Dados_teste.isnull().sum() # Verificação de quem tem e quantos são dados faltantes
Dados_teste['workclass'].describe() #top é a moda e freq a frequência que a moda aparece
Dados_teste['occupation'].describe() #top é a moda e freq a frequência que a moda aparece
Dados_teste['native.country'].describe() #top é a moda e freq a frequência que a moda aparece
#Substituindo os valores NaN pelos valores da moda de cada categoria (estão apresntados nas tabelas acima)

Dados_teste['workclass'] = Dados_teste['workclass'].fillna(Dados_teste['workclass'].describe().top)

Dados_teste['occupation'] = Dados_teste['occupation'].fillna(Dados_teste['occupation'].describe().top)

Dados_teste['native.country'] = Dados_teste['native.country'].fillna(Dados_teste['native.country'].describe().top)
Dados_treino.isnull().sum() #analisando se sobraram dados faltantes
le = LabelEncoder() #instanciando o LabelEncoder
#Transformando as variáveis de categorias em numéricas

for column in dados_de_categoria:

    Dados_teste[column] = le.fit_transform(Dados_teste[column])
atributos_relevantes = ["age", "education.num", "sex", "capital.gain", "capital.loss", "hours.per.week"]

Dados_teste_atributos = Dados_teste[atributos_relevantes].copy() #faz uma tabela com os atributos analisados como relevantes
Dados_teste_atributos.head()
Dados_treino_eixo_y = knn.predict(Dados_teste_atributos) #aplicação do classificador KNN
Dados_finais = []



for i in range(len(Dados_treino_eixo_y)):

    if (Dados_treino_eixo_y[i] == 0):

        Dados_finais.append('<=50K')

    else:

        Dados_finais.append('>50K')

        

#transformação do array em DataFrame

Dataframe_final = pd.DataFrame({'income': Dados_finais})
Dataframe_final.to_csv("submission.csv", index = True, index_label = 'Id')