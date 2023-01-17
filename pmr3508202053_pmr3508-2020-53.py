import numpy as np              # Arrays

import pandas as pd             # Dataframes

import matplotlib.pyplot as plt # Gráficos

import seaborn as sns           # Gráficos mais desenvolvidos

import sklearn                  # Algoritmos de Machine Learning
# Leitura do arquivo de extensão .csv para TREINO

DadosBase = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",

                        index_col=['Id'],

                        na_values="?")
DadosBase.head() # Exibição da tabela de dados de TREINO
print('Tamanho da Base de Dados de TREINO (linhas, colunas):', DadosBase.shape)
DadosBase.describe() # Exibição de dados relevantes para análise
DadosBase.info() # Verificação de quantos dados há por atributo
DadosBase.isnull().sum() # Verificação de que atributos têm dados faltantes e quantos são
DadosBase = DadosBase.drop(['education', 'relationship'], axis=1) # Eliminação de atributos redundantes
DadosBase.head() # Exibição da tabela de dados de TREINO atualizada
print('Tamanho da Base de Dados de TREINO (linhas, colunas):', DadosBase.shape)
DadosBase.describe() # Exibição de dados relevantes para análise
DadosBase.info() # Verificação de quantos dados há por atributo
DadosBase.isnull().sum() # Verificação de que atributos têm dados faltantes e quantos são
sns.catplot(x="age", y="income", palette="husl", kind="boxen", data=DadosBase, height=7, aspect=1)
sns.catplot(x="fnlwgt", y="income", palette="husl", kind="boxen", data=DadosBase, height=7, aspect=1)
sns.catplot(x="education.num", y="income", palette="husl", kind="boxen", data=DadosBase, height=7, aspect=1)
sns.catplot(x="capital.gain", y="income", palette="husl", kind="boxen", data=DadosBase, height=7, aspect=1)
sns.catplot(x="capital.loss", y="income", palette="husl", kind="boxen", data=DadosBase, height=7, aspect=1)
sns.catplot(x="hours.per.week", y="income", palette="husl", kind="boxen", data=DadosBase, height=7, aspect=1)
sns.catplot(x="workclass", hue="income", palette="husl", kind="count", data=DadosBase, height=7, aspect=1)

plt.xticks(rotation=45)
sns.catplot(x="marital.status", hue="income", palette="husl", kind="count", data=DadosBase, height=7, aspect=1)

plt.xticks(rotation=45)
sns.catplot(x="occupation", hue="income", palette="husl", kind="count", data=DadosBase, height=7, aspect=1)

plt.xticks(rotation=45)
sns.catplot(x="race", hue="income", palette="husl", kind="count", data=DadosBase, height=7, aspect=1)

plt.xticks(rotation=45)
sns.catplot(x="sex", hue="income", palette="husl", kind="count", data=DadosBase, height=7, aspect=1)

plt.xticks(rotation=45)
sns.catplot(x="native.country", hue="income", palette="husl", kind="count", data=DadosBase, height=8, aspect=1.75)

plt.xticks(rotation=75)
# Atributos com dados faltantes

print('Para o atributo "workclass":')

print()

print(DadosBase['workclass'].describe())      # Descoberta da moda do atributo

print ('-------------------------------------------')

print('Para o atributo "occupation":')

print()

print(DadosBase['occupation'].describe())     # Descoberta da moda do atributo

print ('-------------------------------------------')

print('Para o atributo "native.country":')

print()

print(DadosBase['native.country'].describe()) # Descoberta da moda do atributo
# Recorrência de cada dado para cada atributo

print ('Percentual dos dados de "workclass" sobre o total: \n')

print ((DadosBase['workclass'].value_counts()/DadosBase['workclass'].count())*100)

print ('--------------------------------------------------------')

print ('Percentual dos dados de "occupation" sobre o total: \n')

print ((DadosBase['occupation'].value_counts()/DadosBase['occupation'].count())*100)

print ('--------------------------------------------------------')

print ('Percentual dos dados de "native.country" sobre o total: \n')

print ((DadosBase['native.country'].value_counts()/DadosBase['native.country'].count())*100)
# Substituição dos dados faltantes pela moda dos atributos



# Para o atributo "workclass"

subworkclass = DadosBase['workclass'].describe().top

DadosBase['workclass'] = DadosBase['workclass'].fillna(subworkclass)



# Para o atributo "occupation"

suboccupation = DadosBase['occupation'].describe().top

DadosBase['occupation'] = DadosBase['occupation'].fillna(suboccupation)



# Para o atributo "native.country"

subnativecountry = DadosBase['native.country'].describe().top

DadosBase['native.country'] = DadosBase['native.country'].fillna(subnativecountry)
DadosBase.isnull().sum() # Verificação de que atributos têm dados faltantes e quantos são
# Atributos categóricos (não-numéricos)

atributosCAT = ['workclass', 'marital.status', 'occupation', 'race', 'sex', 'native.country']

print ('Atributos categóricos considerados:', atributosCAT)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()     # Instanciando o LabelEncoder

DadosBaseCopy = DadosBase.copy()

# Conversão dos atributos categóricos em numéricos

for column in atributosCAT:

    DadosBaseCopy[column] = le.fit_transform(DadosBaseCopy[column])

DadosBaseCopy.head()    # Exibição da tabela de dados de TREINO convertida
DadosBaseCAT = DadosBaseCopy.copy()

# Conversão do atributo "income" de categórico para numérico

DadosBaseCAT['income'] = le.fit_transform(DadosBaseCAT['income'])
# Matriz de correlação

correlation = DadosBaseCAT.corr()



# Máscara para o triângulo superior direito

mascara = np.triu(np.ones_like(correlation, dtype=bool))



# Configuração da figura em matplotlib

f, ax = plt.subplots(figsize=(11, 9))



# Plotagem do mapa de calor

sns.heatmap(correlation, mask=mascara, cmap="YlGnBu", vmax=0.5, vmin = -0.5, center=0, annot=True,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
# Atributos de maior correlação com "income"

HigherCorrelation = ['age', 'education.num', 'sex', 'capital.gain', 'hours.per.week', 'income']



# Separação de dados com maior relevância

DadosBaseHC = DadosBaseCAT[HigherCorrelation].copy()

DadosBaseHC.head() # Exibição da tabela de dados de TREINO separada
# Separação entre atributo de classificação Y e atributos de entrada X

DadosBaseHCY = DadosBaseHC.pop('income') # Atributo de interesse

DadosBaseHCX = DadosBaseHC               # Atributos para análise
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import RobustScaler

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline



# Remoção de outliers



# Dados numéricos normais

AtributosNUM = list(DadosBaseHCX.select_dtypes(include=[np.number]).columns.values)

AtributosNUM.remove('capital.gain')

# AtributosNUM.remove('capital.loss')

# Dados numéricos esparsos

# AtributosESPARSO = ['capital.gain', 'capital.loss']

AtributosESPARSO = ['capital.gain']



# Normalização de dados, para dados numéricos normais

PipelineNUM = Pipeline(steps = [('scaler', StandardScaler())])

# Normalização de dados, para dados numéricos esparsos

PipelineESPARSO = Pipeline(steps = [('scaler', RobustScaler())])



Preprocessador = ColumnTransformer(transformers = [

    ('num', PipelineNUM, AtributosNUM),

    ('spr', PipelineESPARSO, AtributosESPARSO),

])



DadosBaseHCX = Preprocessador.fit_transform(DadosBaseHCX)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score



# Classificador K-NN para os dados de TREINO tratados

knn = KNeighborsClassifier(n_neighbors=14) # Parâmetro inicial aleatório



# Método de validação cruzada

score = cross_val_score(knn, DadosBaseHCX, DadosBaseHCY, cv = 5, scoring="accuracy")

print("Acurácia com validação cruzada:", round(score.mean(),5))
# Otimização dos hiperparâmetros



# Importação do Random Search com validação cruzada

from sklearn.model_selection import RandomizedSearchCV



# Definição do Random Search CV

# O argumento n_iter atribui o número de configurações de hparams que serão testadas

random_search_cv = RandomizedSearchCV(estimator = KNeighborsClassifier(),

                              param_distributions = {'n_neighbors': range(1,50)}, # Testando comprimentos máximos de 1 a 50

                              scoring='accuracy',

                              cv = 5,

                              n_iter = 12)



# Otimização dos dados por GridSearch

random_search_cv.fit(DadosBaseHCX,DadosBaseHCY)



# O número de vizinhos utilizados no processamento melhora o desempenho

print('Melhor número de vizinhos: {}'.format(random_search_cv.best_params_['n_neighbors']))

print('Melhor acurácia: {}'.format(round(random_search_cv.best_score_,5)))
# Classificador K-NN com o melhor número de vizinhos

knn = KNeighborsClassifier(n_neighbors=30)

knn.fit(DadosBaseHCX, DadosBaseHCY)
# Leitura do arquivo de extensão .csv para TESTE

DadosBaseTeste = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv",

                             index_col=['Id'],

                             na_values="?")
print('Tamanho da Base de Dados de TESTE (linhas, colunas):', DadosBaseTeste.shape)
DadosBaseTeste.isnull().sum() # Verificação de que atributos têm dados faltantes e quantos são
DadosBaseTeste = DadosBaseTeste.drop(['education', 'relationship'], axis=1) # Eliminação de atributos redundantes
# Atributos com dados faltantes

print('Para o atributo "workclass":')

print()

print(DadosBaseTeste['workclass'].describe())      # Descoberta da moda do atributo

print ('-------------------------------------------')

print('Para o atributo "occupation":')

print()

print(DadosBaseTeste['occupation'].describe())     # Descoberta da moda do atributo

print ('-------------------------------------------')

print('Para o atributo "native.country":')

print()

print(DadosBaseTeste['native.country'].describe()) # Descoberta da moda do atributo
# Recorrência de cada dado para cada atributo

print ('Percentual dos dados de "workclass" sobre o total: \n')

print ((DadosBaseTeste['workclass'].value_counts()/DadosBaseTeste['workclass'].count())*100)

print ('--------------------------------------------------------')

print ('Percentual dos dados de "occupation" sobre o total: \n')

print ((DadosBaseTeste['occupation'].value_counts()/DadosBaseTeste['occupation'].count())*100)

print ('--------------------------------------------------------')

print ('Percentual dos dados de "native.country" sobre o total: \n')

print ((DadosBaseTeste['native.country'].value_counts()/DadosBaseTeste['native.country'].count())*100)
# Substituição dos dados faltantes pela moda dos atributos



# Para o atributo "workclass"

subworkclass = DadosBaseTeste['workclass'].describe().top

DadosBaseTeste['workclass'] = DadosBaseTeste['workclass'].fillna(subworkclass)



# Para o atributo "occupation"

suboccupation = DadosBaseTeste['occupation'].describe().top

DadosBaseTeste['occupation'] = DadosBaseTeste['occupation'].fillna(suboccupation)



# Para o atributo "native.country"

subnativecountry = DadosBaseTeste['native.country'].describe().top

DadosBaseTeste['native.country'] = DadosBaseTeste['native.country'].fillna(subnativecountry)
DadosBaseTeste.isnull().sum() # Verificação de que atributos têm dados faltantes e quantos são
DadosBaseTesteCopy = DadosBaseTeste.copy()

# Conversão dos atributos categóricos em numéricos

for column in atributosCAT:

    DadosBaseTesteCopy[column] = le.fit_transform(DadosBaseTesteCopy[column])

DadosBaseTesteCopy.head()    # Exibição da tabela de dados de TESTE convertida
# Atributos de maior correlação com "income"

HigherCorrelationTeste = ['age', 'education.num', 'sex', 'capital.gain', 'hours.per.week']



# Separação de dados com maior relevância

DadosBaseHCT = DadosBaseTesteCopy[HigherCorrelationTeste].copy()

DadosBaseHCT.head() # Exibição da tabela de dados de TESTE separada
# Predição do atributo de classificação

DadosBaseTesteY = knn.predict(DadosBaseHCT)
# Conversão dos atributos numéricos em categóricos

DadosBaseFinal = []

for i in range(len(DadosBaseTesteY)):

    if (DadosBaseTesteY[i] == 0):

        DadosBaseFinal.append('<=50K')

    else:

        DadosBaseFinal.append('>50K')

        

# Consolidação do Array em DataFrame

DataFrameFinal = pd.DataFrame({'income': DadosBaseFinal})
# Exportação do DataFrame como arquivo de extensão .csv

DataFrameFinal.to_csv("submission.csv", index = True, index_label = 'Id')