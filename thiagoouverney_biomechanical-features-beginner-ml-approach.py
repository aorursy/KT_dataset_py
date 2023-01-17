# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#pipeline
import pandas as pd
import random as rd, matplotlib.pyplot as plt, math , graphviz , seaborn as sns

#Model_Selection
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold

#models
from sklearn.linear_model import LogisticRegression
#
from sklearn.neighbors import KNeighborsClassifier
#
from sklearn.tree import DecisionTreeClassifier, export_graphviz

#Hyperparamenter
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

#Metrics
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

#preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler,  MaxAbsScaler, normalize
data = pd.read_csv("/kaggle/input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")
data.head()
data.describe(percentiles= [0,.25,.5,.75])
df = data.drop("pelvic_tilt numeric",axis=1).copy()
print("""
We have {0} RX patients and we are going to use {1} features.
""".format(df.drop("class",axis=1).shape[0],df.drop("class",axis=1).shape[1]))
df['class'].value_counts()
numerical = df.select_dtypes(exclude= 'object')
numerical_corr = numerical.corr()
f,ax=plt.subplots(figsize=(10,8))
sns.heatmap(numerical_corr)
plt.title("Numerical Features Correlation", weight='bold', fontsize=18)
plt.xticks(weight='bold')
plt.yticks(weight='bold', color='dodgerblue')

plt.show()
numerical_corr.head()
fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2,2,figsize=(16,12))

ax1.hist(x=df["degree_spondylolisthesis"], color='#0504aa',alpha=0.7)
ax1.set_title('Histogram Spondylolisthesis',fontdict={'fontsize': 14,'fontweight': 'bold'})
ax1.set( ylabel='Frequency') 

ax2.scatter(x=data["degree_spondylolisthesis"], y = data["pelvic_incidence"], color = 'blue', alpha = 0.5)
ax2.set_title('Pelvic Incidence vs Spondylolisthesis',fontdict={'fontsize': 14,'fontweight': 'bold'})
ax2.set( ylabel='Pelvic Incidence') 

ax3.scatter(x=data["degree_spondylolisthesis"], y = data["sacral_slope"], color = 'blue', alpha = 0.5)
ax3.set_title('Sacral Slope vs Spondylolisthesis',fontdict={'fontsize': 14,'fontweight': 'bold'})
ax3.set(xlabel='Degree of Spondylolisthesis', ylabel='Sacral Slope') 

ax4.scatter(x=data["degree_spondylolisthesis"], y = data["pelvic_radius"], color = 'blue', alpha = 0.5)
ax4.set_title('Pelvic Radius vs Spondylolisthesis',fontdict={'fontsize': 14,'fontweight': 'bold'})
ax4.set(xlabel='Degree of Spondylolisthesis', ylabel='Pelvic Radius') 

plt.show()
df.isnull().sum()
fig = plt.figure(figsize=(20,8))

ax1 = plt.subplot2grid((1,2),(0,0))
plt.hist(x=df["degree_spondylolisthesis"], color='#0504aa',alpha=0.7)
plt.title('Histogram Spondylolisthesis')
plt.axvline(x=200,color='r',linestyle = '-')

ax1 = plt.subplot2grid((1,2),(0,1))
filter_spond = df["degree_spondylolisthesis"] < 200
plt.hist(x=df[filter_spond]["degree_spondylolisthesis"], color='#0504aa',alpha=0.7)
plt.title('Histogram Spondylolisthesis Filtred')
plt.show()
plt.hist(df['pelvic_radius'])
plt.show()
df = df[filter_spond].copy()
x=df.drop("class",axis=1).copy()
target=df['class']
y= pd.get_dummies(target)['Abnormal']
valores_C = np.array([0.01,0.1,0.5,1,2,3,5,10,20,50,100])
regularizacao = ['l2']
valores_grid= {'C':valores_C, 'penalty': regularizacao}

kfold = StratifiedKFold(n_splits=5)

modelo = LogisticRegression()

grid_regressao_logistica = GridSearchCV(modelo,param_grid = valores_grid, cv=  5)
grid_regressao_logistica.fit(x,y)
print("""
Melhor acurácia: {0};
Parâmetro C: {1};
Regularização: {2};
""".format(grid_regressao_logistica.best_score_,
           grid_regressao_logistica.best_estimator_.C,
           grid_regressao_logistica.best_estimator_.penalty))
x_treino,x_teste, y_treino, y_teste = train_test_split(x,y,test_size= 0.30,random_state=14)

modelo = LogisticRegression(C=0.1, penalty = 'l2')
modelo.fit(x_treino,y_treino)
predicao = modelo.predict(x_teste)
matrix = confusion_matrix(y_teste,predicao)
print(matrix)
#Normaliando os dados para trabalhar com distância

normalizadores = [MinMaxScaler(feature_range= (0,1)),
                  StandardScaler(),
                  MaxAbsScaler()]

for normalizador in normalizadores:
  X_norm = normalizador.fit_transform(x)


  #Hyperparameter
  valores_k = np.array([3,5,7,9,11])
  calculo_distancia = ['minkowski', 'chebyshev']
  valores_p = np.array([1,2,3,4])
  valores_grid = {'n_neighbors':valores_k, 'metric':calculo_distancia, 'p':valores_p }

  #modelo
  modelo = KNeighborsClassifier()

  #Criando os grid
  gridKNN = GridSearchCV(estimator= modelo , param_grid= valores_grid, cv = 5)
  gridKNN.fit(X_norm,y)


  #resultado
  print("""
  Melhor acurácia: {0};
  Melhor K: {1};
  Melhor Distância: {2};
  Melhor p: {3};
  {4}

  """.format(gridKNN.best_score_,
            gridKNN.best_estimator_.n_neighbors,
            gridKNN.best_estimator_.metric,
            gridKNN.best_estimator_.p,
            normalizador))


#Normaliando os dados para trabalhar com distância

normalizadores = [[normalize(x, norm='l1'), normalize(x, norm='l2'), normalize(x, norm='max')] ,
                  ['normalize-l1', 'normalize-l2','normalize-max']]

for normalizador,tipo in zip(normalizadores[0],normalizadores[1]):
  X_norm = normalizador


  #Hyperparameter
  valores_k = np.array([3,5,7,9,11])
  calculo_distancia = ['minkowski', 'chebyshev']
  valores_p = np.array([1,2,3,4])
  valores_grid = {'n_neighbors':valores_k, 'metric':calculo_distancia, 'p':valores_p }

  #modelo
  modelo = KNeighborsClassifier()


  #Criando os grid
  gridKNN = GridSearchCV(estimator= modelo , param_grid= valores_grid, cv = 5)
  gridKNN.fit(X_norm,y)


  #resultado
  print("""
  Melhor acurácia: {0};
  Melhor K: {1};
  Melhor Distância: {2};
  Melhor p: {3};
  {4}

  """.format(gridKNN.best_score_,
            gridKNN.best_estimator_.n_neighbors,
            gridKNN.best_estimator_.metric,
            gridKNN.best_estimator_.p,
            tipo))
normalizador = StandardScaler()
X_norm = normalizador.fit_transform(x)


x_treino,x_teste, y_treino, y_teste = train_test_split(X_norm,y,test_size= 0.30,random_state=14)

modelo = KNeighborsClassifier(n_neighbors=7,metric= 'chebyshev', p=1 )
modelo.fit(x_treino,y_treino)
predicao = modelo.predict(x_teste)
matrix = confusion_matrix(y_teste,predicao)
print(matrix)
#Definindo valores que serão testados em DecisionTree
minimos_split = np.array([2,3,4,5,6,7,8])
maximo_nivel = np.array([3,4,5,6,7,8,9,10,11])
algoritmo = ['gini', 'entropy']
valores_grid = {'min_samples_split': minimos_split, 'max_depth': maximo_nivel, 'criterion': algoritmo}
#model
modelo = DecisionTreeClassifier()

#Criando as grid
griDecisionTree = GridSearchCV(estimator=modelo, param_grid= valores_grid, cv = 5)
griDecisionTree.fit(x,y)

#Imprimindo melhores parâmetros

print("""
O Melhor Split: {0};
Máxima Profundidade: {1};
Algoritmo escolhido: {2};
Acurácia: {3}
""".format(griDecisionTree.best_estimator_.min_samples_split,
           griDecisionTree.best_estimator_.max_depth,
           griDecisionTree.best_estimator_.criterion,
           griDecisionTree.best_score_))
x_treino,x_teste, y_treino, y_teste = train_test_split(x,y,test_size= 0.30,random_state=14)

modelo = DecisionTreeClassifier(min_samples_split=5,max_depth=6)
modelo.fit(x_treino,y_treino)
predicao = modelo.predict(x_teste)
matrix = confusion_matrix(y_teste,predicao)
print(matrix)
