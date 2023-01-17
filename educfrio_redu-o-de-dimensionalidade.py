import numpy as np
import matplotlib.pyplot as plt
import math
import random
import pandas as pd
import scipy.stats as stat
import seaborn as sns
import os
import pandas
import sklearn

from IPython.display import Image
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn import preprocessing

from sklearn.preprocessing import MinMaxScaler


# Para ter repetibilidade nos resultados
random_state = 1

# Tratar valores infinitos como np.NaN
pandas.options.mode.use_inf_as_na = True

# IMPORTANTE para tornar figuras interativas
%matplotlib notebook

# Tamanho padrão das figuras
figsize=(10,6)

# Verificação do local para carga de dados
path = os.environ['PATH']

if path.startswith('C'):
    IN_KAGGLE = False
else:
    IN_KAGGLE = True
    

# Bibliotecas específicas do livro Introduction to Machine Learning with Python
# https://github.com/amueller/introduction_to_ml_with_python
# pip install mglearn

import mglearn


# Configuração do número de linhas e colunas a serem apresentadas em listagens
pd.set_option('display.max_row', 1000)

pd.set_option('display.max_columns', 50)

os.listdir('../input')
# Função de conversão de dados copiada de https://github.com/shakedzy/dython/blob/master/dython/_private.py
# Autor Shaked Zychlinski

def convert(data, to):
    converted = None
    if to == 'array':
        if isinstance(data, np.ndarray):
            converted = data
        elif isinstance(data, pd.Series):
            converted = data.values
        elif isinstance(data, list):
            converted = np.array(data)
        elif isinstance(data, pd.DataFrame):
            converted = data.as_matrix()
    elif to == 'list':
        if isinstance(data, list):
            converted = data
        elif isinstance(data, pd.Series):
            converted = data.values.tolist()
        elif isinstance(data, np.ndarray):
            converted = data.tolist()
    elif to == 'dataframe':
        if isinstance(data, pd.DataFrame):
            converted = data
        elif isinstance(data, np.ndarray):
            converted = pd.DataFrame(data)
    else:
        raise ValueError("Unknown data conversion: {}".format(to))
    if converted is None:
        raise TypeError('cannot handle data conversion of type: {} to {}'.format(type(data),to))
    else:
        return converted
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


def redes_neurais_regressao(X_, Y_, to_scale=True):

    X_ = convert(X_, 'array')
        
    Y_ = convert(Y_, 'array')
    
    # Transforma Y em array 1-D
    #Y_ = np.ravel(Y_)
    
    if to_scale:
        # Escala variáveis
        scaler = MinMaxScaler(feature_range=(0, 1))

        X_escale = scaler.fit_transform(X_) 
        Y_escale = scaler.fit_transform(Y_) 
    else:
        X_escale = X_
        Y_escale = Y_

    x_train, x_test, y_train, y_test = train_test_split(
        X_escale, Y_escale, test_size=0.1, random_state=random_state,shuffle =True)

    estimatorNN = MLPRegressor(
                              learning_rate = 'adaptive',
                              random_state = random_state,
                              verbose=False,
                                max_iter = 200,
                            hidden_layer_sizes = [100,50,40,30,20,10],   
                    solver = 'adam',
                    alpha = 0.0001,
                    activation = 'relu'
                            )

    estimatorNN.fit(x_train,y_train)
    
    plt.subplots(figsize=figsize)
    plt.plot(range(len(y_test)), y_test,'ro')
    plt.plot(range(len(y_test)), estimatorNN.predict(x_test),'b*')
    

    plt.ylabel('Estimativa')
    plt.title('Rede Estimativa (*) X real (o)')
    plt.grid(True)
    plt.show()
    
    mean_error = mean_absolute_error(y_test, estimatorNN.predict(x_test))
    print('\nErro {}'.format(mean_error))
    
    mean_s_error = mean_squared_error(y_test, estimatorNN.predict(x_test))
    print('\nErro {}'.format(mean_s_error))
    
    r2 = r2_score(y_test, estimatorNN.predict(x_test)) 
    print('\nR2 Score {}'.format(r2))
    
    return estimatorNN,r2
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

def redes_neurais_classificacao(X_, Y_, to_scale=True):

    X_ = convert(X_, 'array')
        
    Y_ = convert(Y_, 'array')
    
    # Transforma Y em array 1-D
    Y_ = np.ravel(Y_)
    
    if to_scale:
        # Escala variáveis
        scaler = MinMaxScaler(feature_range=(0, 1))

        X_escale = scaler.fit_transform(X_) 
        #Y_escale = scaler.fit_transform(Y_) 
    else:
        X_escale = X_

    x_train, x_test, y_train, y_test = train_test_split(
        X_escale, Y_, test_size=0.1, random_state=random_state,shuffle =True)

    estimatorNN = MLPClassifier(
                              learning_rate = 'adaptive',
                              random_state = random_state,
                              verbose=False,
                                max_iter = 200,
                            hidden_layer_sizes = [100,50,40,30,20,10],   
                    solver = 'adam',
                    alpha = 0.0001,
                    activation = 'relu'
                            )

    estimatorNN.fit(x_train,y_train)
    
    plt.subplots(figsize=figsize)
    plt.plot(range(len(y_test)), y_test,'ro')
    plt.plot(range(len(y_test)), estimatorNN.predict(x_test),'b*')
    

    plt.ylabel('Estimativa')
    plt.title('Rede Estimativa (*) X real (o)')
    plt.grid(True)
    plt.show()
    
    # TN FP
    # FN TP
    confusion = confusion_matrix(y_test, estimatorNN.predict(x_test))
    print("\nConfusion matrix:\n{}".format(confusion))
    
    f1 = f1_score(y_test, estimatorNN.predict(x_test), average ='micro')
    print("\nf1 score: {:.2f}".format( f1   ))
    
    erro = np.sum(np.abs(estimatorNN.predict(x_test)-y_test))/len(y_test)
    print('\nErro {}'.format(erro))
    
    
    print(classification_report(y_test, estimatorNN.predict(x_test),
        target_names=["Falso", "Positivo"]))
    
    return estimatorNN,erro
from sklearn.tree import DecisionTreeRegressor

def arvore_regressao(X_, Y_, to_scale=True):
    
    X_ = convert(X_, 'array')
        
    Y_ = convert(Y_, 'array')
    
    # Transforma Y em array 1-D
    Y_ = np.ravel(Y_)
    
    if to_scale:
        # Escala variáveis
        scaler = MinMaxScaler(feature_range=(0, 1))

        X_escale = scaler.fit_transform(X_) 
        #Y_escale = scaler.fit_transform(Y_) 
    else:
        X_escale = X_

    x_train, x_test, y_train, y_test = train_test_split(
        X_escale, Y_, test_size=0.1, random_state=random_state,shuffle =True)
    
    estimatorTree = DecisionTreeRegressor(max_depth=5, random_state = random_state)
    estimatorTree.fit(x_train,y_train)
    
    plt.subplots(figsize=figsize)
    plt.plot(range(len(y_test)), y_test,'ro')
    plt.plot(range(len(y_test)), estimatorTree.predict(x_test),'b*')
    

    plt.ylabel('Estimativa')
    plt.title('Árvore Estimativa (*) X real (o)')
    plt.grid(True)
    plt.show()
    
    print('Importâncias {}'.format(estimatorTree.feature_importances_))
    
    mean_error = mean_absolute_error(y_test, estimatorTree.predict(x_test))
    print('\nErro {}'.format(mean_error))
    
    mean_s_error = mean_squared_error(y_test, estimatorTree.predict(x_test))
    print('\nErro {}'.format(mean_s_error))
    
    r2 = r2_score(y_test, estimatorTree.predict(x_test)) 
    print('\nR2 Score {}'.format(r2))
    
    return estimatorTree,r2
    

from sklearn.tree import DecisionTreeClassifier

def arvore_classificacao(X_, Y_, to_scale=True):
    
    X_ = convert(X_, 'array')
        
    Y_ = convert(Y_, 'array')
    
    # Transforma Y em array 1-D
    Y_ = np.ravel(Y_)
    
    if to_scale:
        # Escala variáveis
        scaler = MinMaxScaler(feature_range=(0, 1))

        X_escale = scaler.fit_transform(X_) 
        #Y_escale = scaler.fit_transform(Y_) 
    else:
        X_escale = X_

    x_train, x_test, y_train, y_test = train_test_split(
        X_escale, Y_, test_size=0.1, random_state=random_state,shuffle =True)
    
    estimatorTree = DecisionTreeClassifier(max_depth=5, random_state = random_state)
    estimatorTree.fit(x_train,y_train)
    
    plt.subplots(figsize=figsize)
    plt.plot(range(len(y_test)), y_test,'ro')
    plt.plot(range(len(y_test)), estimatorTree.predict(x_test),'b*')
    

    plt.ylabel('Estimativa')
    plt.title('Árvore Estimativa (*) X real (o)')
    plt.grid(True)
    plt.show()

    
    
    print('Importâncias {}'.format(estimatorTree.feature_importances_))
    
    confusion = confusion_matrix(y_test, estimatorTree.predict(x_test))
    print("\nConfusion matrix:\n{}".format(confusion))
    
    f1 = f1_score(y_test, estimatorTree.predict(x_test), average ='micro')
    print("\nf1 score: {:.2f}".format( f1   ))
    
    erro = np.sum(np.abs(estimatorTree.predict(x_test)-y_test))/len(y_test)
    print('\nErro {}'.format(erro))
    
    
    print(classification_report(y_test, estimatorTree.predict(x_test),
        target_names=["Falso", "Positivo"]))
    
    return estimatorTree,erro
    
    
if IN_KAGGLE:
    world_happiness = pd.read_csv("../input/world-happiness/2016.csv")
else:
    world_happiness = pd.read_csv("2016.csv")

# Conjunto completo
world_happiness = world_happiness.loc[:,['Country', 'Region', 'Happiness Rank', 'Happiness Score',
       'Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)',
       'Freedom', 'Trust (Government Corruption)', 'Generosity',
       'Dystopia Residual']]



#world_happiness = shuffle(world_happiness).reset_index(drop=True)

# Conjunto resumido para treinamento de modelos
world_happiness_resumido = world_happiness.loc[:,[ 'Happiness Score',
       'Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)',
       'Freedom', 'Trust (Government Corruption)', 'Generosity']]

# Cria variáveis para treinamento de modelos

colunas_fonte = [ 
       'Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)',
       'Freedom', 'Trust (Government Corruption)', 'Generosity'
]

colunas_objetivo = [ 
       'Happiness Score'
]

world_happiness_resumido_X = world_happiness_resumido.loc[:,colunas_fonte] 
world_happiness_resumido_Y = world_happiness_resumido.loc[:,colunas_objetivo]


world_happiness.head(35)
if IN_KAGGLE:
    tips = pd.read_csv('../input/snstips/tips.csv')
    if 'Unnamed: 0' in tips.columns:
        tips.drop(['Unnamed: 0'], inplace=True, axis=1)
else:
    tips = sns.load_dataset('tips')

tips.head()

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

cancer_data = cancer['data']
# 1 benigno, 0 maligno
cancer_target = cancer['target']
cancer_target_names  = cancer['target_names']
cancer_feature_names = cancer['feature_names']
cancer_data_DF = pd.DataFrame(cancer_data,columns=cancer_feature_names) 
cancer_data_DF.head()
cancer_target_DF = pd.DataFrame(cancer_target,columns=['target']) 
cancer_target_DF.head()
world_happiness_resumido_X.var()
from sklearn.feature_selection import VarianceThreshold

sel = VarianceThreshold(threshold=0.05)
world_happiness_resumido_VarianceThreshold = sel.fit_transform(world_happiness_resumido_X)
world_happiness_resumido_VarianceThreshold[0:5,:]
world_happiness_resumido_X.head()
world_happiness_resumido_X.shape
world_happiness_resumido_VarianceThreshold.shape
_,_ = redes_neurais_regressao(world_happiness_resumido_X, 
                              world_happiness_resumido_Y
                                       )
_,_ = redes_neurais_regressao(world_happiness_resumido_VarianceThreshold, 
                              world_happiness_resumido_Y
                                       )
estimatorArvore,_ = arvore_regressao(world_happiness_resumido_VarianceThreshold, 
                              world_happiness_resumido_Y)
min_max_scaler = preprocessing.MinMaxScaler()
cancer_data_DF_scaled = min_max_scaler.fit_transform(cancer_data_DF)
cancer_data_DF_scaled = pd.DataFrame(cancer_data_DF_scaled,columns=cancer_feature_names) 
cancer_data_DF_scaled.head()
cancer_data_DF_scaled.var()
cancer_data_DF_scaled.shape
from sklearn.feature_selection import VarianceThreshold

sel = VarianceThreshold(threshold=0.02)
cancer_data_DF_scaled_VarianceThreshold = sel.fit_transform(cancer_data_DF_scaled)
cancer_data_DF_scaled_VarianceThreshold.shape
_,_ = redes_neurais_classificacao(cancer_data_DF_scaled, 
                              cancer_target_DF
                                       )
_,_ = redes_neurais_classificacao(cancer_data_DF_scaled_VarianceThreshold, 
                              cancer_target_DF
                                       )


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

world_happiness_resumido_SelectKBest = SelectKBest(f_regression, k=2).fit_transform(world_happiness_resumido_X, np.ravel(world_happiness_resumido_Y))

fig, ax = plt.subplots(1, 1, figsize=figsize)
plt.scatter(world_happiness_resumido_SelectKBest[:,0], 
            world_happiness_resumido_SelectKBest[:,1], 
            s=world_happiness_resumido_Y.values**3)
plt.grid(True)
plt.tight_layout()
estimatorNN,_ = redes_neurais_regressao(world_happiness_resumido_SelectKBest, world_happiness_resumido_Y)
estimatorArvore,_ = arvore_regressao(world_happiness_resumido_SelectKBest, world_happiness_resumido_Y)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

cancer_data_SelectKBest = SelectKBest(f_classif, k=2).fit_transform(
    cancer_data_DF_scaled, np.ravel(cancer_target_DF))

fig, ax = plt.subplots(1, 1, figsize=figsize)
plt.scatter(cancer_data_SelectKBest[:,0], 
            cancer_data_SelectKBest[:,1], 
            s=(cancer_target_DF.values+1)*30)
plt.grid(True)
plt.tight_layout()
_,_ = redes_neurais_classificacao(cancer_data_SelectKBest, 
                              cancer_target_DF
                                       )
from sklearn.feature_selection import SelectFromModel

estimatorArvore,_ = arvore_regressao(world_happiness_resumido_X, world_happiness_resumido_Y)

model = SelectFromModel(estimatorArvore, prefit=True)
world_happiness_resumido_SelectKBest = model.transform(world_happiness_resumido_X)

world_happiness_resumido_X.shape
world_happiness_resumido_SelectKBest.shape
fig, ax = plt.subplots(1, 1, figsize=(19,10))
plt.scatter(world_happiness_resumido_SelectKBest[:,0], 
            world_happiness_resumido_SelectKBest[:,1], 
            s=world_happiness_resumido_Y.values**3)
plt.grid(True)
estimatorArvore,_ = arvore_regressao(world_happiness_resumido_SelectKBest, world_happiness_resumido_Y)
from sklearn.feature_selection import SelectFromModel

estimatorArvore,_ = arvore_classificacao(cancer_data_DF_scaled, cancer_target_DF)

model = SelectFromModel(estimatorArvore, prefit=True)
cancer_data_SelectKBest = model.transform(cancer_data_DF)

fig, ax = plt.subplots(1, 1, figsize=(19,10))
plt.scatter(cancer_data_SelectKBest[:,0], 
            cancer_data_SelectKBest[:,1], 
            s=(cancer_target_DF.values+1)*30)
plt.grid(True)
_,_ = redes_neurais_classificacao(cancer_data_SelectKBest, 
                              cancer_target_DF
                                       )

from sklearn.feature_selection import SelectPercentile

select = SelectPercentile(percentile=50)
select.fit(world_happiness_resumido_X, world_happiness_resumido_Y)
world_happiness_resumido_SelectPercentile = select.transform(world_happiness_resumido_X)
print("X.shape: {}".format(world_happiness_resumido_X.shape))
print("X_SelectPercentile.shape: {}".format(world_happiness_resumido_SelectPercentile.shape))
fig, ax = plt.subplots(1, 1, figsize=(19,10))
plt.scatter(world_happiness_resumido_SelectPercentile[:,0], 
            world_happiness_resumido_SelectPercentile[:,1], 
            s=world_happiness_resumido_Y.values**3)
plt.grid(True)
estimatorNN,_ = redes_neurais_regressao(world_happiness_resumido_SelectPercentile, world_happiness_resumido_Y)
estimatorArvore,_ = arvore_regressao(world_happiness_resumido_SelectPercentile, world_happiness_resumido_Y)




from sklearn.decomposition import FactorAnalysis
factor = FactorAnalysis(n_components=2, random_state=random_state).fit(world_happiness_resumido_X)
world_happiness_resumido_factor = factor.transform(world_happiness_resumido_X)
pd.DataFrame(factor.components_,columns=world_happiness_resumido_X.columns)

fig, ax = plt.subplots(1, 1, figsize=(19,10))
plt.scatter(world_happiness_resumido_factor[:,0], 
            world_happiness_resumido_factor[:,1], 
            s=world_happiness_resumido_Y.values**3)
plt.grid(True)
_,_ = redes_neurais_regressao(world_happiness_resumido_factor, world_happiness_resumido_Y)
_,_ = arvore_regressao(world_happiness_resumido_factor, world_happiness_resumido_Y)
from sklearn.decomposition import FactorAnalysis
factor = FactorAnalysis(n_components=2, random_state=random_state).fit(cancer_data_DF)
cancer_data_DF_factor = factor.transform(cancer_data_DF)
pd.DataFrame(factor.components_,columns=cancer_data_DF.columns)

fig, ax = plt.subplots(1, 1, figsize=(19,10))
plt.scatter(cancer_data_DF_factor[:,0], 
            cancer_data_DF_factor[:,1], 
            s=(cancer_target_DF.values+1)*30)
plt.grid(True)
_,_ = redes_neurais_classificacao(cancer_data_DF_factor, cancer_target_DF)
_,_ = arvore_classificacao(cancer_data_DF_factor, cancer_target_DF)
# fonte Introduction to Machine Learning with Python
# by Andreas C. Müller and Sarah Guido

import mglearn
mglearn.plots.plot_pca_illustration()


pca = PCA(random_state=random_state).fit(world_happiness_resumido_X)
print ('Variância por componente: {}'.format(pca.explained_variance_ratio_))
pd.DataFrame(pca.components_,columns=world_happiness_resumido_X.columns)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit (world_happiness_resumido_X)
world_happiness_resumido_X_scaled = scaler.transform(world_happiness_resumido_X)

pca = PCA(random_state=random_state).fit(world_happiness_resumido_X_scaled)
print ('Variância por componente: {}'.format(pca.explained_variance_ratio_))
pd.DataFrame(pca.components_,columns=world_happiness_resumido_X.columns)
# Variância do conjunto original

world_happiness_resumido_X.var()
# Variância dos dados projetados

world_happiness_resumido_PCA = PCA(random_state=random_state).fit_transform(world_happiness_resumido_X_scaled)
np.var(world_happiness_resumido_PCA, axis=0)
# PCA não garante separabilidade do conjunto

from sklearn.decomposition import PCA

world_happiness_resumido_PCA = PCA(n_components=2,random_state=random_state).fit_transform(world_happiness_resumido_X_scaled)

fig, ax = plt.subplots(1, 1, figsize=figsize)
plt.scatter(world_happiness_resumido_PCA[:,0], world_happiness_resumido_PCA[:,1], s=world_happiness_resumido_Y.values**3)
from sklearn.manifold import TSNE
tsne = TSNE(random_state=random_state)
# use fit_transform instead of fit, as TSNE has no transform method
# default 2 componentes
world_happiness_tsne = tsne.fit_transform(world_happiness_resumido_X_scaled)

plt.figure(figsize=figsize)
plt.scatter(world_happiness_tsne[:,0], world_happiness_tsne[:,1], s=world_happiness_resumido_Y.values**3)

plt.xlabel("t-SNE feature 0")
plt.xlabel("t-SNE feature 1")
world_happiness_resumido_X_scaled.shape
world_happiness_tsne.shape
# Kernel PCA pode melhorar a separabilidade
# http://scikit-learn.org/stable/auto_examples/decomposition/plot_kernel_pca.html#sphx-glr-auto-examples-decomposition-plot-kernel-pca-py


from sklearn.decomposition import PCA, KernelPCA


kpca = KernelPCA(
    kernel="sigmoid", 
    fit_inverse_transform=True)

world_happiness_resumido_PCA = kpca.fit_transform(world_happiness_resumido_X_scaled)

fig, ax = plt.subplots(1, 1, figsize=figsize)
plt.scatter(world_happiness_resumido_PCA[:,0], world_happiness_resumido_PCA[:,1], s=world_happiness_resumido_Y.values**3)
world_happiness_resumido_X_scaled.shape
world_happiness_resumido_PCA.shape
_,_ = arvore_regressao(world_happiness_resumido_PCA, world_happiness_resumido_Y)
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
scaler = StandardScaler()
scaler.fit(cancer.data)
X_scaled = scaler.transform(cancer.data)

from sklearn.decomposition import PCA
# keep the first two principal components of the data
pca = PCA(n_components=2)
# fit PCA model to breast cancer data
pca.fit(X_scaled)
# transform data onto the first two principal components
X_pca = pca.transform(X_scaled)
print("Original shape: {}".format(str(X_scaled.shape)))
print("Reduced shape: {}".format(str(X_pca.shape)))

plt.figure(figsize=(8, 8))
mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], cancer.target)
plt.legend(cancer.target_names, loc="best")
plt.gca().set_aspect("equal")
plt.xlabel("First principal component")
plt.ylabel("Second principal component")
plt.matshow(pca.components_, cmap='viridis')
plt.yticks([0, 1], ["First component", "Second component"])
plt.colorbar()
plt.xticks(range(len(cancer.feature_names)),
cancer.feature_names, rotation=60, ha='left')
plt.xlabel("Feature")
plt.ylabel("Principal components")
_,_ = redes_neurais_classificacao(X_pca[:,0:2], cancer_target_DF)
_,_ = arvore_classificacao(X_pca[:,0:2], cancer_target_DF)