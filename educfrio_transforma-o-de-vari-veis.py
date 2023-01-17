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

from sklearn.cluster import KMeans

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
    plt.title('Rede - Estimativa (*) X real (o)')
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
    plt.title('Rede - Estimativa (*) X real (o)')
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
    plt.title('Árvore - Estimativa (*) X real (o)')
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
    plt.title('Árvore - Estimativa (*) X real (o)')
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
np.random.seed(random_state)


mu, sigma = 1, 1 
s = np.random.normal(mu, sigma, 1000)
ruido = np.random.rand(1000)*100
s_transformado = np.power(10+s*10, 2)+ruido

data_frame_exemplo = pd.DataFrame(data={'col1': s, 'col2': s_transformado, 'col3': s_transformado, 'grupo':[2]*1000})

data_frame_exemplo.loc[(data_frame_exemplo.col1<1)&(data_frame_exemplo.col2<500),'grupo'] = 1
data_frame_exemplo.loc[(data_frame_exemplo.col1>=0)&
                       (data_frame_exemplo.col2>=500)&
                       (data_frame_exemplo.col1<=2)&
                       (data_frame_exemplo.col2<=1000),'grupo'] = 2
data_frame_exemplo.loc[(data_frame_exemplo.col1>2)&
                       (data_frame_exemplo.col2>1000),'grupo'] = 3


data_frame_exemplo.iloc[0,:] = [-1.022201,23.687699,50000,1]

data_frame_exemplo.describe()


data_frame_exemplo.head()
data_frame_exemplo.info()

f, ax = plt.subplots(figsize=figsize)
_ = pd.plotting.scatter_matrix(data_frame_exemplo, ax=ax)
data_frame_exemplo.corr()
def analise(data_frame, col1, col2):
    # Plota dados 
    fig = plt.figure(figsize=figsize)
    plt.scatter(data_frame[col1], 
                data_frame[col2], 
                c=data_frame.grupo, 
                s=data_frame.grupo*100, 
                alpha=0.3,
                       cmap='viridis')
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title('Col X Col')
    plt.grid(True) 
    plt.show()
    
    # Classifica com rede neural
    estimatorNN,erro = redes_neurais_classificacao(data_frame.loc[:,[col1,col2]],
                                                   data_frame.loc[:,['grupo']],
                                                   to_scale=False)
    # Classificação Árvore
    estimatorNN,erro = arvore_classificacao(data_frame.loc[:,[col1,col2]],
                                                   data_frame.loc[:,['grupo']],
                                                   to_scale=False)
    # Clusteriza
    y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(data_frame.loc[:,[col1,col2]])
    y_pred = y_pred+1
    fig = plt.figure(figsize=figsize)
    plt.scatter(data_frame[col1], 
                data_frame[col2], 
                c=y_pred, 
                s=y_pred*100, 
                alpha=0.3,
                       cmap='viridis')
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title('Clusterização')
    plt.grid(True) 
    plt.show()
analise(data_frame_exemplo,'col1','col2')
analise(data_frame_exemplo,'col1','col3')
standard = preprocessing.StandardScaler()
data_frame_exemplo_scaled = standard.fit_transform(data_frame_exemplo.loc[:,['col1','col2','col3']])
data_frame_exemplo_scaled = pd.DataFrame(data=data_frame_exemplo_scaled, columns=['col1','col2','col3']) 
data_frame_exemplo_scaled.loc[:,'grupo'] = data_frame_exemplo.grupo
data_frame_exemplo_scaled.describe()
data_frame_exemplo_scaled.corr()
analise(data_frame_exemplo_scaled,'col1','col2')
analise(data_frame_exemplo_scaled,'col1','col3')

standard = preprocessing.MinMaxScaler()
data_frame_exemplo_scaled = standard.fit_transform(data_frame_exemplo.loc[:,['col1','col2','col3']])
data_frame_exemplo_scaled = pd.DataFrame(data=data_frame_exemplo_scaled, columns=['col1','col2','col3']) 
data_frame_exemplo_scaled.loc[:,'grupo'] = data_frame_exemplo.grupo
data_frame_exemplo_scaled.describe()

data_frame_exemplo_scaled.corr()
analise(data_frame_exemplo_scaled,'col1','col2')
analise(data_frame_exemplo_scaled,'col1','col3')


standard = preprocessing.Normalizer(norm='l2')
data_frame_exemplo_scaled = standard.fit_transform(data_frame_exemplo.loc[:,['col1','col2']])
data_frame_exemplo_scaled = pd.DataFrame(data=data_frame_exemplo_scaled, columns=['col1','col2']) 
data_frame_exemplo_scaled.loc[:,'grupo'] = data_frame_exemplo.grupo
data_frame_exemplo_scaled.describe()



data_frame_exemplo_scaled.corr()
data_frame_exemplo_scaled.head()
analise(data_frame_exemplo_scaled,'col1','col2')


standard = preprocessing.Normalizer(norm='l2')
data_frame_exemplo_scaled = standard.fit_transform(data_frame_exemplo.loc[:,['col1','col3']])
data_frame_exemplo_scaled = pd.DataFrame(data=data_frame_exemplo_scaled, columns=['col1','col3']) 
data_frame_exemplo_scaled.loc[:,'grupo'] = data_frame_exemplo.grupo
data_frame_exemplo_scaled.describe()



analise(data_frame_exemplo_scaled,'col1','col3')
from sklearn.preprocessing import QuantileTransformer



standard = QuantileTransformer(n_quantiles=10, random_state=random_state)
data_frame_exemplo_scaled = standard.fit_transform(data_frame_exemplo.loc[:,['col1','col2','col3']])
data_frame_exemplo_scaled = pd.DataFrame(data=data_frame_exemplo_scaled, columns=['col1','col2','col3']) 
data_frame_exemplo_scaled.loc[:,'grupo'] = data_frame_exemplo.grupo
data_frame_exemplo_scaled.describe()



data_frame_exemplo_scaled.corr()
analise(data_frame_exemplo_scaled,'col1','col2')
analise(data_frame_exemplo_scaled,'col1','col3')
min_max_scaler = preprocessing.MinMaxScaler()
cancer_data_DF_scaled = min_max_scaler.fit_transform(cancer_data_DF)
cancer_data_DF_scaled = pd.DataFrame(cancer_data_DF_scaled,columns=cancer_feature_names) 
cancer_data_DF_scaled.head()
f, ax = plt.subplots(1, 1, figsize=figsize)
f.suptitle('Cancer data', fontsize=14)

sns.boxplot(data=cancer_data_DF_scaled,  ax=ax)
#ax.set_xticklabels(cancer_data_DF.columns)
ax.set_xlabel("Atributos",size = 12,alpha=0.8)
ax.set_ylabel("Valores",size = 12,alpha=0.8)
plt.xticks(rotation=60)
plt.tight_layout()
# Correlação entre dados

f, ax = plt.subplots(figsize=figsize)
corr = pd.concat([cancer_data_DF_scaled, cancer_target_DF], axis=1).corr()
hm = sns.heatmap(corr, annot=True, ax=ax, cmap="coolwarm",fmt='.0f',
                 linewidths=.05)
f.subplots_adjust(bottom=0.3)
t= f.suptitle('Correlação entre variáveis', fontsize=14)
from pandas.plotting import parallel_coordinates
fig, ax = plt.subplots(1, 1, figsize=figsize)

parallel_coordinates(frame=pd.concat([cancer_data_DF_scaled, cancer_target_DF], axis=1), 
                     class_column='target'
                     , ax = ax, )
plt.xticks(rotation=60)
fig.subplots_adjust(bottom=0.3)
pd.concat([cancer_data_DF, cancer_target_DF], axis=1).corr()
pd.concat([cancer_data_DF_scaled, cancer_target_DF], axis=1).corr()
estimatorNN,erro = redes_neurais_classificacao(cancer_data_DF,
                                                   cancer_target_DF,
                                                   to_scale=False)
estimatorNN,erro = redes_neurais_classificacao(cancer_data_DF_scaled,
                                                   cancer_target_DF,
                                                   to_scale=False)
    
estimatorNN,erro = arvore_classificacao(cancer_data_DF,
                                                   cancer_target_DF,
                                                   to_scale=False)
# Classificação Árvore
estimatorNN,erro = arvore_classificacao(cancer_data_DF_scaled,
                                                   cancer_target_DF,
                                                   to_scale=False)
if IN_KAGGLE:
    tips = pd.read_csv('../input/snstips/tips.csv')
    if 'Unnamed: 0' in tips.columns:
        tips.drop(['Unnamed: 0'], inplace=True, axis=1)
else:
    tips = sns.load_dataset('tips')

tips.head()
# Códigos de categorias são estabelecidos na ordem alfabética dos valores da coluna
# Primeiro transformar a coluna em 'category'

tips['sex'] = tips['sex'].astype('category')
tips['sex_cat'] = tips['sex'].cat.codes

tips.head()
# A coluna original é apagada
# Usar colchetes na definição das colunas a serem codificadas

tips = pd.get_dummies(tips, columns=['smoker'], prefix=['s'])
tips.head()
# as codificações não seguem a posição dos elementos das categorias, as numerações são atribuídas iniciando em 0 na medida em que os valores aparecem

from pandas.api.types import CategoricalDtype

cat_type = CategoricalDtype(categories=['Mon','Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun'],  ordered=True)
tips['day'] = tips['day'].astype(cat_type)
tips['day_cat'] =tips['day'].cat.codes

tips.tail()
meals = {'breakfast': 1, 'Lunch' : 2, 'Dinner' : 3 }
tips['time_cat'] = tips.time.replace(meals, inplace=False)
tips['time_cat'] = tips.time_cat.astype('int')
tips.head()
#from sklearn.preprocessing import KBinsDiscretizer
#enc = KBinsDiscretizer(n_bins=10, encode='ordinal')
#X_binned = enc.fit_transform(tips.tip)
#X_binned
'''
if IN_KAGGLE:
    tips = pd.read_csv('../input/snstips/tips.csv')
    if 'Unnamed: 0' in tips.columns:
        tips.drop(['Unnamed: 0'], inplace=True, axis=1)
else:
    tips = sns.load_dataset('tips')

tips.head()
'''

from sklearn.preprocessing import Binarizer
X_binned = preprocessing.Binarizer(threshold=1.1).fit_transform(tips.tip.values.reshape(-1, 1))
X_binned[:5,:]
tips['TOTAL_LOG'] = np.log(tips.total_bill)
tips['TOTAL_LOG_INT']  = tips.TOTAL_LOG.astype(int, copy=False)
tips.TOTAL_LOG_INT.value_counts()
tips.info()
tips.describe()
#tips
tips.corr()


from collections import Counter

def conditional_entropy(x, y):

    """

    Calculates the conditional entropy of x given y: S(x|y)



    Wikipedia: https://en.wikipedia.org/wiki/Conditional_entropy



    :param x: list / NumPy ndarray / Pandas DataFrame

        A sequence of measurements

    :param y: list / NumPy ndarray / Pandas DataFrame

        A sequence of measurements

    :return: float

    """

    # entropy of x given y

    y_counter = Counter(y)

    xy_counter = Counter(list(zip(x,y)))

    total_occurrences = sum(y_counter.values())

    entropy = 0.0

    for xy in xy_counter.keys():

        p_xy = xy_counter[xy] / total_occurrences

        p_y = y_counter[xy[1]] / total_occurrences

        entropy += p_xy * math.log(p_y/p_xy)

    return entropy

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = stat.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

def theils_u(x, y):

    """

    Calculates Theil's U statistic (Uncertainty coefficient) for categorical-categorical association.

    This is the uncertainty of x given y: value is on the range of [0,1] - where 0 means y provides no information about

    x, and 1 means y provides full information about x.

    This is an asymmetric coefficient: U(x,y) != U(y,x)



    Wikipedia: https://en.wikipedia.org/wiki/Uncertainty_coefficient



    :param x: list / NumPy ndarray / Pandas DataFrame

        A sequence of categorical measurements

    :param y: list / NumPy ndarray / Pandas DataFrame

        A sequence of categorical measurements

    :return: float

        in the range of [0,1]

    """

    s_xy = conditional_entropy(x,y)

    x_counter = Counter(x)

    total_occurrences = sum(x_counter.values())

    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))

    s_x = stat.entropy(p_x)

    if s_x == 0:

        return 1

    else:

        return (s_x - s_xy) / s_x
    
def correlation_ratio(categories, measurements):

    """

    Calculates the Correlation Ration (sometimes marked by the greek letter Eta) for categorical-continuous association.

    Answers the question - given a continuous value of a measurement, is it possible to know which category is it

    associated with?

    Value is in the range [0,1], where 0 means a category cannot be determined by a continuous measurement, and 1 means

    a category can be determined with absolute certainty.



    Wikipedia: https://en.wikipedia.org/wiki/Correlation_ratio



    :param categories: list / NumPy ndarray / Pandas DataFrame

        A sequence of categorical measurements

    :param measurements: list / NumPy ndarray / Pandas DataFrame

        A sequence of continuous measurements

    :return: float

        in the range of [0,1]

    """

    categories = convert(categories, 'array')

    measurements = convert(measurements, 'array')

    fcat, _ = pd.factorize(categories)

    cat_num = np.max(fcat)+1

    y_avg_array = np.zeros(cat_num)

    n_array = np.zeros(cat_num)

    for i in range(0,cat_num):

        cat_measures = measurements[np.argwhere(fcat == i).flatten()]

        n_array[i] = len(cat_measures)

        y_avg_array[i] = np.average(cat_measures)

    y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)

    numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))

    denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))

    if numerator == 0:

        eta = 0.0

    else:

        eta = numerator/denominator

    return eta
if IN_KAGGLE:
    tips = pd.read_csv('../input/snstips/tips.csv')
    if 'Unnamed: 0' in tips.columns:
        tips.drop(['Unnamed: 0'], inplace=True, axis=1)
else:
    tips = sns.load_dataset('tips')

tips.head()
conditional_entropy(tips.sex,tips.tip)
cramers_v(tips.sex,tips.tip)
theils_u(tips.sex,tips.day)
#correlation_ratio(tips.sex,tips.day)
pd.crosstab(tips.sex,tips.day)