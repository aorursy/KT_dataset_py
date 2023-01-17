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
from sklearn.preprocessing import StandardScaler

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
    plt.title('Estimativa (*) X real (o)')
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
    plt.title('Estimativa (*) X real (o)')
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
    plt.title('Estimativa (*) X real (o)')
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
    plt.title('Estimativa (*) X real (o)')
    plt.grid(True)
    plt.show()

    
    
    print('Importâncias {}'.format(estimatorTree.feature_importances_))
    
    confusion = confusion_matrix(y_test, estimatorTree.predict(x_test))
    print("\nConfusion matrix:\n{}".format(confusion))
    
    f1 = f1_score(y_test, estimatorNN.predict(x_test), average ='micro')
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
# fonte Introduction to Machine Learning with Python
# by Andreas C. Müller and Sarah Guido

mglearn.plots.plot_kmeans_algorithm()

from sklearn.cluster import KMeans
scaler = StandardScaler()
scaler.fit(world_happiness_resumido_X)
world_happiness_resumido_X_scaled = scaler.transform(world_happiness_resumido_X)

y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(world_happiness_resumido_X_scaled)


world_happiness['Cluster'] = y_pred

world_happiness.head(10)


world_happiness.plot.scatter(x='Cluster',y='Happiness Rank')
f, (ax) = plt.subplots(1, 1, figsize=(12, 4))
f.suptitle(' ', fontsize=14)

sns.boxplot(x="Cluster", y="Happiness Score", data=world_happiness,  ax=ax)
ax.set_xlabel(" ",size = 12,alpha=0.8)
ax.set_ylabel(" ",size = 12,alpha=0.8)
from pandas.tools.plotting import parallel_coordinates
fig, ax = plt.subplots(1, 1, figsize=(19,10))
parallel_coordinates(frame=world_happiness, class_column='Cluster', color = ('r','g','b','y'), ax = ax, cols=['Happiness Score',
       'Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)',
       'Freedom', 'Trust (Government Corruption)', 'Generosity',
       'Dystopia Residual'])

from sklearn.metrics.cluster import silhouette_score

silhouette_score(world_happiness_resumido_X_scaled, y_pred)
# Código copiado de https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py
# com alterações
from sklearn.metrics import silhouette_samples
import matplotlib.cm as cm

X = world_happiness_resumido_X_scaled

range_n_clusters = [2, 3, 4, 5, 6]

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

plt.show()
from sklearn.cluster import DBSCAN

dbscan = DBSCAN()
clusters = dbscan.fit_predict(world_happiness_resumido_X)
world_happiness['Cluster_DBSCAN'] = clusters
world_happiness.plot.scatter(x='Cluster_DBSCAN',y='Happiness Rank')
scaler = StandardScaler()
scaler.fit(cancer_data_DF)
cancer_data_scaled_DF = scaler.transform(cancer_data_DF)

y_pred = KMeans(n_clusters=2, random_state=random_state).fit_predict(cancer_data_scaled_DF)

cancer_data_DF['Cluster'] = y_pred
cancer_data_DF.head(10)

plt.plot(cancer_target_DF.values, y_pred,'b.')
plt.plot(range(len(y_pred)), cancer_target_DF, 'r.')

plt.ylabel('Estimativa')
plt.title('Rede Estimativa (*) X real (o)')
plt.grid(True)
plt.show()

confusion = confusion_matrix (y_pred, cancer_target_DF.values)
print ("Matriz de confusão: \n {}".format(confusion))