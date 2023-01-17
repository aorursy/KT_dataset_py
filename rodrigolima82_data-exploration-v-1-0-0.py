# Carregando os pacotes

import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Statistic lib

from scipy import stats

from scipy.stats import skew, norm



# Sklearn lib

from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder



# Utils

import pandasql as ps

import re 

import math, string, os

import datetime



# Options

import warnings

warnings.filterwarnings('ignore')

pd.options.display.max_seq_items = 200

pd.options.display.max_rows = 200

pd.set_option('display.max_columns', None)

import gc

gc.enable()



# Variavel para controlar o treinamento no Kaggle

TRAIN_OFFLINE = False
def read_data():

    

    if TRAIN_OFFLINE:

        print('Carregando arquivo dataset_treino.csv....')

        train = pd.read_csv('../dataset/dataset_treino.csv')

        print('dataset_treino.csv tem {} linhas and {} colunas'.format(train.shape[0], train.shape[1]))

        

        print('Carregando arquivo dataset_teste.csv....')

        test = pd.read_csv('../dataset/dataset_teste.csv')

        print('dataset_teste.csv tem {} linhas and {} colunas'.format(test.shape[0], test.shape[1]))

        

        

        print('Carregando arquivo sample_submission.csv....')

        sample_submission = pd.read_csv('../dataset/sample_submission.csv')

        print('sample_submission.csv tem {} linhas and {} colunas'.format(sample_submission.shape[0], sample_submission.shape[1]))

    else:

        print('Carregando arquivo dataset_treino.csv....')

        train = pd.read_csv('/kaggle/input/competicao-dsa-machine-learning-dec-2019/dataset_treino.csv')

        print('dataset_treino.csv tem {} linhas and {} colunas'.format(train.shape[0], train.shape[1]))

        

        print('Carregando arquivo dataset_treino.csv....')

        test = pd.read_csv('/kaggle/input/competicao-dsa-machine-learning-dec-2019/dataset_teste.csv')

        print('dataset_teste.csv tem {} linhas and {} colunas'.format(test.shape[0], test.shape[1]))



        print('Carregando arquivo dataset_treino.csv....')

        sample_submission = pd.read_csv('/kaggle/input/competicao-dsa-machine-learning-dec-2019/sample_submission.csv')

        print('sample_submission.csv tem {} linhas and {} colunas'.format(sample_submission.shape[0], sample_submission.shape[1]))

    

    return train, test, sample_submission
# Leitura dos dados

train, test, sample_submission = read_data()
# Visualizando os primeiros registros do dataset

train.head()
# Visualizando os tipos das features

train.dtypes
# Visualizando dados estatisticos das variaveis numericas

train.describe().T
def percent_missing(df):

    data = pd.DataFrame(df)

    df_cols = list(pd.DataFrame(data))

    dict_x = {}

    for i in range(0, len(df_cols)):

        dict_x.update({df_cols[i]: round(data[df_cols[i]].isnull().mean()*100,2)})

    

    return dict_x
# Verificando as colunas com dados missing do dataset de treino

missing = percent_missing(train)

df_miss = sorted(missing.items(), key=lambda x: x[1], reverse=True)

print('Percent of missing data')

df_miss[0:133]
# Setup do plot

sns.set_style("white")

f, ax = plt.subplots(figsize=(18, 16))

sns.set_color_codes(palette='deep')



# Identificando os valores missing

missing = round(train.isnull().mean()*100,2)

missing = missing[missing > 0]

missing.sort_values(inplace=True)

missing.plot.bar(color="b")



# Visual presentation

ax.xaxis.grid(False)

ax.set(ylabel="Percent of missing values")

ax.set(xlabel="Features")

ax.set(title="Percent missing data by feature")

sns.despine(trim=True, left=True)
# Funcao para tratar os dados missing de cada variavel

def fill_na(data):

    data.fillna(data.mean(),inplace=True)
# Funcao para criar um plot de distribuicao para cada feature

def plot_distribution(dataset, cols=5, width=20, height=25, hspace=0.4, wspace=0.5):



    plt.style.use('fivethirtyeight')

    fig = plt.figure(figsize=(width, height))

    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)



    rows = math.ceil(float(dataset.shape[1]) / cols)



    for i, column in enumerate(dataset.columns):

        ax = fig.add_subplot(rows, cols, i + 1)

        ax.set_title(column)

        if dataset.dtypes[column] == np.object:



            g = sns.countplot(y=column, 

                              data=dataset,

                              order=dataset[column].value_counts().index[:10])



            substrings = [s.get_text()[:20] for s in g.get_yticklabels()]

            g.set(yticklabels=substrings)

            plt.xticks(rotation=25)

        else:

            g = sns.distplot(dataset[column])

            plt.xticks(rotation=25)
# Primeiro, vou preencher os dados missing com a media (apenas para iniciar as analises)

fill_na(train)
# Correlação de Pearson

cor_mat = train.corr(method = 'pearson')



# Visualizando o grafico de heatmap

f, ax = plt.subplots(figsize=(18, 18))

sns.heatmap(cor_mat,linewidths=.1,fmt= '.3f',ax=ax,square=True,cbar=True,annot=False)
# Descricao: é igual a 1 para indenizações que podem ser aprovadas rapidamente.

train['target'].describe()
# Analisando a variavel target

sns.set_style("white")

sns.set_color_codes(palette='deep')

f, ax = plt.subplots(figsize=(12, 6))



# Fit da distribuicao normal

mu, std = norm.fit(train["target"])



# Verificando a distribuicao de frequencia da variavel target

sns.distplot(train["target"], color="b", fit = stats.norm)

ax.xaxis.grid(False)

ax.set(ylabel="Frequency")

ax.set(xlabel="Target")

ax.set(title="Target distribution: mu = %.2f,  std = %.2f" % (mu, std))

sns.despine(trim=True, left=True)



# Adicionando Skewness e Kurtosis

ax.text(x=1.1, y=1, transform=ax.transAxes, s="Skewness: %f" % train["target"].skew(),\

        fontweight='demibold', fontsize=10, verticalalignment='top', horizontalalignment='right',\

        backgroundcolor='white', color='xkcd:poo brown')

ax.text(x=1.1, y=0.95, transform=ax.transAxes, s="Kurtosis: %f" % train["target"].kurt(),\

        fontweight='demibold', fontsize=10, verticalalignment='top', horizontalalignment='right',\

        backgroundcolor='white', color='xkcd:dried blood')



plt.show()
# Existe um problema de desbalanceamento de classes, ou seja, volume maior de um dos tipos de classe. 

# Podemos ver abaixo que existe uma clara desproporção 

# Apenas 23% sao indenizacoes que nao podem ser aprovadas rapidamente



# Visualizando a distribuição das classes (variavel TARGET)

pd.value_counts(train['target']).plot.bar()

plt.title('TARGET histogram')

plt.xlabel('TARGET')

plt.ylabel('Frequency')



# Visualizando um df com quantidade e percentual da variavel TARGET

df = pd.DataFrame(train['target'].value_counts())

df['%'] = 100*df['target']/train.shape[0]

df
# Primeiro, vamos remover a coluna ID

train.drop(['ID'], axis=1, inplace=True)
# Visualizando o grafico de distribuicao para cada feature (sao 132, entao é só uma amostra)

# Cada linha contem 6 features

columns_to_plot = []



for column in train:

    columns_to_plot.append(column)



plot_distribution(train[columns_to_plot], cols=6, width=100, height=100, hspace=1, wspace=1)