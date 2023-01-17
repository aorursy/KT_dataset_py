# Pacotes basicos

import pandas as pd

import numpy as np

import seaborn as sns

import itertools

import imblearn

import math



# Metricas e Graficos

from sklearn import preprocessing

from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

from seaborn import countplot, lineplot, barplot

from sklearn.metrics import accuracy_score, confusion_matrix

from scipy.stats import kurtosis, skew



# Modelos

from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier



# Tratamento de warning e exibição no Jupyter

import warnings

warnings.filterwarnings(action="ignore")

pd.options.display.max_seq_items = 8000

pd.options.display.max_rows = 8000

pd.set_option('display.max_columns', None)



# Matplot 

import matplotlib.pyplot as plt

import matplotlib.style as style 

%matplotlib inline

style.use('ggplot')



# Outras libs

import pickle

import os

from time import time

import gc

gc.enable()
treino = pd.read_csv('../input/competicao-dsa-machine-learning-sep-2019/X_treino.csv')

teste = pd.read_csv('../input/competicao-dsa-machine-learning-sep-2019/X_teste.csv')

target = pd.read_csv('../input/competicao-dsa-machine-learning-sep-2019/y_treino.csv')

sub = pd.read_csv('../input/competicao-dsa-machine-learning-sep-2019/sample_submission.csv')
# Primeiros registros do dataset de treino

treino.head()
# Primeiros registros do dataset de teste

teste.head()
# Primeiros registros do dataset target

target.head()
# Analise estatística do dataset de treino

treino.describe().T
# Analise estatística do dataset de teste

teste.describe().T
# Analise estatística do dataset target

target.describe()
# Cada serie tem 128 medidas

len(treino.measurement_number.value_counts())
# Verificar se existem dados nulos no dataset de treino

treino.isnull().values.any() 
# Verificar se existem dados nulos no dataset de teste

teste.isnull().values.any() 
# Existem 6 series a mais no dataset de teste

(teste.shape[0] - treino.shape[0]) / 128
# Existe 73 grupos unicos no dataset target 

target['group_id'].nunique()
# Visualizando todos os tipos de superfície do dataset target, ordenado pela quantidade de registros

sns.countplot(y = 'surface',

              data = target,

              order = target['surface'].value_counts().index)

plt.show()
# Visualizando a distribuição das features: group_id e surface

# Créditos: https://www.kaggle.com/gpreda/robots-need-help

fig, ax = plt.subplots(1,1,figsize=(26,8))

grp = pd.DataFrame(target.groupby(['group_id', 'surface'])['series_id'].count().reset_index())

piv = grp.pivot(index='surface', columns='group_id', values='series_id')

grafico = sns.heatmap(piv, linewidths=.1, linecolor='black', annot=True, cmap="YlGnBu")

grafico.set_title('Surface x Grupo_id', size=16)

plt.show()
# Grafico de contador de numero de registros por group_id, ordenado

plt.figure(figsize=(23,5)) 

countplot(x="group_id", data=target, order=target['group_id'].value_counts().index)

plt.show()
series_dict = {}

for series in (treino['series_id'].unique()):

    series_dict[series] = treino[treino['series_id'] == series]  
def plot_series(series_id):

    plt.figure(figsize=(28, 16))

    print(target[target['series_id'] == series_id]['surface'].values[0].title())

    for i, col in enumerate(series_dict[series_id].columns[3:]):

        if col.startswith("o"):

            color = 'red'

        elif col.startswith("a"):

            color = 'green'

        else:

            color = 'blue'

        if i >= 7:

            i+=1

        plt.subplot(3, 4, i + 1)

        plt.plot(series_dict[series_id][col], color=color, linewidth=3)

        plt.title(col)
# Visualizando a serie de código 0

id_series = 0

plot_series(id_series)
f,ax = plt.subplots(figsize=(8, 8))

sns.heatmap(treino.iloc[:,3:].corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
def plot_distribution(df1, df2, label1, label2, features,a=2,b=5):

    i = 0

    sns.set_style('whitegrid')

    plt.figure()

    fig, ax = plt.subplots(a,b,figsize=(17,9))



    for feature in features:

        i += 1

        plt.subplot(a,b,i)

        sns.kdeplot(df1[feature], bw=0.5,label=label1)

        sns.kdeplot(df2[feature], bw=0.5,label=label2)

        plt.xlabel(feature, fontsize=9)

        locs, labels = plt.xticks()

        plt.tick_params(axis='x', which='major', labelsize=8)

        plt.tick_params(axis='y', which='major', labelsize=8)

    plt.show();
# Gráfico de distribuição por dataset (treino x teste)

features = treino.columns.values[3:]

plot_distribution(treino, teste, 'Treino', 'Teste', features)
def plot_classes_distribution(classes, tt, features,a=5,b=2):

    i = 0

    plt.figure()

    fig, ax = plt.subplots(a,b,figsize=(16,24))



    for feature in features:

        i += 1

        plt.subplot(a,b,i)

        for cl in classes:

            ttc = tt[tt['surface']==cl]

            sns.kdeplot(ttc[feature], bw=0.5,label=cl)

        plt.xlabel(feature, fontsize=9)

        locs, labels = plt.xticks()

        plt.tick_params(axis='x', which='major', labelsize=8)

        plt.tick_params(axis='y', which='major', labelsize=8)

    plt.show();
# Gráfico de distribuição por classe

classes = (target['surface'].value_counts()).index

aux = treino.merge(target, on='series_id', how='inner')

plot_classes_distribution(classes, aux, features)
# Funcao para converter Quaternions para Angulos de Euler

def quaternion_to_euler(x, y, z, w):



    t0 = +2.0 * (w * x + y * z)

    t1 = +1.0 - 2.0 * (x * x + y * y)

    X = math.atan2(t0, t1)



    t2 = +2.0 * (w * y - z * x)

    t2 = +1.0 if t2 > +1.0 else t2

    t2 = -1.0 if t2 < -1.0 else t2

    Y = math.asin(t2)



    t3 = +2.0 * (w * z + x * y)

    t4 = +1.0 - 2.0 * (y * y + z * z)

    Z = math.atan2(t3, t4)



    return X, Y, Z



# Funções para criação de features estatísticas

def _kurtosis(x):

    return kurtosis(x)



def skewness(x):

    return skew(x)



def SSC(x):

    x = np.array(x)

    x = np.append(x[-1], x)

    x = np.append(x,x[1])

    xn = x[1:len(x)-1]

    xn_i2 = x[2:len(x)]    

    xn_i1 = x[0:len(x)-2]  

    ans = np.heaviside((xn-xn_i1)*(xn-xn_i2),0)

    return sum(ans[1:]) 



def wave_length(x):

    x = np.array(x)

    x = np.append(x[-1], x)

    x = np.append(x,x[1])

    xn = x[1:len(x)-1]

    xn_i2 = x[2:len(x)]    

    return sum(abs(xn_i2-xn))

    

def norm_entropy(x):

    tresh = 3

    return sum(np.power(abs(x),tresh))



def SRAV(x):    

    SRA = sum(np.sqrt(abs(x)))

    return np.power(SRA/len(x),2)



def mean_abs(x):

    return sum(abs(x))/len(x)



def zero_crossing(x):

    x = np.array(x)

    x = np.append(x[-1], x)

    x = np.append(x,x[1])

    xn = x[1:len(x)-1]

    xn_i2 = x[2:len(x)]    # xn+1

    return sum(np.heaviside(-xn*xn_i2,0))
# Função para criação de novas features

def fn_features_01(df):

    df['totl_anglr_vel'] = (df['angular_velocity_X']**2 + df['angular_velocity_Y']**2 + df['angular_velocity_Z']**2)** 0.5

    df['totl_linr_acc'] = (df['linear_acceleration_X']**2 + df['linear_acceleration_Y']**2 + df['linear_acceleration_Z']**2)**0.5

    df['totl_xyz'] = (df['orientation_X']**2 + df['orientation_Y']**2 + df['orientation_Z']**2)**0.5

    df['acc_vs_vel'] = df['totl_linr_acc'] / df['totl_anglr_vel']

    df['norm_quat'] = (df['orientation_X']**2 + df['orientation_Y']**2 + df['orientation_Z']**2 + df['orientation_W']**2)

    df['mod_quat'] = (df['norm_quat'])**0.5

    df['norm_X'] = df['orientation_X'] / df['mod_quat']

    df['norm_Y'] = df['orientation_Y'] / df['mod_quat']

    df['norm_Z'] = df['orientation_Z'] / df['mod_quat']

    df['norm_W'] = df['orientation_W'] / df['mod_quat']

    

    x, y, z, w = df['norm_X'].tolist(), df['norm_Y'].tolist(), df['norm_Z'].tolist(), df['norm_W'].tolist()

    nx, ny, nz = [], [], []

    for i in range(len(x)):

        xx, yy, zz = quaternion_to_euler(x[i], y[i], z[i], w[i])

        nx.append(xx)

        ny.append(yy)

        nz.append(zz)

    

    df['euler_x'] = nx

    df['euler_y'] = ny

    df['euler_z'] = nz



    return df
# Função para criação de novas features, agrupando por series_id

def fn_features_02(data):

    df = pd.DataFrame()

    

    def mean_change_of_abs_change(x):

        return np.mean(np.diff(np.abs(np.diff(x))))

    

    for col in data.columns:

        if col in ['row_id','series_id','measurement_number', 'orientation_X', 'orientation_Y', 'orientation_Z', 'orientation_W']:

            continue

        df[col + '_mean'] = data.groupby(['series_id'])[col].mean()

        df[col + '_median'] = data.groupby(['series_id'])[col].median()

        df[col + '_max'] = data.groupby(['series_id'])[col].max()

        df[col + '_min'] = data.groupby(['series_id'])[col].min()

        df[col + '_std'] = data.groupby(['series_id'])[col].std()

        df[col + '_range'] = df[col + '_max'] - df[col + '_min']

        df[col + '_maxtoMin'] = df[col + '_max'] / df[col + '_min']

        df[col + '_mean_abs_chg'] = data.groupby(['series_id'])[col].apply(lambda x: np.mean(np.abs(np.diff(x))))

        df[col + '_mean_change_of_abs_change'] = data.groupby('series_id')[col].apply(mean_change_of_abs_change)

        df[col + '_abs_max'] = data.groupby(['series_id'])[col].apply(lambda x: np.max(np.abs(x)))

        df[col + '_abs_min'] = data.groupby(['series_id'])[col].apply(lambda x: np.min(np.abs(x)))

        df[col + '_abs_avg'] = (df[col + '_abs_min'] + df[col + '_abs_max'])/2

        

        # Advanced Features

        df[col + '_skew'] = data.groupby(['series_id'])[col].skew()

        df[col + '_mad'] = data.groupby(['series_id'])[col].mad()

        df[col + '_q25'] = data.groupby(['series_id'])[col].quantile(0.25)

        df[col + '_q75'] = data.groupby(['series_id'])[col].quantile(0.75)

        df[col + '_q95'] = data.groupby(['series_id'])[col].quantile(0.95)

        df[col + '_iqr'] = df[col + '_q75'] - df[col + '_q25']

        df[col + '_SSC'] = data.groupby(['series_id'])[col].apply(SSC) 

        df[col + '_skewness'] = data.groupby(['series_id'])[col].apply(skewness)

        df[col + '_wave_lenght'] = data.groupby(['series_id'])[col].apply(wave_length)

        df[col + '_norm_entropy'] = data.groupby(['series_id'])[col].apply(norm_entropy)

        df[col + '_SRAV'] = data.groupby(['series_id'])[col].apply(SRAV)

        df[col + '_kurtosis'] = data.groupby(['series_id'])[col].apply(_kurtosis) 

        df[col + '_zero_crossing'] = data.groupby(['series_id'])[col].apply(zero_crossing) 



    return df

    
# Aplicando novas features nos datasets de treino e teste

treino = fn_features_01(treino)

teste = fn_features_01(teste)

treino.shape, teste.shape
# Aplicando novas features nos datasets de treino e teste

# Esta celula demora um pouco para concluir (cerca de 10min)

treino = fn_features_02(treino)

teste = fn_features_02(teste)

treino.shape, teste.shape
# Visualizando os primeiros registros do dataset de treino com as novas features

treino.head()
# Preenchendo os valores NA e inf com zero

# Acontece após a criação das novas variáveis estatísticas

treino.fillna(0,inplace=True)

treino.replace(-np.inf,0,inplace=True)

treino.replace(np.inf,0,inplace=True)



teste.fillna(0,inplace=True)

teste.replace(-np.inf,0,inplace=True)

teste.replace(np.inf,0,inplace=True)
# Transformando a feature 'surface' de string para numérico

le = preprocessing.LabelEncoder()

target['surface'] = le.fit_transform(target['surface'])
# Utilizando o método StratifiedKFold para realizar os grupos de treinamento

folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=2019)
previsao = np.zeros((teste.shape[0],9))

real = np.zeros((treino.shape[0]))

score = 0
# Execução da criação e treinamento do modelo

# Utilização do algoritmo RANDOM FOREST CLASSIFIER



for times, (trn_idx, val_idx) in enumerate(folds.split(treino.values, target['surface'].values)):

    rf = RandomForestClassifier(n_estimators=500, n_jobs = -1)

    rf.fit(treino.iloc[trn_idx], target['surface'][trn_idx])

    real[val_idx] = rf.predict(treino.iloc[val_idx])

    previsao += rf.predict_proba(teste) / folds.n_splits

    score += rf.score(treino.iloc[val_idx], target['surface'][val_idx])

    print("Fold: {} score: {}".format(times, rf.score(treino.iloc[val_idx], target['surface'][val_idx])))

    gc.collect()
print('Acuracia Media: ', score / folds.n_splits)
confusion_matrix(real, target['surface'])
sub['surface'] = le.inverse_transform(previsao.argmax(axis=1))

#sub.to_csv('submission_rf.csv', index=False)

sub.head()
# Carregando novamente os datasets

tt_treino = pd.read_csv('../input/competicao-dsa-machine-learning-sep-2019/X_treino.csv')

tt_teste = pd.read_csv('../input/competicao-dsa-machine-learning-sep-2019/X_teste.csv')

tt_y_treino = pd.read_csv('../input/competicao-dsa-machine-learning-sep-2019/y_treino.csv')

ss = pd.read_csv('../input/competicao-dsa-machine-learning-sep-2019/sample_submission.csv')
# Concatenando os datasets de treino e teste

full = pd.concat([tt_treino, tt_teste])

full = full.iloc[:,3:].values.reshape(-1,128,10)
# Funções para calcular a distancia entre as amostras de dados

# O objetivo é identificar links entre os datasets de treino e teste

# Caso exista relações, estas serão utilizadas para avaliação no Kaggle

def sq_dist(a,b):

    return np.sum((a-b)**2, axis=1)



def find_run_edges(data, edge):

    if edge == 'left':

        border1 = 0

        border2 = -1

    elif edge == 'right':

        border1 = -1

        border2 = 0

    else:

        return False

    

    edge_list = []

    linked_list = []

    

    for i in range(len(data)):

        dist_list = sq_dist(data[i, border1, :4], data[:, border2, :4])

        min_dist = np.min(dist_list)

        closest_i   = np.argmin(dist_list)

        if closest_i == i:

            closest_i = np.argsort(dist_list)[1]

        dist_list = sq_dist(data[closest_i, border2, :4], data[:, border1, :4])

        rev_dist = np.min(dist_list)

        closest_rev = np.argmin(dist_list)

        if closest_rev == closest_i:

            closest_rev = np.argsort(dist_list)[1]

        if (i != closest_rev):

            edge_list.append(i)

        else:

            linked_list.append([i, closest_i, min_dist])

            

    return edge_list, linked_list



def find_runs(data, left_edges, right_edges):

    data_runs = []



    for start_point in left_edges:

        i = start_point

        run_list = [i]

        while i not in right_edges:

            tmp = np.argmin(sq_dist(data[i, -1, :4], data[:, 0, :4]))

            if tmp == i: # self-linked sample

                tmp = np.argsort(sq_dist(data[i, -1, :4], data[:, 0, :4]))[1]

            i = tmp

            run_list.append(i)

        data_runs.append(np.array(run_list))

    

    return data_runs
# Procurando por link entre os dados

train_left_edges, train_left_linked  = find_run_edges(full, edge='left')

train_right_edges, train_right_linked = find_run_edges(full, edge='right')

train_runs = find_runs(full, train_left_edges, train_right_edges)

print('Found', len(train_left_edges), 'left edges and', len(train_right_edges), 'right edges.')
ss['surface'] = ''

df_surface = ''



for i in range(151):

    x = train_runs[i]

    x = np.sort(x)

    if x[0]<3810:

        df_surface = tt_y_treino['surface'][x[0]]

        for j in range(len(train_runs[i])):

            if train_runs[i][j]-3810>-1:

                ss['surface'][train_runs[i][j]-3810] = df_surface
ss.head()
y_train_runs = ss.copy()



sub_final = {}

for i in range(0, sub.shape[0]):

    sub_final.update({sub.iloc[i]['series_id'] : sub.iloc[i]['surface'] })

    

y_train_runs.head()
resultado = []

for i in range(0, y_train_runs.shape[0]):

    if (y_train_runs.surface[i] == ''):

        resultado.append(sub_final[i])

    else:

        resultado.append(y_train_runs.surface[i])

        

y_train_runs['surface'] = resultado

y_train_runs.to_csv('best_submission_rf.csv', index=False)

y_train_runs.head()
y_train_runs.surface.value_counts()
y_train_runs.surface.value_counts() / y_train_runs.shape[0]