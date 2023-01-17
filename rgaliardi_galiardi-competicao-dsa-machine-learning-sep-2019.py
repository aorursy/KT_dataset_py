# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os

import gc

import pickle

from numpy.fft import *



from IPython.display import FileLink



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

dirname = None



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

print("Diretório: " + dirname)

# Any results you write to the current directory are saved as output.
from matplotlib import pyplot as plt

import matplotlib.style as style 

import seaborn as sns



style.use('ggplot')



import warnings

warnings.filterwarnings(action="ignore")



import plotly.offline as py 

from plotly.offline import init_notebook_mode, iplot

py.init_notebook_mode(connected=True) # this code, allow us to work with offline plotly version

import plotly.graph_objs as go # it's like "plt" of matplot



sns.set()

%matplotlib inline

%config InlineBackend.figure_format = 'retina'
import itertools

import pandasql as ps

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import multiprocessing as mp



import xgboost as xgb

import lightgbm as lgb



from xgboost import XGBClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import Ridge, LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.pipeline import Pipeline

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier



from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold, StratifiedKFold



from sklearn import preprocessing

from sklearn.preprocessing import scale

from sklearn.preprocessing import Imputer, MinMaxScaler

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder



from sklearn.metrics import accuracy_score

from sklearn.metrics import mean_squared_error as mse

from sklearn.metrics import mean_absolute_error as mae

from sklearn.metrics import r2_score, make_scorer

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



from hyperopt import hp

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
models = []

classes = []

cpus = mp.cpu_count()

categorie = LabelEncoder()

results = pd.DataFrame(columns=['Name', 'Model', 'Predict', 'Accuracy', 'Score', 'File', 'Value'])
models.append(("LogisticRegression",LogisticRegression(solver='lbfgs', multi_class='multinomial')))

models.append(("Ridge",Ridge()))

models.append(("SVC",SVC()))

models.append(("LinearSVC",LinearSVC()))

models.append(("KNeighbors",KNeighborsClassifier()))

models.append(("DecisionTree",DecisionTreeClassifier()))

models.append(("ExtraTrees",ExtraTreesClassifier()))

models.append(("RandomForest",RandomForestClassifier()))

models.append(("RandomForest_entropy",RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=10, random_state=0, max_features=None)))

models.append(("RandomForest_gini",RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=10, random_state=0, max_features=None)))

models.append(("MLPClassifier",MLPClassifier(solver='lbfgs', random_state=0)))



models.append(("XGBoost",XGBClassifier()))

models.append(("XGBoost_XGB",xgb.XGBClassifier(n_estimators=100, max_depth=8, learning_rate=0.1, subsample=0.5)))

models.append(("XGBoost_Pipeline",Pipeline(steps=[('xgboost', xgb.XGBClassifier(objective='multi:softmax',num_class=3))])))
# Visualização da matríz de correlação entre os dados - Simples

def correlation(df):

    corr = df.corr()

    _ , ax = plt.subplots( figsize =( 12 , 10 ) )

    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )

    _ = sns.heatmap(corr, cmap = cmap, square=True, cbar_kws={ 'shrink' : .9 }, ax=ax, annot = True, annot_kws = {'fontsize' : 8 })

    

# Visualização da matríz de correlação entre os dados - Completa

def heatmap(df, title, col):

    fig, ax = plt.subplots(1,1, figsize = (15,6))



    hm = sns.heatmap(df.iloc[:,col:].corr(),

                     ax = ax, cmap = 'coolwarm',

                     annot = True, fmt = '.2f',

                     linewidths = 0.05)

    fig.subplots_adjust(top=0.93)

    fig.suptitle(title, fontsize=12, fontweight='bold')

    

# Histograma com a comparação da variáveis de teste e treino

def comparefeatures(dtrain, dtest, col):

    plt.figure(figsize=(26, 16))

    

    for i, col in enumerate(dtrain.columns[col:]):

        ax = plt.subplot(3, 4, i + 1)

        sns.distplot(dtrain[col], bins=100, label='train')

        sns.distplot(dtest[col], bins=100, label='test')

        ax.legend()  



# Detectando valores incompletos, nulos e erros

def missing_values_table(df, clean = False):

    mis_val = df.isnull().sum()        

    mis_val_percent = 100 * df.isnull().sum() / len(df)        

    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)        

    mis_val_table_ren_columns = mis_val_table.rename(

    columns = {0 : 'Missing Values', 1 : '% of Total Values'})        

    mis_val_table_ren_columns = mis_val_table_ren_columns[

        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(

    '% of Total Values', ascending=False).round(1)



    print ("O dataframe tem " + str(df.shape[1]) + " colunas.\n"      

        "Há " + str(mis_val_table_ren_columns.shape[0]) + " colunas que possuem valores ausentes.")



    if clean and mis_val_table_ren_columns.shape[0] > 0:

        df.fillna(0, inplace = True)

        df.replace(-np.inf, 0, inplace = True)

        df.replace(np.inf, 0, inplace = True)

        mis_val_table_ren_columns = 0

        print("O dataframe foi limpo.")

            

    return mis_val_table_ren_columns

    

# Apresenando subgráficos para comparação das colunas de um dataset

def plot_subplots(df, columns):

    plt.subplots(figsize=(18,15))

    length=len(columns)



    for i,j in zip(columns,range(length)):

        plt.subplot((length/2),3,j+1)

        plt.subplots_adjust(wspace=0.2,hspace=0.5)

        df[i].hist(bins=20,edgecolor='black')

        plt.title(i)

    plt.show()



# Apresentando subgráficos de boxplot para verifição do outliers nos dados

def plot_boxplots(df, layout):

    df.plot(kind='box', subplots=True, layout=layout, figsize=(18,15), sharex=False, sharey=False)

    plt.show()

    

# Apresentando a correlação

def correlation(df):

    corr = df.corr()

    _ , ax = plt.subplots( figsize =( 12 , 10 ) )

    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )

    _ = sns.heatmap(corr, cmap = cmap, square=True, cbar_kws={ 'shrink' : .9 }, ax=ax, annot = True, annot_kws = {'fontsize' : 8 })



# From: Code Snippet For Visualizing Series Id by @shaz13

def plotSeries(series_id, X, y, col = 3):

    style.use('ggplot')

    plt.figure(figsize=(28, 16))

    print(y[y['series_id'] == series_id]['surface'].values[0].title())

    

    series_dict = {}

    for series in (X['series_id'].unique()):

        series_dict[series] = X[X['series_id'] == series] 

    

    for i, col in enumerate(series_dict[series_id].columns[col:]):

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



# Complementando com a série de Fourier os datasets de densidade

def data_denoised(target, data):

    denoised = pd.DataFrame()

    

    for col in data.columns:

        if col in ['row_id','series_id','measurement_number']:

            denoised[col] = data.groupby([target])[col]

        else:

            # Apply filter_signal function to the data in each series

            denoised_data = data.groupby([target])[col].apply(lambda x: filter_signal(x))



            # Assign the denoised data back to X_denoised

            list_denoised_data = []

            for arr in denoised_data:

                for val in arr:

                    list_denoised_data.append(val)



            denoised[col] = list_denoised_data



    return denoised   



#https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles

#quaternion to eular

def quaternion_to_euler(qx,qy,qz,qw):

    import math

    # roll (x-axis rotation)

    sinr_cosp = +2.0 * (qw * qx + qy + qz)

    cosr_cosp = +1.0 - 2.0 * (qx * qx + qy * qy)

    roll = math.atan2(sinr_cosp, cosr_cosp)

    

    # pitch (y-axis rotation)

    sinp = +2.0 * (qw * qy - qz * qx)

    if(math.fabs(sinp) >= 1):

        pitch = copysign(M_PI/2, sinp)

    else:

        pitch = math.asin(sinp)

        

    # yaw (z-axis rotation)

    siny_cosp = +2.0 * (qw * qz + qx * qy)

    cosy_cosp = +1.0 - 2.0 * (qy * qy + qz * qz)

    yaw = math.atan2(siny_cosp, cosy_cosp)

    

    return roll, pitch, yaw            



#eular angle

def eular_angle(data):

    x, y, z, w = data['orientation_X'].tolist(), data['orientation_Y'].tolist(), data['orientation_Z'].tolist(), data['orientation_W'].tolist()

    nx, ny, nz = [], [], []

    for i in range(len(x)):

        xx, yy, zz = quaternion_to_euler(x[i], y[i], z[i], w[i])

        nx.append(xx)

        ny.append(yy)

        nz.append(zz)

    

    data['euler_x'] = nx

    data['euler_y'] = ny

    data['euler_z'] = nz

    

    return data



# from @theoviel at https://www.kaggle.com/theoviel/fast-fourier-transform-denoising

def filter_signal(signal, threshold=1e3):

    fourier = rfft(signal)

    frequencies = rfftfreq(signal.size, d=20e-3/signal.size)

    fourier[frequencies > threshold] = 0

    return irfft(fourier)



# Excluíndo valores duplicados

def drop_duplicate(df, columns = None):

    

    if columns:

        df.drop(columns = columns, inplace = True) 

    df.drop_duplicates()

    df.info()

    return df



# Convertendo uma coluna em valor categórico

def dummie_transform(df, column):

    df[column] = categorie.fit_transform(df[column].astype(str))

    return df



# Revertendo uma coluna de valor categórico

def dummie_untransform(array):

    #df = categorie.inverse_transform(array)

    df = categorie.inverse_transform(array.argmax(axis=1))

    return df



# Transformação e ajuste das escalas dos dados

def scaler_transform(df, col, default = True):

   

    for col in df.columns[col:]:

        if default:

            scaler = StandardScaler()

        else:

            scaler = MinMaxScaler(feature_range=(0, 1))



        df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))

        

    return df



# Feature Engineer

def feature_engineer(data):

    df = pd.DataFrame()

    

    for col in data.columns:

        if col in ['row_id','series_id','measurement_number']:

            continue

            

        df[col + '_mean'] = data.groupby(['series_id'])[col].mean()

        df[col + '_median'] = data.groupby(['series_id'])[col].median()

        df[col + '_max'] = data.groupby(['series_id'])[col].max()

        df[col + '_min'] = data.groupby(['series_id'])[col].min()

        df[col + '_std'] = data.groupby(['series_id'])[col].std()

        df[col + '_range'] = df[col + '_max'] - df[col + '_min']

        df[col + '_maxtoMin'] = df[col + '_max'] / df[col + '_min']

        df[col + '_mad'] = data.groupby(['series_id'])[col].apply(lambda x: np.median(np.abs(np.diff(x))))

        df[col + '_abs_max'] = data.groupby(['series_id'])[col].apply(lambda x: np.max(np.abs(x)))

        df[col + '_abs_min'] = data.groupby(['series_id'])[col].apply(lambda x: np.min(np.abs(x)))

        df[col + '_abs_avg'] = (df[col + '_abs_min'] + df[col + '_abs_max'])/2

    return df    



# Salvando o modelo treinado

def write_pickle(model, file):

    file = file + ".pkl"

    with open(file, mode='wb') as f:

        pickle.dump(model, f)

    return file



# Recuperando um valor treinado

def read_pickle(file):

    file = file + ".pkl"

    with open(file, mode='rb') as f:

        model = pickle.load(f)    

    return model



# Executando o modelo para comparação

def prediction(models, Xtrain, ytrain, iterable = 10):

    # make the model for train data

    print("Início do treinamento do modelo: " + name + "\n")

    

    X_train, X_test, y_train, y_test = train_test_split(Xtrain, ytrain, train_size=0.70, test_size=0.30, random_state=2011, shuffle=True)

    

    for name, model in models:

        model.fit(X_train, y_train)



        print("....Writing model in pickle")

        file = write_pickle(model, name)        



        # make predictions for test data

        predict = model.predict(X_test)



        # evaluate predictions

        score = model.score(X_train, y_train)  

        print("....Model Score: %.2f%%" % (score * 100.0))

        

        # evaluate predictions

        accuracy = accuracy_score(y_test, predict)    

        print("....Accuracy Score: %.2f%%" % (accuracy * 100.0))



        # Implementing your own scoring

        scores = cross_val_score(model, X_train, y_train, cv=iterable)

        print("....Cross Validation Score: %.2f%%" % (scores.mean() * 100.0))



        results.append({'Name': name, 'Model': model, 'Predict': predict, 'Accuracy': accuracy, 'Score': scores, 'File': file, 'Value': scores.mean()}, ignore_index=True)

        print("Modelo treinado!")

    

    print("\n Término do treinamento dos modelos")



    # Limpeza da Memória

    gc.collect()    

    

    # Apresentando o resultado

    results

    return predict



# Executando a lista de modelos para encontra o melhor resultado

def all_prediction(models, Xtrain, ytrain, iterable = 10):

    # make the model for train data    

    print("Início do treinamento dos modelos \n")

    

    X_train, X_test, y_train, y_test = train_test_split(Xtrain, ytrain, train_size=0.70, test_size=0.30, random_state=2011, shuffle=True)



    for name, model in models:

        print("....Treinando o Modelo " + name)

        

        model.fit(X_train, y_train)

        

        print("....Writing model in pickle")

        file = write_pickle(model, name)        

        

        # make predictions for test data

        predict = model.predict(X_test)



        # evaluate predictions

        score = model.score(X_train, y_train)  

        print("....Model Score: %.2f%%" % (score * 100.0))

        

        # evaluate predictions

        accuracy = accuracy_score(y_test, predict)

        print("....Accuracy Score: %.2f%%" % (accuracy * 100.0))

        

        # Implementing your own scoring

        scores = cross_val_score(model, X_train, y_train['surface'], cv=iterable)

        print("....Cross Validation Score: %.2f%%" % (scores.mean() * 100.0))

       

        results.append({'Name:': name, 'Model': model, 'Predict': predict, 'Accuracy': accuracy, 'Score': scores, 'File': file, 'Value': scores.mean()}, ignore_index=True)

        print("....Modelo " + name + " treinado!\n")



        # Limpeza da Memória

        gc.collect()

        

    print("\n Término do treinamento dos modelos")



    # Apresentando o resultado

    results

    return results

    

def split_prediction(models, X_train, y_train, X_test, iterable = 10):

    # make the model for train data    

    print("Início do treinamento dos modelos \n")



    folds = StratifiedKFold(n_splits=iterable, shuffle=True, random_state=2011)

    predicted = np.zeros((X_test.shape[0],9))

    measured = np.zeros((X_train.shape[0]))

    score = 0

    

    for name, model in models:

        

        for times, (trn_idx, val_idx) in enumerate(folds.split(X_data.values, y_train.values)):

            model.fit(X_data.iloc[trn_idx], y_train[trn_idx])

            measured[val_idx] = model.predict(X_data.iloc[val_idx])

            predict += model.predict_proba(X_test) / folds.n_splits

            score += model.score(X_data.iloc[val_idx], y_train[val_idx])



            print("....Fold: {} score: {}".format(times, model.score(X_data.iloc[val_idx], y_train[val_idx])))



            # Implementing your own scoring

            scores += cross_val_score(model, X_train, y_train, cv=iterable)

            print("....Cross Validation Score: %.2f%%" % (scores.mean() * 100.0))



        results.append({'Name:': name, 'Model': model, 'Predict': predict, 'Accuracy': None, 'Score': (scores / folds.n_splits), 'File': file, 'Value': scores.mean()}, ignore_index=True)

        print("....Modelo " + name + " treinado!\n")



        print('\n Average score', score / folds.n_splits)

        

        # Limpeza da Memória

        gc.collect()



    print("\n Término do treinamento dos modelos")

    # Apresentando o resultado

    results

    return results

# Carregando os dados de treino e qualificando as variáveis por tipo para padronizar



fX_train = pd.read_csv(dirname + '/X_treino.csv', low_memory=False

                    ,dtype = {'series_id': np.int16,'measurement_number': np.int16

                              ,'orientation_X': np.float32,'orientation_X': np.float32

                              ,'orientation_Y': np.float32,'orientation_Z': np.float32

                              ,'orientation_W': np.float32,'angular_velocity_X': np.float32

                              ,'angular_velocity_Y': np.float32,'angular_velocity_Z': np.float32

                              ,'linear_acceleration_X': np.float32,'linear_acceleration_Y': np.float32

                              ,'linear_acceleration_Z': np.float32})



fy_train = pd.read_csv(dirname + '/y_treino.csv', low_memory=False

                      ,dtype = {'series_id': np.int16,'group_id': np.int16, 'surface': np.str})



fX_test = pd.read_csv(dirname + '/X_teste.csv', low_memory=False

                    ,dtype = {'series_id': np.int16,'measurement_number': np.int16

                              ,'orientation_X': np.float32,'orientation_X': np.float32

                              ,'orientation_Y': np.float32,'orientation_Z': np.float32

                              ,'orientation_W': np.float32,'angular_velocity_X': np.float32

                              ,'angular_velocity_Y': np.float32,'angular_velocity_Z': np.float32

                              ,'linear_acceleration_X': np.float32,'linear_acceleration_Y': np.float32

                              ,'linear_acceleration_Z': np.float32})



f_submission = pd.read_csv(dirname + '/sample_submission.csv', low_memory=False

                      ,dtype = {'series_id': np.int16, 'surface': np.str})
print('\nCaracterísticas do dado de teste')

print('    Quantidade de dados: {0}\n    Quantidade de características: {1}'.format(fX_test.shape[0], fX_test.shape[1]))



print('Características do dado de treino')

print('    Quantidade de dados: {0}\n    Quantidade de características: {1}'.format(fX_train.shape[0], fX_train.shape[1]))



print('\nCaracterísticas do dado alvo')

print('    Quantidade de dados: {0}\n    Quantidade de características: {1}'.format(fy_train.shape[0], fy_train.shape[1]))



print('\nCaracterísticas do dado de submissão')

print('    Quantidade de dados: {0}\n    Quantidade de características: {1}'.format(f_submission.shape[0], f_submission.shape[1]))
fX_test.describe(include='all').T
fX_train.describe(include='all').T
fy_train.describe(include='all').T
f_submission.describe(include='all').T
#sns.countplot(y='surface',data = y_train)

trace0 = go.Pie(

    labels = fy_train['surface'].value_counts().index,

    values = fy_train['surface'].value_counts().values,

    domain = {'x':[0.55,1]})



trace1 = go.Bar(

    x = fy_train['surface'].value_counts().index,

    y = fy_train['surface'].value_counts().values

    )



data = [trace0, trace1]

layout = go.Layout(

    title = 'Distribuição da frequência por tipo de superfície',

    xaxis = dict(domain = [0,.50]))



fig = go.Figure(data = data, layout = layout)

py.iplot(fig)
heatmap(fX_test, 'Correlação entre as variáveis independentes de teste', 3)
heatmap(fX_train, 'Correlação entre as variáveis independentes de treino', 3)
X_train
fig = plt.figure(figsize=(15,15))



ax = fig.add_subplot(1)

ax.set_title('Distribution of Orientation_X,Y,Z,W',

             fontsize=14, 

             fontweight='bold')

X_train.iloc[:,1:5].boxplot()



ax = fig.add_subplot(2)

ax.set_title('Distribution of Angular_Velocity_X,Y,Z',

             fontsize=14, 

             fontweight='bold')

X_train.iloc[:,5:8].boxplot()



ax = fig.add_subplot(3)

ax.set_title('Distribution of linear_accelaration_X,Y,Z',

             fontsize=14, 

             fontweight='bold')

X_train.iloc[:,8:11].boxplot()
comparefeatures(fX_train, fX_test, 3)
# Checando valores missing do dataset de teste

missing_values_table(fX_test)
# Checando valores missing do dataset de treino

missing_values_table(fX_train)
# Checando valores missing do dataset de resultado

missing_values_table(fy_train)
X_test = drop_duplicate(fX_test)
X_train = drop_duplicate(fX_train)
y_train = drop_duplicate(fy_train)
# Juntando os dados Treino

df_merge = pd.merge(fX_train, fy_train, how="inner", on="series_id")

df_merge.describe(include='all').T
classes = fy_train.surface.value_counts()

classes
query = """ SELECT t.series_id, 

                   t.orientation_X, t.orientation_Y, t.orientation_Z, t.orientation_W, 

                   t.angular_velocity_X, t.angular_velocity_Y, t.angular_velocity_Z, 

                   t.linear_acceleration_X, t.linear_acceleration_Y, t.linear_acceleration_Z

            FROM X_test as t

            INNER JOIN (SELECT series_id, MIN(measurement_number) as measurement_number, COUNT() as qtde

                        FROM X_test

                        GROUP BY series_id) x

                ON t.series_id = x.series_id and t.measurement_number = x.measurement_number"""



X_test_min = ps.sqldf(query, locals())

X_test_min
query = """ SELECT t.series_id, 

                   t.orientation_X, t.orientation_Y, t.orientation_Z, t.orientation_W, 

                   t.angular_velocity_X, t.angular_velocity_Y, t.angular_velocity_Z, 

                   t.linear_acceleration_X, t.linear_acceleration_Y, t.linear_acceleration_Z

            FROM X_test as t

            INNER JOIN (SELECT series_id, MAX(measurement_number) as measurement_number, COUNT() as qtde

                        FROM X_test

                        GROUP BY series_id) x

                ON t.series_id = x.series_id and t.measurement_number = x.measurement_number"""



X_test_max = ps.sqldf(query, locals())

X_test_max
query = """ SELECT t.series_id, 

                   t.surface

            FROM df_merge as t

            INNER JOIN (SELECT series_id, MIN(measurement_number) as measurement_number

                        FROM df_merge

                        GROUP BY series_id) x

                ON t.series_id = x.series_id and t.measurement_number = x.measurement_number"""



y_train_min = ps.sqldf(query, locals())

y_train_min
query = """ SELECT t.series_id, 

                   t.surface

            FROM df_merge as t

            INNER JOIN (SELECT series_id, MAX(measurement_number) as measurement_number

                        FROM df_merge

                        GROUP BY series_id) x

                ON t.series_id = x.series_id and t.measurement_number = x.measurement_number"""



y_train_max = ps.sqldf(query, locals())

y_train_max
X_test = X_test.drop(["row_id", "measurement_number"], axis=1)

X_test
y_train = df_merge[["series_id", "surface"]].copy()

y_train
X_train = df_merge.copy().drop(["row_id", "measurement_number", "group_id", "surface"], axis=1)

X_train
X_test_scaler = scaler_transform(X_test, 3, default = True)

X_test_scaler_min = scaler_transform(X_test_min, 3, default = True)

X_test_scaler_max = scaler_transform(X_test_max, 3, default = True)
X_train_scaler = scaler_transform(X_train, 3, default = True)
plotSeries(10, X_train_scaler, y_train)
comparefeatures(X_test, X_test_scaler, 3)
comparefeatures(X_train, X_train_scaler, 3)
#X_test_denoised = data_denoised1("series_id", X_test_scaler)

#X_test_denoised
#X_test_denoised = data_denoised("series_id", X_test_scaler)

#X_test_denoised_min = data_denoised("series_id", X_test_scaler_min)

#X_test_denoised_max = data_denoised("series_id", X_test_scaler_max)
#X_train_denoised = data_denoised("series_id", X_train_scaler)
#plotSeries(10, X_train_denoised, y_train)
#comparefeatures(X_test, X_test_denoised, 3)
#comparefeatures(X_train, X_train_denoised, 3)
X_test_eular = eular_angle(X_test_scaler)

X_test_eular_min = eular_angle(X_test_scaler_min)

X_test_eular_max = eular_angle(X_test_scaler_max)
X_train_eular = eular_angle(X_train_scaler)
plotSeries(10, X_train_eular, y_train)
comparefeatures(X_test, X_test_eular, 3)
comparefeatures(X_train, X_train_eular, 3)
X_test_engineer = feature_engineer(X_test)

X_test_engineer
X_train_engineer = feature_engineer(X_train)

X_train_engineer
y_train = dummie_transform(y_train, "surface")

y_train.head(5)
def prediction1(models, Xtrain, ytrain, iterable = 10):

    print("Iniciando o modelo")

    

    X_train, X_test, y_train, y_test = train_test_split(Xtrain, ytrain, train_size=0.70, test_size=0.30, random_state=2011, shuffle=True)

    

    for name, model in models:

        model.fit(X_train, y_train)



        print("....Writing model in pickle")

        file = write_pickle(model, name)        



        # make predictions for test data

        predict = model.predict(X_test)



        # evaluate predictions

        score = model.score(X_train, y_train)  

        print("....Model Score: %.2f%%" % (score * 100.0))

        

        # evaluate predictions

        accuracy = accuracy_score(y_test, predict)    

        print("....Accuracy Score: %.2f%%" % (accuracy * 100.0))



        # Implementing your own scoring

        scores = cross_val_score(model, X_train, y_train, cv=iterable)

        print("....Cross Validation Score: %.2f%%" % (scores.mean() * 100.0))



        results.append({'Name': name, 'Model': model, 'Predict': predict, 'Accuracy': accuracy, 'Score': scores, 'File': file, 'Value': scores.mean()}, ignore_index=True)

        print("Modelo treinado!")

    

    print("\n Término do treinamento dos modelos")



    # Limpeza da Memória

    gc.collect()    

    

    # Apresentando o resultado

    results

    return predict
models1 = []

models1.append(("RandomForest_entropy",RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=10, random_state=0, max_features=None)))



pool = mp.Pool(mp.cpu_count())



results = pool.map(prediction1, [models1, X_train, y_train['surface'], 10])



pool.close()

importances = model.feature_importances_

std = np.std([tree.feature_importances_ for tree in model.estimators_], axis = 0)

indices = np.argsort(importances)[::-1]
feature_importances = pd.DataFrame(importances, index = X_data.columns, columns = ['importance'])

feature_importances.sort_values('importance', ascending = False)

feature_importances.head(20)
less_important_features = feature_importances.loc[feature_importances['importance'] < 0.0025]

print('There are {0} features their importance value is less then 0.0025'.format(less_important_features.shape[0]))
#Remove less important features from train and test set.

for i, col in enumerate(less_important_features.index):

    X_data = X_data.drop(columns = [col], axis = 1)

    X_test = X_test.drop(columns = [col], axis = 1)

    

X_data.shape, X_test.shape
predicted = np.zeros((X_test.shape[0],9))

measured= np.zeros((X_data.shape[0]))

score = 0



for times, (trn_idx, val_idx) in enumerate(folds.split(X_data.values, y_train['surface'].values)):

    model = RandomForestClassifier(n_estimators=700, n_jobs = -1)

    model.fit(X_data.iloc[trn_idx], y_train['surface'][trn_idx])

    measured[val_idx] = model.predict(X_data.iloc[val_idx])

    predicted += model.predict_proba(X_test) / folds.n_splits

    score += model.score(X_data.iloc[val_idx], y_train['surface'][val_idx])

    

    print("Fold: {} score: {}".format(times,model.score(X_data.iloc[val_idx], y_train['surface'][val_idx])))

    gc.collect()

    

print('\n Average score', score / folds.n_splits)
# Treinando o modelo com base em uma lista de modelos para comparação

def prediction1(model, X_train, y_train, X_test, iterable = 10):

    # make the model for train data

    name = type(model).__name__



    print("Início do treinamento do modelo: " + name + "\n")

    X_train, X_test, y_train, y_test = train_test_split(sX_train, sy_train, train_size=0.70, test_size=0.30, random_state=2011, shuffle=True)

    

    for name, model in models:

        model.fit(X_train, y_train)



        print("....Writing model in pickle")

        file = write_pickle(model, name)        



        # make predictions for test data

        predict = model.predict(X_test)



        # evaluate predictions

        score = model.score(X_train, y_train)  

        print("....Model Score: %.2f%%" % (score * 100.0))

        

        # evaluate predictions

        accuracy = accuracy_score(y_test, predict)    

        print("....Accuracy Score: %.2f%%" % (accuracy * 100.0))



        # Implementing your own scoring

        scores = cross_val_score(model, X_train, y_train, cv=iterable)

        print("....Cross Validation Score: %.2f%%" % (scores.mean() * 100.0))



        results.append({'Name': name, 'Model': model, 'Predict': predict, 'Accuracy': accuracy, 'Score': scores, 'File': file, 'Value': scores.mean()}, ignore_index=True)

        print("Modelo treinado!")

    

    print("\n Término do treinamento dos modelos")



    # Limpeza da Memória

    gc.collect()    

    

    # Apresentando o resultado

    results

    return predict
# make the model for train data

model = RandomForestClassifier(criterion='entropy')

predicted = prediction1(model, X_data, y_train['surface'], X_test, iterable = 10)
_submission = submission.copy()

filename = r'submission_2.csv'
_submission['surface'] = dummie_untransform(predicted)

_submission.to_csv(filename, index=False)

_submission.head(10)



print('Saved file: ' + filename)



FileLink(filename)
columns = ['row_id', 'series_id']

df_test = drop_duplicate(df_test, columns)
columns = ['series_id', 'row_id', 'group_id']

df_train = drop_duplicate(df_train, columns)
# Análise dos dados - Variaveis independentes de teste

columns=df_test.columns[1:10]

plot_subplots(df_test, columns)
# Gráfico Boxplot - Teste

plot_boxplots(df_test, (3,4))
# Análise dos dados - Variaveis independentes de treino

columns=df_train.columns[1:10]

plot_subplots(df_train, columns)
# Gráfico Boxplot - Teste

plot_boxplots(df_train, (3,4))
df_train['surface'].value_counts(normalize=True).plot(kind='bar', figsize=(12,8))

plt.title('Superficies - Surface')

plt.xlabel('Superfície')

plt.ylabel('Frequência')

plt.show()
## Limpeza da Memória

gc.collect()
correlation(df_train)
df_train = dummie_transform(df_train, "surface")

df_train.head(5)
# Dataset de treino

X_train = df_train.drop(['surface', 'surface_'],axis=1)

X_train = scaler_transform(X_train)

y_train = df_train['surface_']
# Dataset de teste

X_test = scaler_transform(df_test)
param = {

        'num_leaves': 7,

        'max_bin': 119,

        'min_data_in_leaf': 6,

        'learning_rate': 0.03,

        'min_sum_hessian_in_leaf': 0.00245,

        'bagging_fraction': 1.0, 

        'bagging_freq': 5, 

        'feature_fraction': 0.05,

        'lambda_l1': 4.972,

        'lambda_l2': 2.276,

        'min_gain_to_split': 0.65,

        'max_depth': 14,

        'save_binary': True,

        'seed': 1337,

        'feature_fraction_seed': 1337,

        'bagging_seed': 1337,

        'drop_seed': 1337,

        'data_random_seed': 1337,

        'objective': 'binary',

        'boosting_type': 'gbdt',

        'verbose': 1,

        'metric': 'auc',

        'is_unbalance': True,

        'boost_from_average': False,

        'device': 'gpu',

        'gpu_platform_id': 0,

        'gpu_device_id': 0

    }



models.append(("LogisticRegression",LogisticRegression(solver='lbfgs', multi_class='multinomial')))

models.append(("Ridge",Ridge()))

models.append(("SVC",SVC()))

models.append(("LinearSVC",LinearSVC()))

models.append(("KNeighbors",KNeighborsClassifier()))

models.append(("DecisionTree",DecisionTreeClassifier()))

models.append(("ExtraTrees",ExtraTreesClassifier()))

models.append(("RandomForest",RandomForestClassifier()))

models.append(("RandomForest_entropy",RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=10, random_state=0, max_features=None)))

models.append(("RandomForest_gini",RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=10, random_state=0, max_features=None)))

models.append(("MLPClassifier",MLPClassifier(solver='lbfgs', random_state=0)))



#models.append(("LightGBM", lgb.train(param, xg_train, 5000, valid_sets = [xg_valid], verbose_eval=500, early_stopping_rounds = 250)))



models.append(("XGBoost",XGBClassifier()))

models.append(("XGBoost_XGB",xgb.XGBClassifier(n_estimators=100, max_depth=8, learning_rate=0.1, subsample=0.5)))

models.append(("XGBoost_Pipeline",Pipeline(steps=[('xgboost', xgb.XGBClassifier(objective='multi:softmax',num_class=3))])))
# X_train, X_test, y_train, y_test = train_test_split(sX_train, sy_train, train_size=0.75, test_size=0.25, random_state=2011, shuffle=True)
## Limpeza da Memória

gc.collect()
all_prediction(models)
# make the model for train data

model = LogisticRegression(multi_class='multinomial')

prediction(model, iterable = 10)
results
# make the model for train data

model = RandomForestClassifier(criterion='entropy')

prediction(model, iterable = 10)
# make the model for train data

model = Ridge()

df_results = prediction(model, df_results, iterable = 10)
# make the model for train data

model = KNeighborsClassifier()

df_results = prediction(model, df_results, iterable = 10)
xg_train = xgb.DMatrix(X_train, label=y_train)

xg_test = xgb.DMatrix(X_test, label=y_test)

xg_train.save_binary('train.buffer')

xg_test.save_binary('train.buffer')



# use softmax multi-class classification

param['objective'] = 'multi:softmax'

param['silent'] = 1 # cleans up the output

param['num_class'] = 3 # number of classes in target label



xg_list = [(xg_train, 'train'), (xg_test, 'test')]

num_round = 30
# make the model for train data

model = XGBClassifier()

df_results = prediction(model, df_results, iterable = 10)
# creating a confusion matrix 

cm = confusion_matrix(y_test, pred5) 
#Train the XGboost Model for Classification

model = xgb.XGBClassifier(n_estimators=100, max_depth=8, learning_rate=0.1, subsample=0.5)

df_results = prediction(model, df_results, iterable = 10)
#Train the XGboost Model for Classification

name = "XGB_v3"

model = Pipeline(steps=[('xgboost', xgb.XGBClassifier(objective='multi:softmax',num_class=3))])

prediction(model, name, iterable = 10)
#Train the XGboost Model for Classification

name = "XGB_v4"

model = xgb.train(param, xg_train, num_round, watchlist)

prediction(model, name, iterable = 10)
# make predictions for test data

y_pred81 = model8.predict(xg_train)

pred81 = [round(value) for value in y_pred81]

print('Train accuracy score:',accuracy_score(y_train, y_pred81))



# evaluate predictions

accuracy8 = accuracy_score(y_train, pred8)

print("Accuracy: %.2f%%" % (accuracy8 * 100.0))



# Implementing your own scoring

scores8 = cross_val_score(model8, X_train, y_train, cv=10)

print('Accuracy for XGB v4 Classifier : ', scores8.mean())



y_pred82 = model8.predict(xg_test)

pred82 = [round(value) for value in y_pred82]

print('Test accuracy score:',accuracy_score(y_test,y_pred82))
# read the file pickle to model

_model = pickle.load(open("RandomForestClassifier.pkl","rb"))



# Fazendo as previsoes de surface no dataset de teste

prediction = _model.predict(X_test) 



# Voltando a transformacao da variavel target em formato texto

result = dummie_untransform(prediction)
#Gerando Arquivo de Submissao

submission = pd.DataFrame({

    "series_id": dfx_test.series_id, 

    "surface": result

})

submission = submission.drop_duplicates()

submission
submission.groupby(['series_id','surface'])['series_id'].sum()
# Usei essa excelente ideia do Rodrigo Lima de Oliveira

# Executando query para identificar as superficies com maiores quantidade, para fazer o submit

query = """ SELECT x.series_id, x.surface, MAX(x.qtde) maior

           FROM (SELECT series_id, surface, count() as qtde

                   FROM submission

                  GROUP BY series_id, surface) x

          GROUP BY x.series_id"""



submission = ps.sqldf(query, locals())

submission = submission.drop(['maior'],axis=1)



submission
#Convert DataFrame to a csv file that can be uploaded

filename = r'submission.csv'

submission.to_csv(filename, index=False)



print('Saved file: ' + filename)



FileLink(filename)