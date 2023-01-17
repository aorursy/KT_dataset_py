# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Carregando as bibliotecas gráficas para visualização dos dados

from matplotlib import pyplot as plt

import seaborn as sns



sns.set()

%matplotlib inline

%config InlineBackend.figure_format = 'retina'
# Carregando os dados de treino e qualificando as variáveis por tipo para padronizar



dfx_train = pd.read_csv('/kaggle/input/X_treino.csv', low_memory=False

                    ,dtype = {'series_id': np.int16,'measurement_number': np.int16

                              ,'orientation_X': np.float32,'orientation_X': np.float32

                              ,'orientation_Y': np.float32,'orientation_Z': np.float32

                              ,'orientation_W': np.float32,'angular_velocity_X': np.float32

                              ,'angular_velocity_Y': np.float32,'angular_velocity_Z': np.float32

                              ,'linear_acceleration_X': np.float32,'linear_acceleration_Y': np.float32

                              ,'linear_acceleration_Z': np.float32})



dfy_train = pd.read_csv('/kaggle/input/y_treino.csv', low_memory=False

                      ,dtype = {'series_id': np.int16,'group_id': np.int16, 'surface': np.str})



dfx_test = pd.read_csv('/kaggle/input/X_teste.csv', low_memory=False

                    ,dtype = {'series_id': np.int16,'measurement_number': np.int16

                              ,'orientation_X': np.float32,'orientation_X': np.float32

                              ,'orientation_Y': np.float32,'orientation_Z': np.float32

                              ,'orientation_W': np.float32,'angular_velocity_X': np.float32

                              ,'angular_velocity_Y': np.float32,'angular_velocity_Z': np.float32

                              ,'linear_acceleration_X': np.float32,'linear_acceleration_Y': np.float32

                              ,'linear_acceleration_Z': np.float32})

# Resultado da importação dos dados

dfx_train.shape, dfy_train.shape, dfx_test.shape
dfx_train.head(5)
dfy_train.head(5)
dfx_test.head(5)
# Juntando os dados Treino

df_train = pd.merge(dfx_train, dfy_train, how="left", on="series_id")

df_train.describe()
df_train.head(5)
# Gerando do dataset de teste

df_test = dfx_test.copy()

df_test.describe()
df_test.info()
df_train.info()
def missing_values_table(df):

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

        

        return mis_val_table_ren_columns
missing_values_table(df_train)
missing_values_table(df_test)
df_test.drop(columns = ['row_id','measurement_number'], inplace = True) 

df_test.drop_duplicates()

df_test.info()
df_train.drop(columns = ['row_id','group_id','measurement_number'], inplace = True) 

df_train.drop_duplicates()

df_train.info()
# Análise dos dados - Variaveis independentes de teste

columns=df_test.columns[1:11]

plt.subplots(figsize=(18,15))

length=len(columns)



for i,j in zip(columns,range(length)):

    plt.subplot((length/2),3,j+1)

    plt.subplots_adjust(wspace=0.2,hspace=0.5)

    df_test[i].hist(bins=20,edgecolor='black')

    plt.title(i)

plt.show()
# Análise dos dados - Variaveis independentes de teste

columns=df_train.columns[1:11]

plt.subplots(figsize=(18,15))

length=len(columns)



for i,j in zip(columns,range(length)):

    plt.subplot((length/2),3,j+1)

    plt.subplots_adjust(wspace=0.2,hspace=0.5)

    df_train[i].hist(bins=20,edgecolor='black')

    plt.title(i)

plt.show()
df_train['surface'].value_counts().plot(kind='bar', figsize=(12,8))

plt.title('Superficies - Surface')

plt.xlabel('Superfície')

plt.ylabel('Frequência')

plt.show()
import gc



gc.collect()
df = df_train.drop(['series_id'],axis=1)

corr = df.corr()

_ , ax = plt.subplots( figsize =( 12 , 10 ) )

cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )

_ = sns.heatmap(corr, cmap = cmap, square=True, cbar_kws={ 'shrink' : .9 }, ax=ax, annot = True, annot_kws = {'fontsize' : 8 })
from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import Imputer, MinMaxScaler

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder
categorie = preprocessing.LabelEncoder()

df_train["surface_"] = categorie.fit_transform(df_train["surface"].astype(str))
scaler = MinMaxScaler(feature_range=(0, 1))
df_test.head()
X_test = df_test.drop(['series_id'],axis=1)



scaler.fit(df_test)

X_test = scaler.transform(df_test)
# Verificando o shape apos o split entre feature e target

X_test.shape
X_train = df_train.drop(['surface', 'surface_'],axis=1)

y_train = df_train['surface_']
scaler.fit(X_train)

X_train = scaler.transform(X_train)
# Verificando o shape apos o split entre feature e target

X_train.shape, y_train.shape
from sklearn.metrics import mean_squared_error as mse

from sklearn.metrics import mean_absolute_error as mae

from sklearn.metrics import r2_score, make_scorer



def ToWeight(y):

    w = np.zeros(y.shape, dtype=float)

    ind = y != 0

    w[ind] = 1./(y[ind]**2)

    return w





def rmspe(yhat, y):

    w = ToWeight(y)

    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))

    return rmspe





def rmspe_xg(yhat, y):

    # y = y.values

    y = y.get_label()

    y = np.exp(y) - 1

    yhat = np.exp(yhat) - 1

    w = ToWeight(y)

    rmspe = np.sqrt(np.mean(w * (y - yhat)**2))

    return "rmspe", rmspe
import xgboost as xgb

from hyperopt import hp

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials



def train(index, X, y, hp_selection=False):

    #train_store = train[index]

    #X = train_store[train_store.columns.drop(['series_id', 'measurement_number', 'surface', 'surface_'])]

    #y = train_store['surface_']



    train_size = int(X.shape[0]*.99)

    print(f'Regressor for {index} store\nTraining on {X.shape[0]} samples')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)



    #X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]

    #X_test, y_test = X.iloc[train_size:], y.iloc[train_size:]



    xtrain = xgb.DMatrix(X_train, np.log(y_train.values) + 1)

    xtest = xgb.DMatrix(X_test, np.log(y_test.values) + 1)

    

    if hp_selection:

        def score(params):

            num_round = 200

            model = xgb.train(params, xtrain, num_round, feval=rmspe_xg)

            predictions = model.predict(xtest)

            score = rmspe(y=y_test, yhat=predictions)

            return {'loss': score, 'status': STATUS_OK}

        

        def optimize(trials):

            space = {

                     'n_estimators' : hp.quniform('n_estimators', 1, 1000, 1),

                     'eta' : hp.quniform('eta', 0.2, 0.825, 0.025),

                     'max_depth' : hp.choice('max_depth', np.arange(1, 14, dtype=int)),

                     'min_child_weight' : hp.quniform('min_child_weight', 1, 6, 1),

                     'subsample' : hp.quniform('subsample', 0.7, 1, 0.05),

                     'gamma' : hp.quniform('gamma', 0.5, 1, 0.05),

                     'colsample_bytree' : hp.quniform('colsample_bytree', 0.5, 1, 0.05),

                     'eval_metric': 'rmse',

                     'objective': 'reg:linear',

                     'nthread': 4,

                     'silent' : 1

                     }



            best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=250)

            return best

        

        trials = Trials()

        best_opts = optimize(trials)

        best_opts['silent'] = 1

    else:

        best_opts = {'colsample_bytree': 0.7, 

                  'eta': 0.625, 

                  'gamma': 0.8, 

                  'max_depth': 6,

                  'eval_metric': 'rmse',

                  'min_child_weight': 6.0, 

                  'n_estimators': 8.0,  # 585

                  'silent': 1,

                  'nthread': 4,

                  'subsample': 0.95}

        

    watchlist = [(xtrain, 'train'), (xtest, 'eval')]

    num_round = 10000

    regressor = xgb.train(best_opts, xtrain, num_round, watchlist, feval=rmspe_xg,

                          verbose_eval=10, early_stopping_rounds=50)

    print("Validating")

    train_probs = regressor.predict(xtest)

    indices = train_probs < 0

    train_probs[indices] = 0

    error = rmspe(np.exp(train_probs) - 1, y_test.values)

    print('error', error)

    regressor = xgb.train(best_opts, xtest, 10, feval=rmspe_xg, xgb_model=regressor)

    return regressor
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier



# Definindo os valores para o número de folds

num_folds = 15

seed = 2011



# Separando os dados em folds

kfold = KFold(num_folds, True, random_state = seed)



# Criando o modelo

modeloCART = DecisionTreeClassifier()



# Cross Validation

resultado = cross_val_score(modeloCART, X_train, y_train, cv = kfold, scoring = 'accuracy')



# Print do resultado

print("Acurácia: %.3f" % (resultado.mean() * 100))



# Treinando o modelo

modeloCART.fit(X_train, y_train)
# Fazendo as previsoes de surface no dataset de teste

predCART = modeloCART.predict(X_test)



# Voltando a transformacao da variavel target em formato texto

surface_pred = categorie.inverse_transform(predCART)
X_test.size()
#Gerando Arquivo de Submissao

submission = pd.DataFrame({

    "series_id": df_test.series_id, 

    "surface": surface_pred

})

submission

#submission = pd.DataFrame(submission).set_index('series_id')

#submission = submission.drop_duplicates()

gb = submission.groupby(['series_id','surface'])

result = gb['surface'].unique()

#result.reset_index(inplace=True)

result

#submission
store_grouped = dict(list(df_train.groupby('series_id')))

test_grouped = dict(list(df_test.groupby('series_id')))
submission = pd.Series(np.zeros(df_test.series_id.shape))

submission.index += 1



for store in test_grouped:

    #test = test_grouped[store].copy()

    ids = df_test['series_id']

    dpred = xgb.DMatrix(df_test[df_test.columns.drop(['series_id'])]) 

    regressor = train(store, X_train, y_train)

    preds = regressor.predict(dpred)

    preds[preds < 0] = 0

    preds = np.exp(preds) - 1

    submission[ids] = preds



submission[closed_store_ids] = 0
# Criando o arquivo de resultados

submission = pd.DataFrame({

    "series_id": test.series_id, 

    "surface": surface_pred

})