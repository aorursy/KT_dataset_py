# Variável para indicar para qual modelo se deseja realizar grid search.

# Por limitações do kernel do kaggle, fazer o grid search apenas de um modelo por vez.

#    "RFR"      -> Random Forest Regressor grid search

#    "XGB"      -> XGBoost Regressor grid search

#    "TPOT"     -> TPOT Regressor

#    "MLPR"     -> MLP Regressor grid search

#    "Default"  -> Executar os modelos com os parâmetros padrão (sem grid search)

#    "Best"     -> Executar os melhores modelos de acordo com grid search anteriores

grid_search = "Best"
import os

import warnings

import gc

import xgboost



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns 

import datetime as dt



from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ShuffleSplit, train_test_split

from sklearn.metrics import make_scorer, mean_absolute_error, r2_score

from xgboost.sklearn import XGBRegressor

from sklearn.svm import LinearSVR



%matplotlib inline

warnings.filterwarnings("ignore")
properties2016 = pd.read_csv('../input/properties_2016.csv', low_memory = False)

properties2017 = pd.read_csv('../input/properties_2017.csv', low_memory = False)
def changeToFloat32(properties):

    for c, dtype in zip(properties.columns, properties.dtypes):

        if dtype == np.float64:

            properties[c] = properties[c].astype(np.float32)
changeToFloat32(properties2016)

changeToFloat32(properties2017)
train2016 = pd.read_csv('../input/train_2016_v2.csv', parse_dates=['transactiondate'], low_memory=False)

train2017 = pd.read_csv('../input/train_2017.csv', parse_dates=['transactiondate'], low_memory=False)
sample_submission = pd.read_csv('../input/sample_submission.csv', low_memory = False)
def add_date_features(df):

    df["transaction_year"] = df["transactiondate"].dt.year

    df["transaction_month"] = (df["transactiondate"].dt.year - 2016)*12 + df["transactiondate"].dt.month

    df["transaction_day"] = df["transactiondate"].dt.day

    df["transaction_quarter"] = (df["transactiondate"].dt.year - 2016)*4 +df["transactiondate"].dt.quarter

    df.drop(["transactiondate"], inplace=True, axis=1)

    return df
dropcols = ['finishedsquarefeet12', 'finishedsquarefeet13'\

            ,'finishedsquarefeet15', 'finishedsquarefeet6'\

            ,'finishedsquarefeet50', 'fullbathcnt', 'calculatedbathnbr']



properties2016.drop(dropcols, axis=1, inplace=True)

properties2017.drop(dropcols, axis=1, inplace=True)
train2016 = add_date_features(train2016)

train2017 = add_date_features(train2017)
train2016 = pd.merge(train2016, properties2016, how = 'left', on = 'parcelid')

train2017 = pd.merge(train2017, properties2017, how = 'left', on = 'parcelid')
train2016.head()
train2017.head()
# Necessário anular os valores das colunas de 'tax' de acordo com as regras da competição

train2017.iloc[:, train2017.columns.str.startswith('tax')] = np.nan
train_df = pd.concat([train2016, train2017], axis = 0)

test_df = pd.merge(sample_submission[['ParcelId']], properties2016.rename(columns = {'parcelid': 'ParcelId'}), how = 'left', on = 'ParcelId')
del properties2016, properties2017, train2016, train2017

gc.collect();
train_df.shape
test_df.shape
def add_new_features(df):

    df['AREA_UTIL_IMOVEL'] = df['calculatedfinishedsquarefeet'] / df['lotsizesquarefeet']

    df['AREA_QUINTAL'] = df['lotsizesquarefeet'] - df['calculatedfinishedsquarefeet']

    df['TOTAL_COMODOS'] = df['bathroomcnt'] + df['bedroomcnt']

    

    import datetime

    now = datetime.datetime.now()

    df['IDADE_IMOVEL'] = now.year - df['yearbuilt']

    

    df['TOTAL_IMPOSTOS'] = df['taxvaluedollarcnt'] + df['taxamount']

    

    #Taxa de impostos total por taxas de uma pesquisa

    df['TAXA_IMPOSTOS'] = df['taxvaluedollarcnt'] / df['taxamount']

       

    #Quantidade de imoveis por municipio

    county_count = df['regionidcounty'].value_counts().to_dict()

    df['IMOVEIS_MUNICIPIO'] = df['regionidcounty'].map(county_count)



    #Quantidade de imoveis por cidade

    city_count = df['regionidcity'].value_counts().to_dict()

    df['IMOVEIS_CIDADE'] = df['regionidcity'].map(city_count)

    

    df['latitude'] = df['latitude'] / 1e7

    df['longitude'] = df['longitude'] / 1e7
add_new_features(train_df)

add_new_features(test_df)
def cat_to_code(df):

    object_type = df.select_dtypes(include=['object']).columns.values

    df[object_type] = df[object_type].astype('category')

    for column in object_type:

        df[column] = df[column].cat.codes
cat_to_code(train_df)

cat_to_code(test_df)
test_df['transactiondate'] = pd.Timestamp('2016-12-01') 

test_df = add_date_features(test_df)

test_df.fillna(-999, inplace=True)
train_df.dropna(thresh=0.70*len(train_df), axis=1, inplace=True)

train_df.shape
train_df.loc[:, (train_df != train_df.iloc[0]).any()].shape
train_df.dropna(inplace=True)
train_df.isnull().values.any()
train_df.shape
#x_train, x_test, y_train, y_test = train_test_split(train_df.drop(['logerror', 'parcelid'], axis=1), train_df.logerror, test_size=1/3, random_state=42)#
#print("x_train: {}, x_test: {}, y_train: {}, y_test: {}".format(x_train.shape, x_test.shape, y_train.shape, y_test.shape))
def linearRegressor(X_train, Y_train, X_test, Y_test):

  regressor = LinearRegression(fit_intercept=True)



  model = regressor.fit(X_train, Y_train)



  pred_LR = regressor.predict(X_test)

  

  resultados(Y_test, pred_LR, "Linear Regressor")



  # Returns the trained model

  return model
def randomForestRegressor(X_train, Y_train, X_test, Y_test):

  # Gerar conjuntos de validação-cruzada para o treinamento de dados

  cv_sets = ShuffleSplit(n_splits=5, test_size = 0.25)



  rfr =  RandomForestRegressor(random_state=42)



  # Numero de arvores no random forest

  n_estimators = [int(x) for x in np.linspace(start = 180, stop = 220, num = 11)]

  # Numero de features para considerar a cada separacao

  max_features = ['auto', 'sqrt']

  # Profundidade maxima da arvore

  max_depth = [int(x) for x in np.linspace(20, 30, num = 6)]

  # Quantidade minima de amostras para se separar um no

  min_samples_split = list(range(4,8))

  # Quantidade minima de amostras requeridas em cada no folha

  min_samples_leaf = list(range(1,4))



  # Grid search anterior

  #n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]

  #max_features = ['auto', 'sqrt']

  #max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]

  #min_samples_split = [2, 5, 10, 15, 100]

  #min_samples_leaf = [1, 2, 5, 10]



  # Create the random grid

  params = {'n_estimators': n_estimators,

                 'max_features': max_features,

                 'max_depth': max_depth,

                 'min_samples_split': min_samples_split,

                 'min_samples_leaf': min_samples_leaf}



  #Transformar 'performance_metric' em uma função de pontuação utilizando 'make_scorer' 

  scoring_fnc = make_scorer(mean_absolute_error)



  # Gerar o objeto de busca em matriz

  grid = RandomizedSearchCV(estimator = rfr, param_distributions = params, scoring=scoring_fnc, 

                            cv=cv_sets, n_iter=100, verbose=2)



  # Ajustar o objeto de busca em matriz com os dados para calcular o modelo ótimo

  grid = grid.fit(X_train, Y_train)



  # Escolher o melhor estimador para predizer os dados de teste

  best_rfr = grid.best_estimator_



  print("Best Estimator: \n{}\n".format(grid.best_estimator_))

  print("Best Parameters: \n{}\n".format(grid.best_params_))



  pred_RFR = best_rfr.predict(X_test)

  

  resultados(Y_test, pred_RFR, "Random Forest Regressor")

   

  # Returns the best trained model

  return best_rfr
def xgb(X_train, Y_train, X_test, Y_test):

  

  cv_sets = ShuffleSplit(n_splits=5, test_size = 0.25)

  

  xgb = XGBRegressor()



  # xgboost params

  params = {

              'objective':['reg:linear'],

              'learning_rate': [.03, .033, .035, .037],

              'max_depth': [5, 6, 7],

              'min_child_weight': [2, 4, 10],

              'silent': [1],

              'subsample': [0.6, 0.7, 0.8],

              'colsample_bytree': [0.6, 0.7, 0.8],

              'n_estimators': [500],

              'base_score': [np.mean(Y_train), 0.5],

              'reg_lambda': [0.3, 0.5, 0.8],

              'eval_metric': ['mae', 'logloss']

              

  }



  # Gerar o objeto de busca em matriz

  grid = RandomizedSearchCV(estimator = xgb, param_distributions = params, scoring='neg_mean_squared_error', 

                            cv=cv_sets, n_iter=100, verbose=2)



  grid = grid.fit(X_train, Y_train)



  # Escolher o melhor estimador para predizer os dados de teste

  best_xgb = grid.best_estimator_



  print("Best Estimator: \n{}\n".format(grid.best_estimator_))

  print("Best Parameters: \n{}\n".format(grid.best_params_))



  pred_xgb = best_xgb.predict(X_test)

  

  resultados(Y_test, pred_xgb, "XGBoost Regressor")



  # Returns the trained model

  return best_xgb
def tpotRegressor(X_train, Y_train, X_test, Y_test):

    from tpot import TPOTRegressor



    tpot = TPOTRegressor(generations=10, population_size=100, scoring='neg_mean_absolute_error', verbosity=2)

    tpot.fit(X_train, Y_train)

    print(tpot.score(X_test, Y_test))

    tpot.export('tpot_zillow_pipeline.py')
def resultados(Y_test, predictions, model_name):



    #calculando o erro de uma árvore de decisão para regressão:

    mae_RFR = mean_absolute_error(predictions, Y_test)

    print ("Erro médio absoluto: {}".format(mae_RFR))



    #Acurácia do modelo

    #r2_RFR = r2_score(predictions, Y_test)

    #print ("Índice R² (score): {}".format(r2_RFR))

    

    sns.set(style="whitegrid")

    fig, axs = plt.subplots(ncols=2, sharey=False, figsize=(15,5))

    sns.residplot(predictions, Y_test, color="g", ax=axs[0]).set_title("Residuals plot of " + model_name)

    sns.scatterplot(x=Y_test, y=predictions, ax=axs[1]).set_title("Model Error")

    axs[1].set(xlabel='True Values', ylabel='Predicted Values')
def submitPredictions(predictions, model_name):

    y_pred=[]



    for i,predict in enumerate(predictions):

        y_pred.append(str(round(predict,4)))

    y_pred=np.array(y_pred)



    output = pd.DataFrame({'ParcelId': test_df['ParcelId'].astype(np.int32),

            '201610': y_pred, '201611': y_pred, '201612': y_pred,

            '201710': y_pred, '201711': y_pred, '201712': y_pred})



    cols = output.columns.tolist()

    cols = cols[-1:] + cols[:-1]

    output = output[cols]



    from datetime import datetime



    print( "\nWriting results to disk ..." )

    output.to_csv('sub{}_{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S'), model_name), index=False)
def submitBestModelRFR(X_train, Y_train, X_test, Y_test):



    print("Using Best Random Forest Regressor Model...")

    

    # Best model according to previous grid search

    best_rfr = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=25,

               max_features='auto', max_leaf_nodes=None,

               min_impurity_decrease=0.0, min_impurity_split=None,

               min_samples_leaf=1, min_samples_split=5,

               min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=None,

               oob_score=False, random_state=42, verbose=0, warm_start=False)

    

    # Visualizing model performance on training dataset

    best_rfr.fit(X_train, Y_train)

    # Results for the training dataset

    predictions_best_rfr = best_rfr.predict(X_test)

    resultados(Y_test, predictions_best_rfr, "Best RFR")



    # Testing the model on test dataframe and submiting results

    submitPredictions(best_rfr.predict(test_df[X_train.columns.values]), "best_RFR")
def submitBestModelXGB(X_train, Y_train, X_test, Y_test):

    

    print("Using Best XGBoost Regressor Model...")

    

    # Best model according to previous grid search

    best_xgb = XGBRegressor(base_score=0.014415016651286966, booster='gbtree',

       colsample_bylevel=1, colsample_bytree=0.6, eval_metric='logloss',

       gamma=0, importance_type='gain', learning_rate=0.03,

       max_delta_step=0, max_depth=5, min_child_weight=10, missing=None,

       n_estimators=500, n_jobs=1, nthread=None, objective='reg:linear',

       random_state=0, reg_alpha=0, reg_lambda=0.3, scale_pos_weight=1,

       seed=None, silent=1, subsample=0.8)



    #Previous XGB Parameters

#     XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,

#            colsample_bytree=0.7, gamma=0, importance_type='gain',

#            learning_rate=0.03, max_delta_step=0, max_depth=5,

#            min_child_weight=4, missing=None, n_estimators=500, n_jobs=1,

#            nthread=None, objective='reg:linear', random_state=0, reg_alpha=0,

#            reg_lambda=1, scale_pos_weight=1, seed=None, silent=1,

#            subsample=0.7)



    # Visualizing model performance on training dataset

    best_xgb.fit(X_train, Y_train)

    # Results for the training dataset

    predictions_best_xgb = best_xgb.predict(X_test)

    resultados(Y_test, predictions_best_xgb, "Best XGB")



    # Testing the model on test dataframe and submiting results

    submitPredictions(best_xgb.predict(test_df[X_train.columns.values]), "best_XGB")
def submitBestModelTPOT(X_train, Y_train, X_test, Y_test):

    

    print("Using Best TPOT Model (Linear SVR)...")

    

    best_svr = LinearSVR(C=0.1, dual=True, epsilon=0.001, loss="squared_epsilon_insensitive", tol=0.0001)

    

    # Visualizing model performance on training dataset

    best_svr.fit(X_train, Y_train)

    # Results for the training dataset

    predictions_best_svr = best_svr.predict(X_test)

    resultados(Y_test, predictions_best_svr, "Best SVR")



    # Testing the model on test dataframe and submiting results

    submitPredictions(best_svr.predict(test_df[X_train.columns.values]), "best_SVR")
def defaultXGB(X_train, Y_train, X_test, Y_test):

    print ("Using default XGB Regressor...")

    

    default_xgb = XGBRegressor()

    

    # Visualizing model performance on training dataset

    default_xgb.fit(X_train, Y_train)

    # Results for the training dataset

    predictions_default_xgb = default_xgb.predict(X_test)

    resultados(Y_test, predictions_default_xgb, "Default XGB")



    # Testing the model on test dataframe and submiting results

    submitPredictions(default_xgb.predict(test_df[X_train.columns.values]), "default_XGB")
def defaultRFR(X_train, Y_train, X_test, Y_test):

    print ("Using default RFR Regressor...")

    

    default_rfr = RandomForestRegressor()

        

    # Visualizing model performance on training dataset

    default_rfr.fit(X_train, Y_train)

    # Results for the training dataset

    predictions_default_rfr = default_rfr.predict(X_test)

    resultados(Y_test, predictions_default_rfr, "Default RFR")



    # Testing the model on test dataframe and submiting results

    submitPredictions(default_rfr.predict(test_df[X_train.columns.values]), "default_RFR")
train_county = train_df[train_df.propertycountylandusecode == train_df.propertycountylandusecode.value_counts().argmax()]
train_county.shape
X_train, X_test, Y_train, Y_test = train_test_split(train_county.drop(['logerror', 'parcelid'], axis=1), train_county.logerror, test_size=1/3, random_state=42)
print("x_train: {}, x_test: {}, y_train: {}, y_test: {}".format(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape))
print("Running Linear Regressor...")

model_lr = linearRegressor(X_train, Y_train, X_test, Y_test)

submitPredictions(model_lr.predict(test_df[X_train.columns.values]), "LR")
if (grid_search == "Default"):

    defaultXGB(X_train, Y_train, X_test, Y_test) 

    defaultRFR(X_train, Y_train, X_test, Y_test)
if (grid_search == "RFR"):

    print("Running Random Forest Regressor...")

    model_rfr = randomForestRegressor(X_train, Y_train, X_test, Y_test)

    submitPredictions(model_rfr.predict(test_df[X_train.columns.values]), "RFR")
if (grid_search == "XGB"):

    print("Running XGBoost...")

    model_xgb = xgb(X_train, Y_train, X_test, Y_test)

    submitPredictions(model_xgb.predict(test_df[X_train.columns.values]), "XGB")
if (grid_search == "TPOT"):

    tpotRegressor(X_train, Y_train, X_test, Y_test)
if (grid_search == "Best"):

    submitBestModelRFR(X_train, Y_train, X_test, Y_test)

    submitBestModelXGB(X_train, Y_train, X_test, Y_test)

    submitBestModelTPOT(X_train, Y_train, X_test, Y_test)
#Retorno do TPOT

'''

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.svm import LinearSVR



# NOTE: Make sure that the class is labeled 'target' in the data file

tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)

features = tpot_data.drop('target', axis=1).values

training_features, testing_features, training_target, testing_target = \

            train_test_split(features, tpot_data['target'].values, random_state=None)



# Average CV score on the training set was:-0.0678352369864653

exported_pipeline = LinearSVR(C=0.1, dual=True, epsilon=1.0, loss="squared_epsilon_insensitive", tol=0.0001)



exported_pipeline.fit(training_features, training_target)

results = exported_pipeline.predict(testing_features)

'''
from sklearn.neural_network import MLPRegressor

from sklearn.preprocessing import StandardScaler
def submitNeuralNetwork(X_train, X_test, Y_train, Y_test):

    

    print("Using Neural Network ...")

    scaler = StandardScaler()

    X_train_sc = scaler.fit_transform(X_train)

    X_test_sc = scaler.transform(X_test)

    

    rn_regressor = MLPRegressor(hidden_layer_sizes = (350,200,120,),early_stopping = True)

    rn_regressor.fit(X_train_sc, Y_train)

    

    predictions_rn = rn_regressor.predict(X_test_sc)

    

    # Visualizing model performance on training dataset

    # Results for the training dataset

    resultados(Y_test, predictions_rn, "Neural Network")



    # Testing the model on test dataframe and submiting results

    #submitPredictions(rn_regressor.predict(test_df[X_train.columns.values]), "Neural_Network")
submitNeuralNetwork(X_train, X_test, Y_train, Y_test)
def NeuralNetwork_gs(X_train, X_test, Y_train, Y_test):

    

    cv_sets = ShuffleSplit(n_splits=5, test_size = 0.25)

    

    print("Using Neural Network with Grid Search...")

    scaler = StandardScaler()

    X_train_sc = scaler.fit_transform(X_train)

    X_test_sc = scaler.transform(X_test)



    rn_regressor = MLPRegressor()



    # neural network params

    params = {

        'hidden_layer_sizes':[(150,100,),(200,150,100,),(300,200,100)],

        'activation':['relu'],

        'solver':['sgd','adam'],

        'learning_rate': ['constant','adaptive'],

        'learning_rate_init':[.001],

        'early_stopping':[True]

    }



    # Gerar o objeto de busca em matriz

    grid = GridSearchCV(estimator = rn_regressor, param_grid = params, scoring='neg_mean_squared_error',cv=cv_sets, verbose=2,n_jobs = -1)



    grid = grid.fit(X_train_sc, Y_train)



    # Escolher o melhor estimador para predizer os dados de teste

    best_model = grid.best_estimator_



    print("Best Estimator: \n{}\n".format(grid.best_estimator_))

    print("Best Parameters: \n{}\n".format(grid.best_params_))



    pred_rn = best_model.predict(X_test_sc)

  

    resultados(Y_test, pred_rn, "MLP Regressor")



    # Returns the trained model

    return best_model

    
if (grid_search == "MLPR"):

    rn_model = NeuralNetwork_gs(X_train, X_test, Y_train, Y_test)