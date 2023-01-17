
#Importar Modulo
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
print('Ambiente Preparado!')

#Importar Librerias
import xgboost as xgb
import numpy as np
import pandas as pd

print('Librerias Importadas!')

import os
print(os.listdir("../input"))

def vela(a,b):
    
    return (a-b)/a 
    
#z = vela(30, 20)

#print(z)

def ema(vela2, vela1, vela0):
    
    ema0 = 2/4 * (vela0)
    
    ema1 = ema0 + 2/4 * (vela1 - ema0)
    
    ema2 = ema1 + 2/4 * (vela2 - ema1)
     
    return ema2

#w = ema(15, 15, 15)

#print(w)
#Funciones

def get_x(market_train_df):

    x = market_train_df
    
    try:
        x.drop(columns=['assetCode'], inplace=True)
    except:
        pass
    
    try:
        x.drop(columns=['assetName'], inplace=True)
    except:
        pass
    
    try:
        x.drop(columns=['returnsOpenNextMktres10'], inplace=True)
    except:
        pass
    
    try:
        x.drop(columns=['universe'], inplace=True)
    except:
        pass
    
    try:
        x.drop(columns=['returnsClosePrevRaw1'], inplace=True)
    except:
        pass
    
    try:
        x.drop(columns=['returnsOpenPrevRaw1'], inplace=True)
    except:
        pass
    
    try:
        x.drop(columns=['returnsOpenPrevRaw10'], inplace=True)
    except:
        pass
    
    try:
        x.drop(columns=['returnsClosePrevRaw10'], inplace=True)
    except:
        pass
        
    x['vela'] =  vela(market_train_df['close'],market_train_df['open'])
        
    x['vela1'] =  vela(market_train_df['returnsClosePrevMktres1'],market_train_df['returnsOpenPrevMktres1'])
    
    x['vela10'] =  vela(market_train_df['returnsClosePrevMktres10'],market_train_df['returnsOpenPrevMktres10'])
    
    #x['EMA'] =  ema(market_train_df['returnsClosePrevMktres10'],market_train_df['returnsClosePrevMktres1'],market_train_df['close'])
    
    try:
        x.drop(columns='time', inplace=True)
    except:
        pass
        
    try:
        x.drop(columns='volume', inplace=True)
    except:
        pass
     
        
    x.fillna(-1000,inplace=True)

    return x

def get_y(market_train_df):

    y2 = market_train_df
    
    try:
        y = y2['returnsOpenNextMktres10']
    except:
        y = None
        pass

    return y
    
def get_xy(market_train_df):

    y = get_y(market_train_df)
    x = get_x(market_train_df)
    
    return x, y		

def make_predictions(predictions_template_df, market_obs_df):
			
    x = get_x(market_obs_df)

    x = x[train_cols]
    
    data = xgb.DMatrix(x, nthread=-1)
    
    predictions_template_df.confidenceValue = np.clip(m.predict(data, ntree_limit=0),-1,1)
        
#Main

#Recupera Datos y News para Entrenamiento 2007-2016
(market_train_df, news_train_df) = env.get_training_data()

print('Datos y News Recuperados!')

x, y = get_xy(market_train_df)

data = xgb.DMatrix(x, label=y, feature_names=x.columns,nthread=-1)

print('Get Matrix!')

train_cols = x.columns

params = {'max_depth':10,'min_child_weight': 1,'eta':0.8,'objective': 'reg:linear', 'eval_metric:' : "mae",}

m = xgb.train(params, data, 240, evals=[(data, 'Test')])

print('Train!')

#Loop por cada Dia a Predecir con Datos y News Nuevos
for (market_obs_df, news_obs_df, predictions_template_df) in env.get_prediction_days():

        #Realizo Prediccion
        make_predictions(predictions_template_df, market_obs_df)

        #print('Make Prediction!')
      
        #Envio Prediccion
        env.predict(predictions_template_df)

print('Ok!')

#Escribe predicción en file csv

xgb.plot_importance(m)

env.write_submission_file()


