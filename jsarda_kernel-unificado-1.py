
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
    

def get_w(news_train_df):

    w = news_train_df
    
    w['assetCode'] =  news_train_df['assetCodes']
    
    try:
        w.drop(columns=['assetCodes'], inplace=True)
    except:
        pass
    
    try:
        w.drop(columns=['assetName'], inplace=True)
    except:
        pass
    
    try:
        w.drop(columns=['sourceTimestamp'], inplace=True)
    except:
        pass
    
    try:
        w.drop(columns=['firstCreated'], inplace=True)
    except:
        pass
    
    try:
        w.drop(columns=['sourceId'], inplace=True)
    except:
        pass
    
    try:
        w.drop(columns=['headline'], inplace=True)
    except:
        pass
    
    try:
        w.drop(columns=['takeSequence'], inplace=True)
    except:
        pass
    
    try:
        w.drop(columns=['provider'], inplace=True)
    except:
        pass
    
    try:
        w.drop(columns=['subjects'], inplace=True)
    except:
        pass

    try:
        w.drop(columns=['audiences'], inplace=True)
    except:
        pass

    try:
        w.drop(columns=['bodySize'], inplace=True)
    except:
        pass

    try:
        w.drop(columns=['companyCount'], inplace=True)
    except:
        pass

    try:
        w.drop(columns=['headlineTag'], inplace=True)
    except:
        pass

    try:
        w.drop(columns=['marketCommentary'], inplace=True)
    except:
        pass
    
    try:
        w.drop(columns=['sentenceCount'], inplace=True)
    except:
        pass
    
    try:
        w.drop(columns=['firstMentionSentence'], inplace=True)
    except:
        pass
    
    try:
        w.drop(columns=['urgency'], inplace=True)
    except:
        pass
    
    try:
        w.drop(columns=['relevancy'], inplace=True)
    except:
        pass
    
    try:
        w.drop(columns=['sentimentClass'], inplace=True)
    except:
        pass
 
    try:
        w.drop(columns=['noveltyCount12H'], inplace=True)
    except:
        pass
 
    try:
        w.drop(columns=['noveltyCount24H'], inplace=True)
    except:
        pass
 
    try:
        w.drop(columns=['noveltyCount3D'], inplace=True)
    except:
        pass
 
    try:
        w.drop(columns=['noveltyCount5D'], inplace=True)
    except:
        pass
 
    try:
        w.drop(columns=['noveltyCount7D'], inplace=True)
    except:
        pass

    try:
        w.drop(columns=['volumeCounts12H'], inplace=True)
    except:
        pass

    try:
        w.drop(columns=['volumeCounts24H'], inplace=True)
    except:
        pass
    
    try:
        w.drop(columns=['volumeCounts3D'], inplace=True)
    except:
        pass
    
    try:
        w.drop(columns=['volumeCounts5D'], inplace=True)
    except:
        pass

    try:
        w.drop(columns=['volumeCounts7D'], inplace=True)
    except:
        pass
    
    
    w['ratio'] =  news_train_df['sentimentWordCount']/news_train_df['wordCount']
        
     
    try:
        w.drop(columns=['sentimentWordCount'], inplace=True)
    except:
        pass
    
    try:
        w.drop(columns=['wordCount'], inplace=True)
    except:
        pass
    
    try:
        w.drop(columns=['sentimentNegative'], inplace=True)
    except:
        pass
 
    try:
        w.drop(columns=['sentimentNeutral'], inplace=True)
    except:
        pass
 
    
    
    return w
def get_xwy(market_train_df, news_train_df):

    y = get_y(market_train_df)
    x = get_x(market_train_df)
    w = get_w(news_train_df)
    
    return x, w, y		

def make_predictions(predictions_template_df, market_obs_df, news_obs_df):
			
    x = get_x(market_obs_df)
    
    w = get_w(news_obs_df)

    #print(w)
    
    cdf = pd.merge(x, w, how='left', on=['time', 'assetCode'], copy=False) 

    cdf = cdf[train_cols]
    
    #print(cdf.shape)
    #print(cdf.columns)
    #print(cdf)
    
    data = xgb.DMatrix(cdf, nthread=-1)
    
    predictions_template_df.confidenceValue = np.clip(m.predict(data,ntree_limit=0),-1,1)

#Main

#Recupera Datos y News para Entrenamiento 2007-2016
(market_train_df, news_train_df) = env.get_training_data()

print('Datos y News Recuperados!')

x, w, y = get_xwy(market_train_df, news_train_df)

del market_train_df, news_train_df

#print(w.columns)
#print(x.columns)

print('Get XWY!')

cdf = pd.merge(x, w, how='left', on=['time', 'assetCode'], copy=False) 

print(w['time'])
print(w['assetCode'])

print(format(cdf['time'].isnull().sum()))


print('Merge!')

w.drop(columns=['time'], inplace=True)
x.drop(columns=['time'], inplace=True)
cdf.drop(columns=['time'], inplace=True)
w.drop(columns=['assetCode'], inplace=True)
x.drop(columns=['assetCode'], inplace=True)
cdf.drop(columns=['assetCode'], inplace=True)

data = xgb.DMatrix(cdf, label=y, feature_names=cdf.columns)


print('Get Matrix!')

train_cols = cdf.columns

params = {'max_depth':10,'min_child_weight': 1,'eta':.8,'objective':'reg:linear','eval_metric:' : "mae",}

m = xgb.train(params, data, 3, evals=[(data, 'Test')])

print('Train!')

#Loop por cada Dia a Predecir con Datos y News Nuevos
for (market_obs_df, news_obs_df, predictions_template_df) in env.get_prediction_days():

        
        #Realizo Prediccion
        make_predictions(predictions_template_df, market_obs_df, news_obs_df)

        #print('Make Prediction!')
      
        #Envio Prediccion
        env.predict(predictions_template_df)
        
        break

print('Ok!')

#Escribe predicción en file csv

xgb.plot_importance(m)

env.write_submission_file()

