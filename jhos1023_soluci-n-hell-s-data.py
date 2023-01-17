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

campaña = pd.read_csv("/kaggle/input/datathon-belcorp-prueba/campana_consultora.csv")

campaña = campaña.iloc[:,1:]

del campaña["codigocanalorigen"]

del campaña["codigofactura"]

del campaña["flagpasopedidoweb"]

del campaña["geografia"]

campaña.head()
campaña["flag_total"]=campaña["flagactiva"]+campaña["flagpasopedidocuidadopersonal"]+campaña["flagpasopedidomaquillaje"]+campaña["flagpasopedidotratamientocorporal"]+campaña["flagpasopedidotratamientofacial"]+campaña["flagpasopedidofragancias"]                             

campaña["segmentacion"]=campaña["segmentacion"].astype('category').cat.codes

campaña["evaluacion_nuevas"]=campaña["evaluacion_nuevas"].astype('category').cat.codes

# Separar las campañas

a1818=campaña[campaña["campana"]==201818]    

a1815=campaña[campaña["campana"]==201815]    

a1817=campaña[campaña["campana"]==201817]    

a1813=campaña[campaña["campana"]==201813]    

a1816=campaña[campaña["campana"]==201816]    

a1812=campaña[campaña["campana"]==201812]    

a1901=campaña[campaña["campana"]==201901] #   

a1814=campaña[campaña["campana"]==201814]    

a1906=campaña[campaña["campana"]==201906]   # 

a1905=campaña[campaña["campana"]==201905]  #  

a1903=campaña[campaña["campana"]==201903]    #

a1902=campaña[campaña["campana"]==201902]    #

a1904=campaña[campaña["campana"]==201904]    #

a1811=campaña[campaña["campana"]==201811]    

a1810=campaña[campaña["campana"]==201810]    

a1809=campaña[campaña["campana"]==201809]    

a1808=campaña[campaña["campana"]==201808]    

a1807=campaña[campaña["campana"]==201807]    

#Formar el train

campaña_prueba=pd.concat([a1905,a1904,a1903,a1902,a1901,a1818,a1817,a1816,a1815,a1814,a1813,a1812,a1811,a1810,a1809,a1808,a1807])

t1=campaña_prueba.groupby(["IdConsultora"]).sum().add_prefix("SUM_").reset_index()

del t1["SUM_campana"]

a1906=a1906.iloc[:,:3].reset_index()

del a1906["index"]

train_df=a1906.merge(t1,on="IdConsultora",how="inner")

del train_df["campana"]

train_df.head()
# Formar el test

campaña_test=pd.concat([a1906,a1905,a1904,a1903,a1902,a1901,a1818,a1817,a1816,a1815,a1814,a1813,a1812,a1811,a1810,a1809,a1808])

t2=campaña_test.groupby(["IdConsultora"]).sum().add_prefix("SUM_").reset_index()

del t2["SUM_campana"]
# Importar submission

submission=pd.read_csv("/kaggle/input/datathon-belcorp-prueba/predict_submission.csv")

consultora_sub=submission.merge(t2, right_on='IdConsultora',left_on="idconsultora",how="left")

del consultora_sub["IdConsultora"]

del consultora_sub["flagpasopedido"]

consultora_sub.head()
#Importar maestro consultoras

maestro=pd.read_csv("/kaggle/input/datathon-belcorp-prueba/maestro_consultora.csv").iloc[:,1:]



maestro=maestro.join(pd.get_dummies(maestro.estadocivil))

del maestro["estadocivil"]

maestro
# Crear variable

maestro["ratio1"]=((maestro.campanaultimopedido-maestro.campanaprimerpedido)-(maestro.campanaultimopedido-maestro.campanaprimerpedido).mean())/pow((maestro.campanaultimopedido-maestro.campanaprimerpedido).var(),0.5)

# Juntar con el train y test 

del maestro["campanaingreso"]

del maestro["campanaprimerpedido"]



train_df1=train_df.merge(maestro,on=["IdConsultora"])

consultora_sub1=consultora_sub.merge(maestro, right_on='IdConsultora',left_on="idconsultora",how="left")

del consultora_sub1["IdConsultora"]
# MODELADO

import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt 

import gc

import os

import logging

import datetime

import warnings

import numpy as np

import pandas as pd

import seaborn as sns

import lightgbm as lgb

from tqdm import tqdm_notebook

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.model_selection import StratifiedKFold

param = {

    'bagging_freq': 5, #5

    'bagging_fraction': 0.4, #0.4

    'boost_from_average':'false',

    'min_child_samples': 30,

    'boost': 'gbdt',

    'feature_fraction': 0.5, #0.05

    'learning_rate': 0.01, #0.01

    'max_depth': -1,  

    'metric':'auc',

    'min_data_in_leaf': 80, #80

    'min_sum_hessian_in_leaf': 10, #10

    'num_leaves': 13, #13

    'num_threads': 8, 

    'tree_learner': 'serial',

    'objective': 'binary', 

    'verbosity': 1,

    "is_unbalance":False,

    "random_state":1234

}

train_df=train_df1

test_df=consultora_sub1

features = [c for c in train_df.columns if c not in ["campanaultimopedido","Flagpasopedido","IdConsultora","idconsultora"]]#"SUM_flagdigital","SUM_flagpedidoanulado","flagsupervisor"]]

target=train_df["Flagpasopedido"]

train_df.fillna(train_df.mean(), inplace=True) #reemplazar nan con media

test_df.fillna(test_df.mean(), inplace=True) 



folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=44000)

oof = np.zeros(len(train_df))

predictions = np.zeros(len(test_df))

feature_importance_df = pd.DataFrame()



for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):

    print("Fold {}".format(fold_))

    trn_data = lgb.Dataset(train_df.iloc[trn_idx][features], label=target.iloc[trn_idx])

    val_data = lgb.Dataset(train_df.iloc[val_idx][features], label=target.iloc[val_idx])



    num_round = 10000

    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 1000)

    oof[val_idx] = clf.predict(train_df.iloc[val_idx][features], num_iteration=clf.best_iteration)

    

    fold_importance_df = pd.DataFrame()

    fold_importance_df["Feature"] = features

    fold_importance_df["importance"] = clf.feature_importance()

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    

    predictions += clf.predict(test_df[features], num_iteration=clf.best_iteration) / folds.n_splits



print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))   
stock_test1=pd.read_csv("/kaggle/input/datathon-belcorp-prueba/predict_submission.csv")

sub_df = pd.DataFrame({"idconsultora":stock_test1["idconsultora"].values})

sub_df["flagpasopedido"] = predictions

testeo=pd.concat([test_df[["campanaultimopedido"]],sub_df],axis=1)



for i in range(0,len(testeo)):

  if testeo.iloc[i,0] < 201907:

    testeo.iloc[i,2]=0

  elif testeo.iloc[i,0] == 201907:

    testeo.iloc[i,2]=1

del testeo["campanaultimopedido"]

testeo.to_csv("testeo.csv",index=False)