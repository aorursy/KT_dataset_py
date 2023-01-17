import pandas as pd

import pandas_profiling

import lightgbm as lgb

import os, datetime, time, shutil, os, traceback, gc

import pandas as pd

import numpy as np

from sklearn.model_selection import KFold, RepeatedKFold

from pandas_profiling import ProfileReport

from pathlib import Path

from sklearn import preprocessing

from sklearn.metrics import mean_squared_log_error

import unidecode

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import re
train = pd.read_csv("../input/murcia-car-challenge/train.csv")

test = pd.read_csv("../input/murcia-car-challenge/test.csv")

print (train[train.Precio>=2000000])



train = train[train.Precio<2000000]
palabrasModeloEliminar = ["CAMBIO","GARANTIA","IMPECABLE","GASOLINA","DIESEL","FINANCIAMOS","CON","ASNEF","RESTAURADO",\

	"KILOMETROS","PLAZAS", "HDI","TDI","CDI","MOTOR","VELOCIDADES","UNICO","DUENO","SOLO","AUTOMATICO",\

	"BLUE","BLUEDRIVE","BLUEHDI","BLUEMOTION","BLUETEC","RESERVADO","D4D","COMPRO","COCHES","CERTIFICADOS"]



cambios = [["HYBRID","HIBRIDO"]]



nuevasColumnas = ["GARANTIA","IMPECABLE","BLUE","GLP","HDI","TDI","CDI"]

exregulares = ["PUERTAS","CV","KM","KMS","KILOMETROS","KW"]

exregularesquitarespacios = ["CLASE","SERIE"]



marcas = []



quitar = ["\"",","]



def tran(valor):

    try:

        return unidecode.unidecode(valor.upper().strip())

    except:

        return valor



def tranModelo(valor):

	vorig = valor

	valor = unidecode.unidecode(valor.upper().strip())

	for eli in marcas:

		valor = valor.replace(eli,"")

	for eli in quitar:

		valor = valor.replace(eli,"")

	for eli in palabrasModeloEliminar:

		valor = valor.replace(eli,"")

	for cam in cambios:

		valor = valor.replace(cam[0],cam[1])	

	for res in exregularesquitarespacios:

		mat = re.search(r'(' + res + '\s+\w+)', valor)

		if mat:

			valor = valor.replace (mat.group(0), mat.group(0).replace(" ", ""))

	for res in 	exregulares:

		mat = re.search(r'(\w+\s*' + res + ')', valor)

		if mat:

			valor = valor.replace (mat.group(0), "")



	#if (not valor):

	#		print (vorig)

	return valor



def tranTiempo(valor):

	valor = unidecode.unidecode(valor.upper().strip())

	if (valor == "DESTACADO"):

		return 0

	elif (valor == "NUEVO ANUNCIO"):

		return 1

	elif ("SEG" in valor or "MIN" in valor or "HORA" in valor):

		return 1

	else:

		valor = valor.replace('DIAS', '')

		valor = valor.replace(' ','')

		return int(valor)



rempMarcas = [["MERCEDES-BENZ","MERCEDES"]]



def tranMarca(valor):

	valor = unidecode.unidecode(valor.upper().strip())

	for cam in rempMarcas:

		valor = valor.replace(cam[0],cam[1])	

	return valor
train['tipoData'] = 0

test['tipoData'] = 1

targetSinLog = train['Precio']

train['Precio'] = np.log1p(train['Precio'])

target = train['Precio']



train.drop(['Id','Precio','Localidad'], axis=1, inplace=True)

test.drop(['Id','Localidad'], axis=1, inplace=True)



all = pd.concat ([train,test])



all.rename(columns={'Año': 'Anyo'}, inplace=True)



for newColumn in nuevasColumnas:

    all[newColumn] = all['Modelo'].str.contains(newColumn)



marcas = all['Marca'].unique()

all['Marca'] = all['Marca'].apply(tranMarca)    

    

all['Tiempo'] = all['Tiempo'].apply(tranTiempo)



all['Modelo'] = all['Modelo'].apply(tranModelo)
countVec = CountVectorizer(min_df=0.001, strip_accents='ascii')

countVecResult = countVec.fit_transform(all['Modelo'])

all.drop(['Modelo'], axis=1, inplace=True)	

pandasCount = pd.DataFrame(countVecResult.toarray(), columns=countVec.get_feature_names())

all = all.join(pandasCount)
categorical_columns = [col for col in all.columns if all[col].dtype == 'object']

for col in categorical_columns:

	all[col] = all[col].apply(tran)

	all = pd.concat([all, pd.get_dummies(all[col], prefix=col, dummy_na= True)],axis=1)

	all.drop([col], axis=1, inplace=True)

	gc.collect()

train = all[all['tipoData'] == 0]

test = all[all['tipoData'] == 1]

del all

gc.collect()



df_train_columns = [c for c in train.columns if c not in ['tipoData']]



'''

params = {'bagging_seed': 3246584, 'feature_fraction_seed': 3246584, 'lambda_l1': 0.7154229992732665, \

    'lambda_l2': 0.024214581481318928, 'learning_rate': 0.011592695279683055, 'max_bin': 4399, 'max_depth': 55, \

    'metric': 'rmse', 'min_data_in_leaf': 1, 'num_leaves': 424, 'objective': 'regression', 'verbose': -1}

'''



params = {'objective': 'regression',  'metric': 'rmse', "bagging_seed" : 2020, 'verbose': -1}#



test = test[df_train_columns]

folds = RepeatedKFold(n_splits=5, n_repeats=1, random_state=3246584)



divisiones = 0

metricas = []
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train,target.values)):

	divisiones += 1

	print("fold {}".format(fold_))

	trn_data = lgb.Dataset(train.iloc[trn_idx][df_train_columns], label=target.iloc[trn_idx])

	val_data = lgb.Dataset(train.iloc[val_idx][df_train_columns], label=target.iloc[val_idx])



	num_round = 10000

	clf = lgb.train(params, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=False, early_stopping_rounds = 100)

	

	val_aux = clf.predict(train.iloc[val_idx][df_train_columns], num_iteration=clf.best_iteration)	

	val_aux = np.expm1(val_aux)

	auxScore = np.sqrt(mean_squared_log_error( targetSinLog.iloc[val_idx], val_aux ))

	print (auxScore)

	metricas.append(auxScore)



	pred_aux = clf.predict(test, num_iteration=clf.best_iteration) 

	if (fold_ == 0):

		predictions = pred_aux

	else:			

		predictions += pred_aux
print ('final')

print ('mean: ' + str(np.mean(metricas)))

print ('std: ' + str(np.std(metricas)))

print ('max: ' + str(np.max(metricas)))

print ('min: ' + str(np.min(metricas)))  



predictions = predictions / divisiones

predictions = np.expm1(predictions)



Submission=pd.read_csv("../input/murcia-car-challenge/sampleSubmission.csv")

Submission['Precio']=predictions.copy()

Submission.to_csv("LgbmV1_base.csv", index=False)  	