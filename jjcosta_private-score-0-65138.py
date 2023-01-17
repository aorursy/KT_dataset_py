# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import lightgbm as lgb

import os, datetime, time, shutil, os, traceback, gc

import pandas as pd

import numpy as np

from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn import preprocessing

from sklearn.metrics import accuracy_score





# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/murcia-beer-challenge/beer_train.csv")

test = pd.read_csv("../input/murcia-beer-challenge/beer_test.csv")


le = preprocessing.LabelEncoder()

train['Style'] = le.fit_transform(train['Style']) 

target = train['Style']
train.drop(['Id','Style'], axis=1, inplace=True)

test.drop(['Id'], axis=1, inplace=True)
train['tipo'] = 0

test['tipo'] = 1

all = pd.concat ([train,test])

del train, test

gc.collect()



categorical_columns = [col for col in all.columns if all[col].dtype == 'object']

for col in categorical_columns:

	all = pd.concat([all, pd.get_dummies(all[col], prefix=col, dummy_na= True)],axis=1)

	all.drop([col], axis=1, inplace=True)

	gc.collect()



train = all[all['tipo'] == 0]

test = all[all['tipo'] == 1]

del all

gc.collect()



train.drop(['tipo'], axis=1, inplace=True)

test.drop(['tipo'], axis=1, inplace=True)
params = {'objective': 'multiclass', "num_class" : 11, 'metric': 'multi_logloss', "bagging_seed" : 2020, 'verbose': -1}#

metricas = []

folds = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=3246584)

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train,target)):

	print("fold {}".format(fold_))

	trn_data = lgb.Dataset(train.iloc[trn_idx], label=target.iloc[trn_idx])

	val_data = lgb.Dataset(train.iloc[val_idx], label=target.iloc[val_idx])



	num_round = 10000

	clf = lgb.train(params, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=False, early_stopping_rounds = 100)



	val_aux_prob = clf.predict(train.iloc[val_idx], num_iteration=clf.best_iteration)	

	val_aux =  [np.argmax(line) for line in val_aux_prob]

	auxScore = accuracy_score(target.iloc[val_idx], val_aux)

	metricas.append(auxScore)

	print (auxScore)



	pred_aux_prob = clf.predict(test, num_iteration=clf.best_iteration)

	if (fold_ == 0):

		predictions_prob = pred_aux_prob

	else:			

		predictions_prob += pred_aux_prob
print ('final')

print ('mean: ' + str(np.mean(metricas)))

print ('std: ' + str(np.std(metricas)))

print ('max: ' + str(np.max(metricas)))

print ('min: ' + str(np.min(metricas)))  



submit_num = [np.argmax(line) for line in predictions_prob]

submit = le.inverse_transform(submit_num)



Submission=pd.read_csv("../input/murcia-beer-challenge/beer_sampleSubmission.csv")

Submission['Style']=submit.copy()

Submission.to_csv("bestPS.csv", index=False) 