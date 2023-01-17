# from datetime # 

import datetime

import pandas as pd

import numpy as np



%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt



#from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_curve, precision_recall_curve

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, auc

# from sklearn.model_selection import StratifiedKFold

# import lightgbm as lgb

from sklearn.model_selection import train_test_split 



#def read_data(file_path):

print('Loading datasets...')

file_path = '../input/cat-in-the-dat-ii/'

train = pd.read_csv(file_path + 'train.csv', sep=',')

test = pd.read_csv(file_path + 'test.csv', sep=',')

print('Datasets loaded')

# return train, test

# train, test = read_data(PATH)



print(train.shape, test.shape)

print(train.head(2))

print(test.head(2))



X = train.drop(['id','target'], axis = 1)

categorical_features = [col for c, col in enumerate(X.columns) \

                        if not ( np.issubdtype(X.dtypes[c], np.number )  )  ]

y = train['target']



print( len(categorical_features), X.shape, y.shape, y.mean()  )

X = X.fillna(-9999)

for f in categorical_features:

    X[f] = X[f].astype('category')



X1,X2, y1,y2 = train_test_split(X,y, test_size = 0.2, random_state = 0, stratify = y )

print(X1.shape, X2.shape, y1.shape, y2.shape, y1.mean(), y2.mean(), y.mean() )
# params from: https://www.kaggle.com/lucamassaron/catboost-in-action-with-dnn



import datetime

from catboost import CatBoostClassifier



print('Start fit.', datetime.datetime.now() )



best_params = {'bagging_temperature': 0.8,

               'depth': 5,

               'iterations': 50000,

               'l2_leaf_reg': 30,

               'learning_rate': 0.05,

               'random_strength': 0.8}



model = CatBoostClassifier( **best_params,

                          loss_function='Logloss',

                          eval_metric = 'AUC',

                          nan_mode='Min',

                          thread_count=4,  task_type = 'GPU',

                          verbose = False)



model.fit(X1 , y1 , eval_set = (X2 , y2), cat_features = categorical_features,

            verbose_eval=300, 

             early_stopping_rounds=500,

             use_best_model=True,

             plot=True)         

         

pred = model.predict_proba(X2)[:,1]

score = roc_auc_score(y2 , pred)

print(score)  

print('End fit.', datetime.datetime.now() )

# Results of launch saved:

# 0:	learn: 0.6827005	test: 0.6822092	best: 0.6822092 (0)	total: 84.8ms	remaining: 1h 10m 40s

# 300:	learn: 0.7806345	test: 0.7820942	best: 0.7820942 (300)	total: 19.6s	remaining: 54m 2s

# 600:	learn: 0.7833911	test: 0.7840392	best: 0.7840392 (600)	total: 38.7s	remaining: 53m 3s

# 900:	learn: 0.7848153	test: 0.7844915	best: 0.7844916 (898)	total: 59.1s	remaining: 53m 41s

# 1200:	learn: 0.7859464	test: 0.7846847	best: 0.7846847 (1200)	total: 1m 18s	remaining: 53m 11s

# 1500:	learn: 0.7870806	test: 0.7848300	best: 0.7848300 (1500)	total: 1m 37s	remaining: 52m 26s

# 1800:	learn: 0.7880922	test: 0.7848453	best: 0.7848582 (1754)	total: 1m 57s	remaining: 52m 25s

# 2100:	learn: 0.7891364	test: 0.7848671	best: 0.7848782 (2079)	total: 2m 17s	remaining: 52m 5s

# 2400:	learn: 0.7901407	test: 0.7849226	best: 0.7849235 (2395)	total: 2m 36s	remaining: 51m 32s

# 2700:	learn: 0.7911730	test: 0.7849048	best: 0.7849265 (2403)	total: 2m 56s	remaining: 51m 33s

# bestTest = 0.7849265337

# bestIteration = 2403

# Shrink model to first 2404 iterations.

# 0.784926509685652

# End fit. 2020-01-14 20:10:23.686622



        
X_test = test.drop('id',axis = 1 )

X_test = X_test.fillna(-99999)

for f in categorical_features:

    X_test[f] = X_test[f].astype('category')

    

pd.DataFrame({'id': test['id'], 'target': model.predict_proba(X_test)[:,1]}).to_csv('submission.csv', index=False)
