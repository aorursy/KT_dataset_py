import datetime, time

#from collections import Counter

#import category_encoders as ce

# from datetime import timedelta 

# from datetime import datetime

#from scipy import interp

import pandas as pd

import numpy as np

#import itertools

#import warnings



%matplotlib inline

#import seaborn as sns

import matplotlib.pyplot as plt

#from matplotlib import rcParams



from sklearn.model_selection import train_test_split 

#from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_curve, precision_recall_curve

from sklearn.metrics import roc_auc_score # , accuracy_score,  f1_score, auc

#from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb
def read_data(file_path):

    print('Loading datasets...')

    train = pd.read_csv(file_path + 'train.csv', sep=',')

    test = pd.read_csv(file_path + 'test.csv', sep=',')

    print('Datasets loaded')

    return train, test



PATH = '../input/cat-in-the-dat-ii/'

train, test = read_data(PATH)

print(train.shape, test.shape)

print(train.head(2))

print(test.head(2))



X = train.drop(['id','target'], axis = 1)

categorical_features = [col for c, col in enumerate(X.columns) \

                        if not ( np.issubdtype(X.dtypes[c], np.number )  )  ]

y = train['target']



print( len(categorical_features), X.shape, y.shape, y.mean()  )

for f in categorical_features:

    X[f] = X[f].astype('category')



X1,X2, y1,y2 = train_test_split(X,y, test_size = 0.2, random_state = 0, stratify = y )

print(X1.shape, X2.shape, y1.shape, y2.shape, y1.mean(), y2.mean(), y.mean() )
model = lgb.LGBMClassifier(**{

                'learning_rate': 0.05,

                'feature_fraction': 0.1,

                'min_data_in_leaf' : 12,

                'max_depth': 2, # it was 3 

                'reg_alpha': 1,

                'reg_lambda': 1,

                'objective': 'binary',

                'metric': 'auc',

                'n_jobs': -1,

                'n_estimators' : 5000,

                'feature_fraction_seed': 42,

                'bagging_seed': 42,

                'boosting_type': 'gbdt',

                'verbose': 200,

                'is_unbalance': True,

                'boost_from_average': False})



import datetime

print('Start fit.', datetime.datetime.now() )



model = model.fit(X, y,

                  eval_set = [(X1, y1), 

                              (X2, y2)],

                  verbose = 1000,

                  eval_metric = 'auc',

                  early_stopping_rounds = 1000)



print('End fit.', datetime.datetime.now() )



X_test = test.drop('id',axis = 1 )

for f in categorical_features:

    X_test[f] = X_test[f].astype('category')

pd.DataFrame({'id': test['id'], 'target': model.predict_proba(X_test)[:,1]}).to_csv('submission_max_depth2_trained_on_whole.csv', index=False)
