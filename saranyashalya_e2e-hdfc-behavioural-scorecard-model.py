import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



random_seed = 1



from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

from sklearn.metrics import f1_score

import xgboost as xgb

import lightgbm as lgb

from hyperopt import fmin, Trials, hp, tpe, STATUS_OK





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

#Reading training and test dataset

df_train = pd.read_csv("../input/hdfc-2019/DataSet/Train.csv")

df_test = pd.read_csv("../input/hdfc-2019/DataSet/Test.csv")
df_train.shape, df_test.shape
pd.set_option('display.max_columns', 500)

df_train.head()
df_test.head()
#Col1 is unique entries and Col2 is target column

print(df_train['Col1'].unique().shape, df_train.shape)

print(df_test['Col1'].unique().shape, df_test.shape)
X_train = df_train.drop(['Col1','Col2'], axis = 1)

y_train = df_train['Col2']



X_test = df_test.drop(['Col1'], axis = 1)
X_train.shape, y_train.shape, X_test.shape
#ratio of 2 classes

y_train.value_counts(normalize = True)*100
X_train.info()
X_test.info()
# There is different in the count of object datatypes in train and test



for col in X_train.select_dtypes('object').columns:

    print(col)

    print(X_train[col].unique())

    print("_______")
for col in X_test.select_dtypes('object').columns:

    print(col)

    print(X_test[col].unique())

    print("_____")
#converting all string numeric data to float and '-' to nan



def convert_to_float(row):

    if row=='-':

        return np.nan

    else:

        return float(row)
columns_need_treatment = list(X_train.select_dtypes('object').columns) + list(X_test.select_dtypes('object').columns)

print(len(columns_need_treatment))

print(columns_need_treatment)
for col in columns_need_treatment:

    X_train[col] = X_train[col].apply(convert_to_float)

    X_test[col] = X_test[col].apply(convert_to_float)
X_train.info(), X_test.info()
## Duplicate rows



duplicate_columns_in_train = X_train.duplicated()

duplicate_columns_in_test = X_test.duplicated()
sum(duplicate_columns_in_train), sum(duplicate_columns_in_test)
y_train[duplicate_columns_in_train].value_counts(normalize = True)
X_train['duplicate_row'] = False

X_test['duplicate_row'] = False

X_train.loc[duplicate_columns_in_train,'duplicate_row'] = True

X_test.loc[duplicate_columns_in_test,'duplicate_row'] = True

## Duplicate features



features = X_train.columns

duplicate_columns = set()

for i in range(len(features)):

    for j in range(i+1, len(features)):

        if np.all(X_train[features[i]] == X_train[features[j]]):

            print(features[i], features[j])

            duplicate_columns.add(features[j])
len(duplicate_columns)
selected_features = [_ for _ in X_train.columns if _ not in duplicate_columns]

print(len(selected_features))
X_train, X_valid, y_train, y_valid = train_test_split(X_train[selected_features], y, random_state = random_seed, test_size = 0.2)
def score(params):

    try:

        print("Training with params: ",params)

        num_rounds = int(params['n_estimators'])

        del params['n_estimators']

        dtrain = xgb.DMatrix(X_train,label = y_train)

        dvalid = xgb.DMatrix(X_valid, label = y_valid)

        

        watchlist = [(dtrain,'train'),(dvalid,'valid')]

        xgb_model = xgb.train( params, dtrain, num_rounds, evals = watchlist, verbose_eval = False )

        predictions = xgb_model.predict(dvalid, ntree_limit = xgb_model.best_iteration + 1)

        

        predictions = (prediction >= 0.5).astype('int')

        score = f1_score(y_valid, predictions, average ='weighted')

        print("Score : {0}\n".format(score))

        

        loss = 1-score

        print("Loss : {0}".format(loss))

        return {'loss' :loss, 'status':STATUS_OK}

    

    except AssertionError as obj:

        loss = 1 - 0

        return {'loss':loss, 'status':STATUS_OK}

    except Exception as obj:

        loss = 1 - 0

        return {'loss':loss, 'status':STATUS_OK}

    

    
def optimize(trials, max_evals, random_state = random_seed):

    space = { 

        'n_estimators' : hp.quniform('n_estimators', 100,300,1),

        'eta' : hp.quniform('eta',0.025,0.5, 0.025),

        'max_depth' : hp.choice('max_depth', np.arange(1,7,dtype = int)),

        'min_child_weight' : hp.quniform('min_child_weight',1,6,1), # "stop trying to split once your sample size in a node goes below a given threshold".

        'subsample' : hp.quniform('subsample', 0.5,1, 0.05),

        'gamma' : hp.quniform('gamma', 0,1,0.05), #Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger gamma is, the more conservative the algorithm will be.

        'colsample_by_tree' : hp.quniform('colsample_by_tree',0.5,1,0.05),

        'scale_pos_weight' : hp.quniform('scale_pos_weight',1,4,0.05), #Control the balance of positive and negative weights, useful for unbalanced classes. A typical value to consider: sum(negative instances) / sum(positive instances).

        'reg_alpha' : hp.quniform('reg_alpha', 0,1,0.05), #L1 regularization term on weights. Increasing this value will make model more conservative.

        'reg_lambda' : hp.quniform('reg_lambda',1,5,0.05), #L2 regularization term on weights. Increasing this value will make model more conservative.

        'eval_metric': 'logloss',

        'objective': 'binary:logistic',

        'nthread': 4,

        'booster':'gbtree',

        'tree_method':'exact',

        'silent':1,

        'seed':random_seed      

        

    }

    best = fmin(score, space, algo = tpe.suggest, trials = trials, max_evals = max_evals)

    return best

trials = Trials()

max_evals = 25



best_hyperparams = optimize(trials, max_evals)

print("Best hyperparameters are : \n", best_hyperparams)
#best hyperparameters

best_hyperparams
param = best_hyperparams

num_round = int(param['n_estimators'])

del param['n_estimators']
#oof (out of fold prediction)

num_splits = 5

folds = StratifiedKFold(n_splits = num_splits, shuffle = True, random_state=random_seed)

dxtest = xgb.DMatrix(X_test[selected_features])
y_test_pred = np.zeros((X_test[selected_features].shape[0],1))

print(y_test_pred.shape)

y_valid_scores =[]
X_TRAIN = X_train[selected_features].copy()

Y_TRAIN = y_train.copy()

X_TRAIN = X_TRAIN.reindex()

Y_TRAIN = Y_TRAIN.reindex()



for fold, (train_index, valid_index) in enumerate(folds.split(X_TRAIN, Y_TRAIN)):

    print("Fold...", fold)

    X_train, X_valid = X_TRAIN.iloc[train_index], X_TRAIN.iloc[valid_index]

    y_train, y_valid= Y_TRAIN.iloc[train_index], Y_TRAIN.iloc[valid_index]

    

    dtrain = xgb.DMatrix(X_train, label =y_train)

    dvalid = xgb.DMatrix(X_valid, label = y_valid)

    

    evallist = [(dtrain,'train'),(dvalid,'valid')]

    

    xgb_model = xgb.train(param, dtrain, num_round, evallist , verbose_eval = 50)

    

    y_pred_valid = xgb_model.predict(dvalid, ntree_limit= xgb_model.best_iteration + 1)

    

    y_valid_scores.append(f1_score(y_valid, (y_pred_valid >=0.5).astype(int),average ='weighted'))

    

    y_pred = xgb_model.predict(dxtest, ntree_limit = xgb_model.best_iteration + 1)

    

    y_test_pred += y_pred.reshape(-1,1)



y_test_pred /= num_splits





                          
y_valid_scores, np.mean(y_valid_scores)
output = df_test[['Col1']].copy()

output['Col2'] = (y_test_pred >=0.5).astype(int)
output.head()
output['Col2'].value_counts()/output.shape[0] *100

output.to_csv("submission.csv", index = False)