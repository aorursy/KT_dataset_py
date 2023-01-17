import os



import numpy as np

import pandas as pd

import lightgbm as lgb

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report
%%time

# Only load those columns in order to save space

keep_cols = ['event_id', 'game_session', 'installation_id', 'event_count', 'event_code', 'title', 'game_time', 'type', 'world']



train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv', usecols=keep_cols)

test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv', usecols=keep_cols)

train_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')

submission = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')
def group_and_reduce(df):

    # group1 and group2 are intermediary "game session" groups,

    # which are reduced to one record by game session. group1 takes

    # the max value of game_time (final game time in a session) and 

    # of event_count (total number of events happened in the session).

    # group2 takes the total number of event_code of each type

    group1 = df.drop(columns=['event_id', 'event_code']).groupby(

        ['game_session', 'installation_id', 'title', 'type', 'world']

    ).max().reset_index()



    group2 = pd.get_dummies(

        df[['installation_id', 'event_code']], 

        columns=['event_code']

    ).groupby(['installation_id']).sum()



    # group3, group4 and group5 are grouped by installation_id 

    # and reduced using summation and other summary stats

    group3 = pd.get_dummies(

        group1.drop(columns=['game_session', 'event_count', 'game_time']),

        columns=['title', 'type', 'world']

    ).groupby(['installation_id']).sum()



    group4 = group1[

        ['installation_id', 'event_count', 'game_time']

    ].groupby(

        ['installation_id']

    ).agg([np.sum, np.mean, np.std])



    return group2.join(group3).join(group4)
%%time

train_small = group_and_reduce(train)

test_small = group_and_reduce(test)



print(train_small.shape)

train_small.head()
small_labels = train_labels[['installation_id', 'accuracy_group']].set_index('installation_id')

train_joined = train_small.join(small_labels).dropna()



x_train, x_val, y_train, y_val = train_test_split(

    train_joined.drop(columns='accuracy_group').values,

    train_joined['accuracy_group'].values,

    test_size=0.15, random_state=2019

)
from sklearn.metrics import confusion_matrix

def quadKappa(act,pred,n=4,hist_range=(0,3)):

    

    O = confusion_matrix(act.astype(int),pred.astype(int))

    O = np.divide(O,np.sum(O))

    

    W = np.zeros((n,n))

    for i in range(n):

        for j in range(n):

            W[i][j] = ((i-j)**2)/((n-1)**2)

            

    act_hist = np.histogram(act,bins=n,range=hist_range)[0]

    prd_hist = np.histogram(pred,bins=n,range=hist_range)[0]

    

    E = np.outer(act_hist,prd_hist)

    E = np.divide(E,np.sum(E))

    

    num = np.sum(np.multiply(W,O))

    den = np.sum(np.multiply(W,E))

    print('QuadKappa',1-np.divide(num,den))

    return 1-np.divide(num,den)
import lightgbm as lgb

from sklearn.model_selection import cross_val_score

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from hyperopt.pyll import scope

from time import time



def hyperopt(param_space, X_train, y_train, X_test, y_test, num_eval):

    

    start = time()

    def objective_function(params):

        clf = lgb.LGBMClassifier(**params)

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        score = quadKappa(y_test,y_pred)

        return {'loss': 1-score, 'status': STATUS_OK}



    trials = Trials()

    best_param = fmin(objective_function, 

                      param_space, 

                      algo=tpe.suggest, 

                      max_evals=num_eval, 

                      trials=trials,

                      rstate= np.random.RandomState(1))

    loss = [x['result']['loss'] for x in trials.trials]

    

    best_param_values = [x for x in best_param.values()]

    

    if best_param_values[0] == 0:

        boosting_type = 'gbdt'

    else:

        boosting_type= 'dart'

    

    clf_best = lgb.LGBMClassifier(learning_rate=best_param_values[2],

                                  num_leaves=int(best_param_values[5]),

                                  max_depth=int(best_param_values[3]),

                                  n_estimators=int(best_param_values[4]),

                                  boosting_type=boosting_type,

                                  colsample_bytree=best_param_values[1],

                                  reg_lambda=best_param_values[6],

                                 )

                                  

    clf_best.fit(X_train, y_train)

    

    print("")

    print("##### Results")

    print("Score best parameters: ", min(loss)*-1)

    print("Best parameters: ", best_param)

    print("Test Score: ", clf_best.score(X_test, y_test))

    print("Time elapsed: ", time() - start)

    print("Parameter combinations evaluated: ", num_eval)

    

    return trials,best_param
num_eval =100

param_hyperopt= {

    'objective':'multiclass',

    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(1)),

    'max_depth': scope.int(hp.quniform('max_depth', 5, 15, 1)),

    'n_estimators': scope.int(hp.quniform('n_estimators', 100, 500, 1)),

    'num_leaves': scope.int(hp.quniform('num_leaves', 5, 50, 1)),

    'boosting_type': hp.choice('boosting_type', ['gbdt']),

    'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0),

    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),

}

#results_hyperopt,para = hyperopt(param_hyperopt, x_train, y_train.astype(int), x_val, y_val.astype(int), num_eval)
para={'objective':'multiclass','boosting_type': 'gbdt', 'colsample_by_tree': 0.7910502111662128, 'learning_rate': 0.023277750577652565, 'max_depth': 11.0, 'n_estimators': 397.0, 'num_leaves': 10.0, 'reg_lambda': 0.48705531681758474}

para['boosting_type']='gbdt'

para['num_leaves'] = int(para['num_leaves'] )

para['max_depth'] = int(para['max_depth'])

para['n_estimators'] = int(para['n_estimators'])

clf = lgb.LGBMClassifier(**para)

clf.fit(train_joined.drop(columns='accuracy_group').values,

train_joined['accuracy_group'].values.astype(int))
y_pred = clf.predict(test_small)

test_small['accuracy_group'] = y_pred.astype(int)

test_small[['accuracy_group']].to_csv('submission.csv')

test_small.head()