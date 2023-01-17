import pandas as pd

import numpy as np

import datetime

import pandas as pd

import numpy as np

from sklearn.cross_validation import KFold

from sklearn.cross_validation import train_test_split

from sklearn.metrics import roc_auc_score

from sklearn.grid_search import GridSearchCV

import xgboost as xgb

from xgboost.sklearn import XGBClassifier

import random

from operator import itemgetter

import time

import copy



random.seed(2016)





def create_feature_map(features):

    outfile = open('xgb.fmap', 'w')

    for i, feat in enumerate(features):

        outfile.write('{0}\t{1}\tq\n'.format(i, feat))

    outfile.close()





def get_importance(gbm, features):

    create_feature_map(features)

    importance = gbm.get_fscore(fmap='xgb.fmap')

    importance = sorted(importance.items(), key=itemgetter(1), reverse=True)

    return importance



def create_submission(score, test, prediction):

    now = datetime.datetime.now()

    sub_file = 'submission_' + str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'

    print('Writing submission: ', sub_file)

    f = open(sub_file, 'w')

    f.write('activity_id,outcome\n')

    total = 0

    for id in test['activity_id']:

        str1 = str(id) + ',' + str(prediction[total])

        str1 += '\n'

        total += 1

        f.write(str1)

    f.close()





def intersect(a, b):

    return list(set(a) & set(b))



def get_features(train, test):

    trainval = list(train.columns.values)

    testval = list(test.columns.values)

    output = intersect(trainval, testval)

    output.remove('people_id')

    return sorted(output)



def xgboost_model(train, test, features, target, random_state=0):

    eta = 1.0

    max_depth = 10

    

    subsample = 0.8

    colsample_bytree = 0.8

    start_time = time.time()



    print('XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth, subsample, colsample_bytree))

    params = {

        "objective": "binary:logistic",

        "booster" : "gbtree",

        "eval_metric": "auc",

        "eta": eta,

        "tree_method": 'exact',

        "max_depth": max_depth,

        "subsample": subsample,

        "colsample_bytree": colsample_bytree,

        "silent": 1,

        "seed": random_state,

    }

    num_boost_round = 115

    early_stopping_rounds = 10

    test_size = 0.1



    X_train, X_valid = train_test_split(train, test_size=test_size, random_state=random_state)

    print('Length train:', len(X_train.index))

    print('Length valid:', len(X_valid.index))

    y_train = X_train[target]

    y_valid = X_valid[target]

    dtrain = xgb.DMatrix(X_train[features], y_train)

    dvalid = xgb.DMatrix(X_valid[features], y_valid)



    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=True)



    print("Validating...")

    check = gbm.predict(xgb.DMatrix(X_valid[features]), ntree_limit=gbm.best_iteration+1)

    score = roc_auc_score(X_valid[target].values, check)

    print('Check error value: {:.6f}'.format(score))



    imp = get_importance(gbm, features)

    print('Importance array: ', imp)



    print("Predict test set...")

    test_prediction = gbm.predict(xgb.DMatrix(test[features]), ntree_limit=gbm.best_iteration+1)



    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))

    return test_prediction.tolist(), score





def simple_load():



    print("Read people.csv...")

    people = pd.read_csv("../input/people.csv",

                       dtype={'people_id': np.str,

                              'activity_id': np.str,

                              'char_38': np.int32},

                       parse_dates=['date'])



    print("Load train.csv...")

    train = pd.read_csv("../input/act_train.csv",

                        dtype={'people_id': np.str,

                               'activity_id': np.str,

                               'outcome': np.int8},

                        parse_dates=['date'])



    print("Load test.csv...")

    test = pd.read_csv("../input/act_test.csv",

                       dtype={'people_id': np.str,

                              'activity_id': np.str},

                       parse_dates=['date'])



    print("Process tables...")

    for table in [train, test]:

        table['activity_category'] = table['activity_category'].str.lstrip('type ').astype(np.int32)

        for i in range(1, 11):

            table['char_' + str(i)].fillna('type -999', inplace=True)

            table['char_' + str(i)] = table['char_' + str(i)].str.lstrip('type ').astype(np.int32)

    people['group_1'] = people['group_1'].str.lstrip('group ').astype(np.int32)

    for i in range(1, 10):

        people['char_' + str(i)] = people['char_' + str(i)].str.lstrip('type ').astype(np.int32)

    for i in range(10, 38):

        people['char_' + str(i)] = people['char_' + str(i)].astype(np.int32)



    print("Merge...")

    train = train.merge(people, on="people_id", suffixes=("_act", ""))

    test = test.merge(people, on="people_id", suffixes=("_act", ""))

    

    # Set index to activity id

    train = train.set_index("activity_id")

    test = test.set_index("activity_id")



    # Correct some data types

    for field in ["date_act", "date"]:

        train[field] = pd.to_datetime(train[field])

        test[field] = pd.to_datetime(test[field])



    return train, test





def xgboost_process(train,test,features):

    print("Process tables... ")

    for table in [train, test]:

        table['year'] = table['date'].dt.year

        table['month'] = table['date'].dt.month

        table['day'] = table['date'].dt.day

        table.drop('date', axis=1, inplace=True)

    features.remove('date')

    features.remove('date_act')

    return train, test, features

    



def model():



    # Load in the data set simply by merging together

    train, test = simple_load()

    

    # Get features

    features = get_features(train,test)

    

    # XGBoost processing

    train, test, features = xgboost_process(train, test, features)

    

    #test_prediction, score = xgboost_model(train, test, features, 'outcome')

    #create_submission(score, test, test_prediction)

    

    param_test1 = {

        'max_depth':list(range(3,10,2)),

        'min_child_weight':list(range(1,6,2))

    }

    gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5, min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

    gsearch1.fit(train[features],train['outcome'])

    gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

    

    

    

    



def main():



    # Write a benchmark file to the submissions folder

    model()

if __name__ == "__main__":

    main()