# coding: utf-8

__author__ = 'ZFTurbo & anokas :)'



import datetime

import pandas as pd

import numpy as np

import xgboost as xgb

from sklearn.cross_validation import KFold

from sklearn.metrics import roc_auc_score

from scipy.io import loadmat

from operator import itemgetter

import random

import os

import time

import glob



random.seed(2016)

np.random.seed(2016)
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



def intersect(a, b):

    return list(set(a) & set(b))





def print_features_importance(imp):

    for i in range(len(imp)):

        print("# " + str(imp[i][1]))

        print('output.remove(\'' + imp[i][0] + '\')')



def create_submission(score, test, prediction):

    # Make Submission

    now = datetime.datetime.now()

    sub_file = 'submission_' + str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'

    print('Writing submission: ', sub_file)

    f = open(sub_file, 'w')

    f.write('File,Class\n')

    total = 0

    for id in test['Id']:

        patient = id // 100000

        fid = id % 100000

        str1 = str(patient) + '_' + str(fid) + '.mat' + ',' + str(prediction[total])

        str1 += '\n'

        total += 1

        f.write(str1)

    f.close()



def get_features(train, test):

    trainval = list(train.columns.values)

    testval = list(test.columns.values)

    output = intersect(trainval, testval)

    output.remove('Id')

    # output.remove('file_size')

    return sorted(output)



def read_test_train():

    print("Load train.csv...")

    train = pd.read_csv("simple_train.csv")

    print("Load test.csv...")

    test = pd.read_csv("simple_test.csv")

    print("Process tables...")

    features = get_features(train, test)

    return train, test, features
def mat_to_pandas(path):

    mat = loadmat(path)

    names = mat['dataStruct'].dtype.names

    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}

    return pd.DataFrame(ndata['data'], columns=ndata['channelIndices'][0])



def create_training_data():

    print('Create train.csv...')

    files = sorted(glob.glob("../input/train_*/*.mat"))

    out = open("simple_train.csv", "w")

    out.write("Id,patient_id")

    for i in range(16):

        out.write(",avg_" + str(i))

        out.write(",median_" + str(i))

        out.write(",absavg_" + str(i))

        out.write(",stdev_" + str(i))

        out.write(",nonzero_" + str(i))

        out.write(",max_" + str(i))

        out.write(",min_" + str(i))

        out.write(",spread_" + str(i))

    out.write(",file_size,result\n")

    len_files = len(files)

    i = 0

    for fl in files:

        i += 1

        if i % 100 == 0:

            print('Read', i, 'of', len_files, '(' + str(round((i / len_files) * 100, 2)) + '%)')

        id_str = os.path.basename(fl)[:-4]

        arr = id_str.split("_")

        patient = int(arr[0])

        id = int(arr[1])

        result = int(arr[2])

        new_id = patient*100000 + id

        try:

            tables = mat_to_pandas(fl)

        except:

            continue

        out.write(str(new_id))

        out.write("," + str(patient))

        for f in sorted(list(tables.columns.values)):

            mean1 = tables[f].mean()

            median1 = np.median(tables[f])

            absmea1 = np.mean(np.abs(tables[f]))

            stdev1 = np.std(tables[f])

            count_nonzero1 = np.count_nonzero(tables[f])

            max1 = np.max(tables[f])

            min1 = np.min(tables[f])

            spread1 = max1-min1

            out.write("," + str(mean1))

            out.write("," + str(median1))

            out.write("," + str(absmean1))

            out.write("," + str(stdev1))

            out.write("," + str(count_nonzero1))

            out.write("," + str(max1))

            out.write("," + str(min1))

            out.write("," + str(spread1))

        out.write("," + str(os.path.getsize(fl)))

        out.write("," + str(result) + "\n")

        # break

    out.close()

    

def create_testing_data():

    print('Create test.csv...')

    files = sorted(glob.glob("../input/test_*/*.mat"))

    out = open("simple_test.csv", "w")

    out.write("Id,patient_id")

    for i in range(16):

        out.write(",avg_" + str(i))

        out.write(",median_" + str(i))

        out.write(",absavg_" + str(i))

        out.write(",stdev_" + str(i))

        out.write(",nonzero_" + str(i))

        out.write(",max_" + str(i))

        out.write(",min_" + str(i))

        out.write(",spread_" + str(i))

    out.write(",file_size\n")

    len_files = len(files)

    i = 0

    for fl in files:

        i += 1

        if i % 100 == 0:

            print('Read', i, 'of', len_files, '(' + str(round((i / len_files) * 100, 2)) + '%)')

        id_str = os.path.basename(fl)[:-4]

        arr = id_str.split("_")

        patient = int(arr[0])

        id = int(arr[1])

        new_id = patient*100000 + id

        try:

            tables = mat_to_pandas(fl)

        except:

            continue

        out.write(str(new_id))

        out.write("," + str(patient))

        for f in sorted(list(tables.columns.values)):

            mean1 = tables[f].mean()

            median1 = np.median(tables[f])

            absmea1 = np.mean(np.abs(tables[f]))

            stdev1 = np.std(tables[f])

            count_nonzero1 = np.count_nonzero(tables[f])

            max1 = np.max(tables[f])

            min1 = np.min(tables[f])

            spread1 = max1-min1

            out.write("," + str(mean1))

            out.write("," + str(median1))

            out.write("," + str(absmean1))

            out.write("," + str(stdev1))

            out.write("," + str(count_nonzero1))

            out.write("," + str(max1))

            out.write("," + str(min1))

            out.write("," + str(spread1))

            out.write("," + str(os.path.getsize(fl)))

        out.write("\n")

        # break

    out.close()
create_training_data()
create_testing_data()
def run_single(train, test, features, target, random_state=1):

    eta = 0.02

    max_depth = 5

    subsample = 0.9

    colsample_bytree = 0.9

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

    num_boost_round = 1000

    early_stopping_rounds = 50

    test_size = 0.2



    kf = KFold(len(train.index), n_folds=int(round(1/test_size, 0)), shuffle=True, random_state=random_state)

    train_index, test_index = list(kf)[0]

    print('Length of train: {}'.format(len(train_index)))

    print('Length of valid: {}'.format(len(test_index)))



    X_train, X_valid = train[features].as_matrix()[train_index], train[features].as_matrix()[test_index]

    y_train, y_valid = train[target].as_matrix()[train_index], train[target].as_matrix()[test_index]



    print('Length train:', len(X_train))

    print('Length valid:', len(X_valid))



    dtrain = xgb.DMatrix(X_train, y_train)

    dvalid = xgb.DMatrix(X_valid, y_valid)



    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist,

                    early_stopping_rounds=early_stopping_rounds, verbose_eval=True)



    print("Validating...")

    check = gbm.predict(xgb.DMatrix(X_valid), ntree_limit=gbm.best_iteration+1)

    score = roc_auc_score(y_valid, check)

    print('Check error value: {:.6f}'.format(score))



    imp = get_importance(gbm, features)

    print('Importance array: ', imp)



    print("Predict test set...")

    test_prediction = gbm.predict(xgb.DMatrix(test[features].as_matrix()), ntree_limit=gbm.best_iteration+1)



    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))

    return test_prediction.tolist(), score
train, test, features = read_test_train()

print('Length of train: ', len(train))

print('Length of test: ', len(test))

print('Features [{}]: {}'.format(len(features), sorted(features)))

test_prediction, score = run_single(train, test, features, 'result')

create_submission(score, test, test_prediction)