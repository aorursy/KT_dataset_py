# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost as xgb

from sklearn import model_selection, preprocessing, ensemble, metrics

from bayes_opt import BayesianOptimization
# read the data into pandas dataframe #

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



# Getting the id column #

train_id = train.id.values

test_id = test.id.values



# names of numerical and categorical columns #

num_cols = ['bone_length', 'rotting_flesh', 'hair_length', 'has_soul']

cat_cols = ['color']



# label encode the cat variable #

lbl = preprocessing.LabelEncoder()

lbl.fit(list(train['color'])+list(test['color']))

train['color'] = lbl.transform(list(train['color']))

test['color'] = lbl.transform(list(test['color']))

color_classes = lbl.classes_



# label encode the target variable #

lbl = preprocessing.LabelEncoder()

train['type'] = lbl.fit_transform(list(train['type']))

train_y = train.type.values

type_classes = lbl.classes_



# Get the train and test (X and y variables) #

train_X = np.array(train[num_cols+cat_cols])

test_X = np.array(test[num_cols+cat_cols])
# A helper function #

def cross_validation_genrator(splits=5, random_state=None):

    kf = model_selection.KFold(n_splits=splits, shuffle=True, random_state=random_state)

    for dev_index, val_index in kf.split(train_y):

        dev_X, val_X = train_X[dev_index,:], train_X[val_index,:]

        dev_y, val_y = train_y[dev_index], train_y[val_index]

        yield dev_X, dev_y, val_X, val_y

        

def run_rfcv(max_depth, min_samples_split, max_features):

    cv_scores = []

    for dev_X, dev_y, val_X, val_y in cross_validation_genrator(splits=5, random_state=0):

        model = ensemble.RandomForestClassifier(max_depth=int(max_depth),

                                                min_samples_split=int(min_samples_split),

                                                max_features=min(max_features, 0.999),

                                                n_estimators=100,

                                                n_jobs=7,

                                                random_state=1

                                                )

        model.fit(dev_X, dev_y)

        pred_val_y = model.predict(val_X)

        accuracy = metrics.accuracy_score(val_y, pred_val_y)

        cv_scores.append(accuracy)

    return np.mean(cv_scores)

    

def run_rf(max_depth, min_samples_split, max_features):

    model = ensemble.RandomForestClassifier(max_depth=int(max_depth),

                                            min_samples_split=int(min_samples_split),

                                            max_features=min(max_features, 0.999),

                                            n_estimators=100,

                                            n_jobs=7,

                                            random_state=1)

    model.fit(train_X, train_y)

    preds = model.predict(test_X)

    return preds

    

rf_params = {'max_depth': (25,40),

             'min_samples_split': (2, 25),

             'max_features': (0.1, 0.5)}

rf_bo = BayesianOptimization(run_rfcv, rf_params)

rf_bo.maximize(init_points=10, n_iter=25, acq='ei')



print("Final Result : ")

print("Best score : ",rf_bo.res['max']['max_val'])

rf_best_params = rf_bo.res['max']['max_params']

print("Best params : ",rf_best_params)

pred_test_y = run_rf(rf_best_params['max_depth'], rf_best_params['min_samples_split'], rf_best_params['max_features'])

pred_test_y = [type_classes[pred] for pred in pred_test_y]

out_df = pd.DataFrame({'id':test_id})

out_df['type'] = pred_test_y

out_df.to_csv("rf_bayesian_opt.csv", index=False)