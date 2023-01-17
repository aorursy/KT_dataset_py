from pandas import set_option

%matplotlib inline

from pathlib import Path

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

# for preprocessing the data

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import log_loss

# the model

from sklearn.linear_model import LogisticRegression



# for combining the preprocess with model training

from sklearn.pipeline import Pipeline



# for optimizing parameters of the pipeline

from sklearn.model_selection import GridSearchCV



import xgboost as xgb

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
train = pd.read_csv('../input/train_values.csv',index_col='patient_id')
train_labels = pd.read_csv('../input/train_labels.csv',index_col='patient_id')
train.shape
test = pd.read_csv('../input/test_values.csv', index_col='patient_id')
train.head()
train.shape
train.dtypes
set_option('display.width', 100)

set_option('precision', 3)

train.describe()
train_labels.head()
test.head()
train_labels.shape
test.shape
train_labels.groupby('heart_disease_present').size()
train_labels.heart_disease_present.value_counts().plot.bar(title='Number with Heart Disease')
selected_features = ['age', 

                     'sex', 

                     'max_heart_rate_achieved', 

                     'resting_blood_pressure']

#train_values_subset = train[selected_features]
sns.pairplot(train.join(train_labels), 

             hue='heart_disease_present', 

             vars=selected_features)
set_option('display.width', 100)

set_option('precision', 3)

correlations = train.corr(method='pearson')

correlations
sns.heatmap(train.corr(method='pearson'),annot=True, cmap='terrain', linewidths=0.1)

fig=plt.gcf()

fig.set_size_inches(8,6)

plt.show()
sns.pairplot(train)

plt.show()
train.skew()
train.isnull().sum()

test.isnull().sum()
one_hot_encoded_training_predictors = pd.get_dummies(train)

one_hot_encoded_test_predictors = pd.get_dummies(test)

train,test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,

                                                                    join='left', 

                                                                    axis=1)
from xgboost import plot_importance

model = xgb.XGBClassifier()

model.fit(train, train_labels.heart_disease_present)

# plot feature importance

plot_importance(model)

plt.show()
from numpy import sort

from sklearn.feature_selection import SelectFromModel

X_train, X_test, y_train, y_test = train_test_split(train, train_labels.heart_disease_present, test_size=0.10, random_state=7)

# fit model on all training data

model = xgb.XGBClassifier()

model.fit(X_train, y_train)

# make predictions for test data and evaluate

y_pred = model.predict(X_test)

predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))

# Fit model using each importance as a threshold

thresholds = sort(model.feature_importances_)

for thresh in thresholds:

	# select features using threshold

	selection = SelectFromModel(model, threshold=thresh, prefit=True)

	select_X_train = selection.transform(X_train)

	# train model

	selection_model = xgb.XGBClassifier()

	selection_model.fit(select_X_train, y_train)

	# eval model

	select_X_test = selection.transform(X_test)

	y_pred = selection_model.predict(select_X_test)

	predictions = [round(value) for value in y_pred]

	accuracy = accuracy_score(y_test, predictions)

	print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))
train_df = train
train_X, test_X, train_Y, test_Y = train_test_split(train, train_labels.heart_disease_present, test_size=0.10, stratify=train_labels.heart_disease_present, random_state=42)



print(train_X.shape, test_X.shape)

print()

print('Number of rows in Train dataset:',train_X.shape[0])

#print(train_Y['heart_disease_present'].value_counts())

print()

print('Number of rows in Test dataset:',test_X.shape[0])

#print(test_Y['heart_disease_present'].value_counts())
scaler = StandardScaler()

train_X = scaler.fit_transform(train_X)

test_X = scaler.transform(test_X)
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold

model = xgb.XGBClassifier()

n_estimators = range(50, 400, 50)

param_grid = dict(n_estimators=n_estimators)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)

grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)

grid_result = grid_search.fit(train_X, train_Y.values.ravel())

# summarize results

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

	print("%f (%f) with: %r" % (mean, stdev, param))

# plot

plt.errorbar(n_estimators, means, yerr=stds)

plt.title("XGBoost n_estimators vs Log Loss")

plt.xlabel('n_estimators')

plt.ylabel('Log Loss')

plt.savefig('n_estimators.png')
# grid search

model = xgb.XGBClassifier()

max_depth = range(1, 11, 2)

print(max_depth)

param_grid = dict(max_depth=max_depth)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)

grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold, verbose=1)

grid_result = grid_search.fit(train_X, train_Y.values.ravel())

# summarize results

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

	print("%f (%f) with: %r" % (mean, stdev, param))

# plot

plt.errorbar(max_depth, means, yerr=stds)

plt.title("XGBoost max_depth vs Log Loss")

plt.xlabel('max_depth')

plt.ylabel('Log Loss')

plt.savefig('max_depth.png')
%%time



model_base = xgb.XGBClassifier(max_depth=1,

                        subsample=0.33,

                        objective='binary:logistic',

                        n_estimators=50,

                        learning_rate = 0.12)

eval_set = [(train_X, train_Y), (test_X, test_Y)]

model_base.fit(train_X, train_Y.values.ravel(), early_stopping_rounds=15, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=True)
# make predictions for test data

y_pred = model_base.predict(test_X)

predictions = [round(value) for value in y_pred]
# evaluate predictions

accuracy = accuracy_score(test_Y, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
# retrieve performance metrics

results = model_base.evals_result()

epochs = len(results['validation_0']['error'])

x_axis = range(0, epochs)

# plot log loss

fig, ax = plt.subplots()

ax.plot(x_axis, results['validation_0']['logloss'], label='Train')

ax.plot(x_axis, results['validation_1']['logloss'], label='Test')

ax.legend()

plt.ylabel('Log Loss')

plt.title('XGBoost Log Loss')

plt.show()

# plot classification error

fig, ax = plt.subplots()

ax.plot(x_axis, results['validation_0']['error'], label='Train')

ax.plot(x_axis, results['validation_1']['error'], label='Test')

ax.legend()

plt.ylabel('Classification Error')

plt.title('XGBoost Classification Error')

plt.show()
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

from sklearn.metrics import roc_auc_score
train = train_X

valid = test_X



y_train = train_Y

y_valid = test_Y

def objective(space):



    clf = xgb.XGBClassifier(n_estimators = 50,

                            max_depth = space['max_depth'],

                            min_child_weight = space['min_child_weight'],

                            subsample = space['subsample'])



    eval_set  = [( train, y_train), ( valid, y_valid)]



    clf.fit(train, y_train,

            eval_set=eval_set, eval_metric="auc",

            early_stopping_rounds=30)



    pred = clf.predict_proba(valid)[:,1]

    auc = roc_auc_score(y_valid, pred)

    print ("SCORE:", auc)



    return{'loss':1-auc, 'status': STATUS_OK }





space ={

        'max_depth': hp.choice("x_max_depth", np.arange(5, 25, dtype=int)),

        'min_child_weight': hp.choice ('x_min_child',np.arange(1, 10, dtype=int)),

        'subsample': hp.uniform ('x_subsample', 0.8, 1)

    }
trials = Trials()

best = fmin(fn=objective,

            space=space,

            algo=tpe.suggest,

            max_evals=100,

            trials=trials)



print (best)
%%time



model = xgb.XGBClassifier(max_depth=19,

                        subsample=0.8325428224507427,

                          min_child = 6,

                        objective='binary:logistic',

                        n_estimators=50,

                        learning_rate = 0.1)

eval_set = [(train_X, train_Y), (test_X, test_Y)]

model.fit(train_X, train_Y.values.ravel(), early_stopping_rounds=15, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=True)
# make predictions for test data

y_pred_hyperopt = model.predict(test_X)

predictions = [round(value) for value in y_pred_hyperopt]
# evaluate predictions

accuracy = accuracy_score(test_Y, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
# retrieve performance metrics

results = model.evals_result()

epochs = len(results['validation_0']['error'])

x_axis = range(0, epochs)

# plot log loss

fig, ax = plt.subplots()

ax.plot(x_axis, results['validation_0']['logloss'], label='Train')

ax.plot(x_axis, results['validation_1']['logloss'], label='Test')

ax.legend()

plt.ylabel('Log Loss')

plt.title('XGBoost Log Loss')

plt.show()

# plot classification error

fig, ax = plt.subplots()

ax.plot(x_axis, results['validation_0']['error'], label='Train')

ax.plot(x_axis, results['validation_1']['error'], label='Test')

ax.legend()

plt.ylabel('Classification Error')

plt.title('XGBoost Classification Error')

plt.show()
#X_train, X_test, y_train, y_test = train_test_split(train, train_labels.heart_disease_present,test_size=.1, random_state=42)



print(train_X.shape)

print(test_X.shape)



print(train_Y.shape)

print(test_Y.shape)
dtrain = xgb.DMatrix(train_X, label=train_Y)

dtest = xgb.DMatrix(test_X, label=test_Y)
params = {

    # Parameters that we are going to tune.

    'max_depth':6,

    'min_child_weight': 1,

    'eta':.3,

    'subsample': 1,

    'colsample_bytree': 1,

    # Other parameters

    'objective':'binary:logistic',

     'n_estimators' :50,

     'learning_rate' : 0.12

}
params['eval_metric'] = "logloss"
num_boost_round = 999
model = xgb.train(

    params,

    dtrain,

    num_boost_round=num_boost_round,

    evals=[(dtest, "Test")],

    early_stopping_rounds=10,

    

)

print("Best log loss: {:.2f} with {} rounds".format(

                 model.best_score,

                 model.best_iteration+1))
cv_results = xgb.cv(

    params,

    dtrain,

    num_boost_round=num_boost_round,

    seed=42,

    nfold=5,

    metrics={'logloss'},

    early_stopping_rounds=10

)

cv_results
cv_results['test-logloss-mean'].min()
gridsearch_params = [

    (max_depth, min_child_weight)

    for max_depth in range(1,25)

    for min_child_weight in range(3,20)

]
# Define initial best params and MAE

min_logloss = float("Inf")

best_params = None

for max_depth, min_child_weight in gridsearch_params:

    print("CV with max_depth={}, min_child_weight={}".format(

                             max_depth,

                             min_child_weight))

    # Update our parameters

    params['max_depth'] = max_depth

    params['min_child_weight'] = min_child_weight

    # Run CV

    cv_results = xgb.cv(

        params,

        dtrain,

        num_boost_round=num_boost_round,

        seed=42,

        nfold=5,

        metrics={'logloss'},

        early_stopping_rounds=10

    )

    # Update best MAE

    mean_logloss = cv_results['test-logloss-mean'].min()

    boost_rounds = cv_results['test-logloss-mean'].argmin()

    print("\tlogloss {} for {} rounds".format(min_logloss, boost_rounds))

    if mean_logloss < min_logloss:

        min_logloss = mean_logloss

        best_params = (max_depth,min_child_weight)

print("Best params: {}, {}, logloss: {}".format(best_params[0], best_params[1], min_logloss))
params['max_depth'] = 1

params['min_child_weight'] = 4
gridsearch_params = [

    (subsample, colsample)

    for subsample in [i/10. for i in range(1,10)]

    for colsample in [i/10. for i in range(1,10)]

]
min_logloss = float("Inf")

best_params = None

# We start by the largest values and go down to the smallest

for subsample, colsample in reversed(gridsearch_params):

    print("CV with subsample={}, colsample={}".format(

                             subsample,

                             colsample))

    # We update our parameters

    params['subsample'] = subsample

    params['colsample_bytree'] = colsample

    # Run CV

    cv_results = xgb.cv(

        params,

        dtrain,

        num_boost_round=num_boost_round,

        seed=42,

        nfold=5,

        metrics={'logloss'},

        early_stopping_rounds=10

    )

    # Update best score

    mean_logloss = cv_results['test-logloss-mean'].min()

    boost_rounds = cv_results['test-logloss-mean'].argmin()

    print("\tlogloss {} for {} rounds".format(mean_logloss, boost_rounds))

    if mean_logloss < min_logloss:

        min_logloss = mean_logloss

        best_params = (subsample,colsample)

print("Best params: {}, {}, logloss: {}".format(best_params[0], best_params[1], min_logloss))
params['subsample'] = 0.8

params['colsample_bytree'] = 0.1
%time

# This can take some time…

min_logloss = float("Inf")

best_params = None

for eta in [.3, .2, .1, .05, .01, .005]:

    print("CV with eta={}".format(eta))

    # We update our parameters

    params['eta'] = eta

    # Run and time CV

    %time cv_results = xgb.cv(params,dtrain,num_boost_round=num_boost_round,seed=42,nfold=5,metrics=['logloss'],early_stopping_rounds=10)

    # Update best score

    mean_logloss = cv_results['test-logloss-mean'].min()

    boost_rounds = cv_results['test-logloss-mean'].argmin()

    print("\tMAE {} for {} rounds\n".format(mean_logloss, boost_rounds))

    if mean_logloss < min_logloss:

        min_logloss = mean_logloss

        best_params = eta

print("Best params: {}, logloss: {}".format(best_params, min_logloss))
params['eta'] = 0.3
params
model = xgb.train(

    params,

    dtrain,

    num_boost_round=num_boost_round,

    evals=[(dtest, "Test")],

    early_stopping_rounds=30

)
num_boost_round = model.best_iteration + 1

best_model = xgb.train(

    params,

    dtrain,

    num_boost_round=num_boost_round,

    evals=[(dtest, "Test")]

)
print(train_X.shape)

print(test_X.shape)



print(train_Y.shape)

print(test_Y.shape)
xgb_model = xgb.XGBClassifier(params = params)

eval_set = [(train_X, train_Y), (test_X, test_Y)]

xgb_model.fit(train_X, train_Y.values.ravel(), early_stopping_rounds=15, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=True)
# make predictions for test data

y_pred_gs = xgb_model.predict(test_X)

predictions = [round(value) for value in y_pred_gs]
# evaluate predictions

accuracy = accuracy_score(test_Y, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
# retrieve performance metrics

results = xgb_model.evals_result()

epochs = len(results['validation_0']['error'])

x_axis = range(0, epochs)

# plot log loss

fig, ax = plt.subplots()

ax.plot(x_axis, results['validation_0']['logloss'], label='Train')

ax.plot(x_axis, results['validation_1']['logloss'], label='Test')

ax.legend()

plt.ylabel('Log Loss')

plt.title('XGBoost Log Loss')

plt.show()

# plot classification error

fig, ax = plt.subplots()

ax.plot(x_axis, results['validation_0']['error'], label='Train')

ax.plot(x_axis, results['validation_1']['error'], label='Test')

ax.legend()

plt.ylabel('Classification Error')

plt.title('XGBoost Classification Error')

plt.show()
###################################################################################
scaler = StandardScaler()

train_df = scaler.fit_transform(train_df)
in_sample_preds = xgb_model.predict_proba(train_df)
log_loss(train_labels.heart_disease_present, in_sample_preds)
test_values_subset = test
test_values_subset = scaler.transform(test_values_subset)
predictions = xgb_model.predict_proba(test_values_subset)[:, 1]
predictions = np.round(predictions, 2)
submission_format = pd.read_csv('../input/submission_format.csv', index_col='patient_id')
my_submission = pd.DataFrame(data=predictions,

                             columns=submission_format.columns,

                             index=submission_format.index)
my_submission.head(10)
my_submission.to_csv('submission_result.csv', index='patient_id')

submission = pd.read_csv('submission_result.csv')
submission.head()
sample_preds = model_base.predict_proba(train_df)
log_loss(train_labels.heart_disease_present, sample_preds)
predictions_base = model_base.predict_proba(test_values_subset)[:, 1]
predictions_base = np.round(predictions_base, 2)
submission_format_base = pd.read_csv('../input/submission_format.csv', index_col='patient_id')
submission_base = pd.DataFrame(data=predictions_base,

                             columns=submission_format_base.columns,

                             index=submission_format_base.index)
submission_base.head(10)
submission_base.to_csv('submission_baseline.csv', index='patient_id')

submission_b = pd.read_csv('submission_baseline.csv')
submission_b.head()