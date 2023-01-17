import pandas as pd

import numpy as np

import datetime as dt

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

import shap

warnings.filterwarnings("ignore")



from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV

from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier, IsolationForest

from sklearn.pipeline import Pipeline

from sklearn.tree import export_graphviz

import pydot



from imblearn.over_sampling import SMOTE, RandomOverSampler

from imblearn.under_sampling import RandomUnderSampler

import imblearn.pipeline



from xgboost import XGBClassifier

import xgboost as xgb

from xgboost import plot_tree



from hyperopt import hp

from hyperopt import fmin, tpe



%matplotlib inline

plt.style.use('ggplot')
shap.initjs()
df = pd.read_csv('../input/creditcardfraud/creditcard.csv')
df.shape
df.head()
# Convert all columns to lower case

df.columns = [col.lower() for col in df.columns]



df.head()
# Check for nulls

df.info()
df.describe()
# Define all features

orig_feats = [col for col in df.columns if 'class' not in col]



len(orig_feats)
sns.pairplot(df[orig_feats[:10]+['class']].sample(10000, random_state=1), hue='class')
sns.pairplot(df[orig_feats[10:20]+['class']].sample(10000, random_state=1), hue='class')
sns.pairplot(df[orig_feats[20:]+['class']].sample(10000, random_state=1), hue='class')
target = 'class'
# Only 0.17% of the dataset is labelled as fraud

df[target].value_counts(normalize=True)
df[target].hist()
df_corr = df.corr()
df_corr.style.background_gradient().set_precision(2)
df.corr()['class'].sort_values()
X = df[orig_feats]

y = df[target]
# Use stratify to ensure samples of fraud label are in the test set

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, stratify=y, random_state=1)



print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
y_train.value_counts(normalize=True)
y_test.value_counts(normalize=True)
naive_preds = np.zeros(y_test.shape[0])

len(naive_preds)
roc_auc_score(y_test, naive_preds)
print(classification_report(y_test, naive_preds))
num_boost_rounds = 1000

early_stopping_rounds = 10



initial_params = {'objective': 'binary:logistic', 'eval_metric': 'auc'}
logit_cv = cross_val_score(estimator=LogisticRegression(), 

                            X=X_train, 

                            y=y_train, 

                            scoring='roc_auc',

                            cv=StratifiedKFold(n_splits=5, random_state=1))
rf_cv = cross_val_score(estimator=RandomForestClassifier(),

                         X=X_train, 

                         y=y_train,

                         scoring='roc_auc',

                         cv=StratifiedKFold(n_splits=5, random_state=1))
xgb_cv = cross_val_score(estimator=XGBClassifier(num_boost_rounds=num_boost_rounds, 

                                                 early_stopping_rounds=early_stopping_rounds,

                                                 **initial_params),

                          X=X_train,

                          y=y_train,

                          scoring='roc_auc',

                          cv=StratifiedKFold(n_splits=5, random_state=1))
print(f'Logistic Regression CV Mean AUC score: {logit_cv.mean()}')

print(f'Random Forest CV Mean AUC score: {rf_cv.mean()}')

print(f'XGBoost CV Mean AUC score: {xgb_cv.mean()}')

model_cv_results = {'logit': logit_cv, 'random_forest': rf_cv, 'xgb': xgb_cv}
fig, ax = plt.subplots(figsize=(10,8))

plt.boxplot(model_cv_results.values())

ax.set_xticklabels(model_cv_results.keys())

plt.title('AUC scores using Different Classification Algorithms')
oversamp_pipeline = imblearn.pipeline.Pipeline([('oversample', RandomOverSampler(random_state=42)),

                                                ('xgb', XGBClassifier(num_boost_rounds=num_boost_rounds, 

                                                                      early_stopping_rounds=early_stopping_rounds,

                                                                      **initial_params))])
undersamp_pipeline = imblearn.pipeline.Pipeline([('undersample', RandomUnderSampler(random_state=42)),

                                                 ('xgb', XGBClassifier(num_boost_rounds=num_boost_rounds, 

                                                                       early_stopping_rounds=early_stopping_rounds,

                                                                       **initial_params))])
smote_pipeline = imblearn.pipeline.Pipeline([('smote', SMOTE(random_state=42)),

                                             ('xgb', XGBClassifier(num_boost_rounds=num_boost_rounds, 

                                                                   early_stopping_rounds=early_stopping_rounds,

                                                                   **initial_params))])
sampling_methods = {'random_oversampling': oversamp_pipeline,

                    'random_undersampling': undersamp_pipeline,

                    'smote': smote_pipeline}
sampling_cv_results = {}



for method, pipeline in sampling_methods.items():

    cv_results = cross_val_score(pipeline, X_train, y_train, cv=StratifiedKFold(n_splits=5, random_state=1), scoring='roc_auc')

    sampling_cv_results[method] = cv_results

    print(method, cv_results.mean())
fig, ax = plt.subplots(figsize=(10,8))

plt.boxplot(sampling_cv_results.values())

ax.set_xticklabels(sampling_cv_results.keys())

plt.title('AUC scores of XGB Classifier using Different Sampling Methods')
# Set up grid for hyperopt

space = {

    'max_depth': hp.quniform('max_depth', 4, 10, 2),

    'min_child_weight': hp.quniform('min_child_weight', 1, 20, 2),

    'gamma': hp.quniform('gamma', 0, 5, 0.5),

    'subsample': hp.uniform('subsample', 0.5, 0.9),

    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 0.9),

    'eta': hp.uniform('eta', 0.01, 0.3),

    'objective': 'binary:logistic',

    'eval_metric': 'auc'

}
# Objective function

def objective(params):

    params = {'max_depth': int(params['max_depth']),

              'min_child_weight': int(params['min_child_weight']),

              'gamma': params['gamma'],

              'subsample': params['subsample'],

              'colsample_bytree': params['colsample_bytree'],

              'eta': params['eta'],

              'objective': params['objective'],

              'eval_metric': params['eval_metric']}

    

    xgb_clf = XGBClassifier(num_boost_rounds=num_boost_rounds, 

                            early_stopping_rounds=early_stopping_rounds,

                            **params)

    

    best_score = cross_val_score(xgb_clf, X_train, y_train, scoring='roc_auc', cv=5, n_jobs=3).mean()

    

    loss = 1 - best_score 

    

    return loss
best_result = fmin(fn=objective, space=space, max_evals=20, 

                   rstate=np.random.RandomState(42), algo=tpe.suggest)
best_result
best_params = best_result

best_params['max_depth'] = int(best_params['max_depth'])

best_params['min_child_weight'] = int(best_params['min_child_weight'])

best_params['gamma'] = best_params['gamma']

best_params['colsample_bytree'] = round(best_params['colsample_bytree'], 1)

best_params['eta'] = round(best_params['eta'], 1)

best_params['subsample'] = round(best_params['subsample'], 1)

best_params
final_model = imblearn.pipeline.Pipeline([('smote', SMOTE(random_state=1)),

                                          ('xgb', XGBClassifier(num_boost_rounds=num_boost_rounds,

                                                                early_stopping_rounds=early_stopping_rounds, 

                                                                **best_params))])
final_model.fit(X_train, y_train)
final_preds = final_model.predict_proba(X_test)[:,1]
test = pd.merge(X_test, y_test, left_index=True, right_index=True)

test.head()
test.shape
df_preds = test.copy()

df_preds['fraud_score'] = final_preds
df_preds.head()
df_preds['fraud_score'].describe()
df_preds.to_csv('./final_model_preds.csv', index=False)
roc_auc_score(y_test, final_preds)
print(classification_report(y_test, final_preds.round()))
explainer = shap.TreeExplainer(final_model[1])
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
shap.summary_plot(shap_values, X_test, plot_type='bar')