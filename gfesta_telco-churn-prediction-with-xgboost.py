# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
telco_original = pd.read_csv('../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')



telco_original['TotalCharges'] = telco_original.TotalCharges.replace({' ': 0})

telco_original['TotalCharges'] = pd.to_numeric(telco_original.TotalCharges, errors='coerce')

# remove the 9 rows with missing values

print(telco_original.info())



telco_original = telco_original.drop('customerID', axis=1)



telco_original['Churn'] = telco_original.Churn.replace({'No': 0, 'Yes':1})



X, y = telco_original.drop('Churn', axis=1), telco_original.Churn
telco_original.nunique()
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split



# generate the list of categorical and numerical variables

categorical_variables = X.nunique()[X.nunique() < 5].keys().to_list()



numerical_variables=list(set(X.columns) - set(categorical_variables))



ohe = OneHotEncoder(drop='first', sparse=False)



X_ohe = ohe.fit_transform(X[categorical_variables])

X_ohe_df = pd.DataFrame(X_ohe, columns=ohe.get_feature_names(categorical_variables))



# Merging the transformed dataframe togheter

X = pd.merge(X[numerical_variables], X_ohe_df, left_index=True, right_index=True)



X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, 

                                                    test_size=0.2, random_state=123)
import xgboost as xgb

from hyperopt import STATUS_OK



train_dmatrix = xgb.DMatrix(data=X_train, label=y_train)



N_FOLDS = 5



# define objective to minimize

def objective(params, n_folds = N_FOLDS):

    params['objective'] = 'binary:logistic'

    # Perform cross-validation: cv_results

    cv_results = xgb.cv(dtrain=train_dmatrix, params=params,

                  nfold=n_folds, num_boost_round=10000, early_stopping_rounds=100, 

                  metrics="auc", as_pandas=True, seed=123)



    # Print the accuracy

    loss = 1 - cv_results["test-auc-mean"].iloc[-1]

    n_estimators = cv_results["test-auc-mean"].idxmax() + 1

    return {'loss': loss, 'params': params, 'n_estimators': n_estimators, 'status': STATUS_OK}
from hyperopt import hp

from hyperopt.pyll.stochastic import sample



hyperparameter_space = {

    'n_jobs': -1,

    'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),

    'subsample': hp.uniform('subsample', 0.6, 1.0),

    'min_child_weight': hp.quniform('min_child_weight', 1, 7, 2),

    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),

    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),

    'max_depth': hp.randint('max_depth', 1,16),

    'gamma': hp.uniform('gamma', 0.1,0.4),

    'max_delta_step': hp.randint('max_delta_step',0,10),

    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2))

}
from hyperopt import Trials



bayes_trials = Trials()
from hyperopt import fmin

from hyperopt import tpe



MAX_EVALS = 50



best = fmin(fn = objective, space = hyperparameter_space, algo = tpe.suggest, max_evals = MAX_EVALS,

           trials = bayes_trials, rstate = np.random.RandomState(50))
best
best['num_boost_round']=10000

best['early_stopping_rounds']=100



xgb_best = xgb.XGBClassifier(**best)



xgb_best.fit(X_train, y_train)
# Import roc_curve

from sklearn.metrics import roc_curve

import matplotlib.pyplot as plt



def add_roc_plot(model, test_x, test_y, legend_text):

    y_pred_prob = model.predict_proba(test_x)[:, 1]

    # Calculate the roc metrics

    fpr, tpr, thresholds = roc_curve(test_y, y_pred_prob)

    # Plot the ROC curve

    plt.plot(fpr, tpr, label=legend_text)

    plt.legend()



    

models_list = [xgb_best]

model_names = ['Extreme Gradient Boosting']



plt.figure(figsize=(6, 6))

[add_roc_plot(model, X_test, y_test, legend_text) for model, legend_text in zip(models_list, model_names)]



# Add labels and diagonal line

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.xlim((0,1))

plt.ylim((0,1))

plt.plot([0, 1], [0, 1], "k-")

plt.show()
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score



list_scores = [roc_auc_score, recall_score, precision_score, accuracy_score]

calc_scores = []

def compute_scores(model, x_test, y_test, scores):

    return [round(score(y_test, model.predict(x_test)), 2) for score in scores]

    

[calc_scores.append(compute_scores(model, X_test, y_test, list_scores)) for model in models_list] 



score_names = ['roc_auc', 'recall', 'precision', 'accuracy']

scores_df = pd.DataFrame(calc_scores, columns=score_names, index=model_names)



scores_df
import matplotlib.pyplot as plt



fig, ax = plt.subplots(1,1,figsize=(12,10))

xgb.plot_importance(xgb_best, ax = ax)

plt.show()