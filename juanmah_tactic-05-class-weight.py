import pip._internal as pip

pip.main(['install', '--upgrade', 'numpy==1.17.2'])

import numpy as np

import pandas as pd



import warnings

warnings.filterwarnings('ignore')



from lightgbm import LGBMClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.experimental import enable_hist_gradient_boosting

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier

from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier

from sklearn.ensemble import IsolationForest

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.model_selection import cross_val_predict, cross_val_score, GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.utils.multiclass import unique_labels

from xgboost import XGBClassifier



import time

import pickle

from plotly.subplots import make_subplots

import plotly.graph_objs as go



from lwoku import get_prediction

from grid_search_utils import plot_grid_search, table_grid_search
# Read training and test files

X_train = pd.read_csv('../input/learn-together/train.csv', index_col='Id', engine='python')

X_test = pd.read_csv('../input/learn-together/test.csv', index_col='Id', engine='python')



# Define the dependent variable

y_train = X_train['Cover_Type'].copy()



# Define a training set

X_train = X_train.drop(['Cover_Type'], axis='columns')
prediction = pd.read_csv('../input/tactic-02-stack-classifiers/submission_lg.csv', index_col='Id', engine='python')

train_frequency = y_train.value_counts()

test_frequency = prediction['Cover_Type'].value_counts()

weight = (test_frequency / test_frequency.min()).to_dict()



fig = make_subplots(rows=1, cols=2)

fig.add_trace(go.Bar(

                     name='Train',

                     x=train_frequency.index,

                     y=train_frequency

                    ),

              row=1,

              col=1)

fig.add_trace(go.Bar(

                     name='Predicted test',

                     x=test_frequency.index,

                     y=test_frequency

                    ),

              row=1,

              col=2)

fig.update_xaxes(title_text='Cover type', dtick = 1)

fig.update_yaxes(title_text='Frequency')

fig.update_layout(width=2 * 360 + 100,

                  height=360,

                  title='Frequency of cover type in train and predicted test',

                  hovermode='closest',

                  template='none'

                 )

fig.show()
with open('../input/tactic-03-hyperparameter-optimization-lr/clf_liblinear.pickle', 'rb') as fp:

    clf = pickle.load(fp)

lr_li_w_clf = clf.best_estimator_

lr_li_w_clf.class_weight = weight

lr_li_w_clf
with open('../input/tactic-03-hyperparameter-optimization-lr/clf_saga.pickle', 'rb') as fp:

    clf = pickle.load(fp)

lr_sa_w_clf = clf.best_estimator_

lr_sa_w_clf.class_weight = weight

lr_sa_w_clf
with open('../input/tactic-03-hyperparameter-optimization-lr/clf_sag.pickle', 'rb') as fp:

    clf = pickle.load(fp)

lr_sg_w_clf = clf.best_estimator_

lr_sg_w_clf.class_weight = weight

lr_sg_w_clf
with open('../input/tactic-03-hyperparameter-optimization-lr/clf_lbfgs.pickle', 'rb') as fp:

    clf = pickle.load(fp)

lr_lb_w_clf = clf.best_estimator_

lr_lb_w_clf.class_weight = weight

lr_lb_w_clf
with open('../input/tactic-03-hyperparameter-optimization-lr/clf_newton-cg.pickle', 'rb') as fp:

    clf = pickle.load(fp)

lr_w_clf = clf.best_estimator_

lr_w_clf.class_weight = weight

lr_w_clf
# with open('../input/tactic-03-hyperparameter-optimization-svc/clf_weighted.pickle', 'rb') as fp:

#     svc_w_clf = pickle.load(fp)

# svc_w_clf.class_weight = weight

# svc_w_clf
with open('../input/tactic-03-hyperparameter-optimization-xtra-trees/clf.pickle', 'rb') as fp:

    clf = pickle.load(fp)

xt_w_clf = clf.best_estimator_

xt_w_clf.class_weight = weight

xt_w_clf
with open('../input/tactic-03-hyperparameter-optimization-rf/clf.pickle', 'rb') as fp:

    clf = pickle.load(fp)

rf_w_clf = clf.best_estimator_

rf_w_clf.class_weight = weight

rf_w_clf
weight_0 = weight.copy()

for k in range(1, 8):

    weight_0[k - 1] = weight_0.pop(k)

weight_0
with open('../input/tactic-03-hyperparameter-optimization-lightgbm/clf.pickle', 'rb') as fp:

    clf = pickle.load(fp)

lg_w_clf = clf.best_estimator_

lg_w_clf.class_weight = weight_0   

lg_w_clf
models = [

          ('lr_li-w', lr_li_w_clf),

          ('lr_sa-w', lr_sa_w_clf),

          ('lr_sg-w', lr_sg_w_clf),

          ('lr_lb-w', lr_lb_w_clf),

          ('lr-w', lr_w_clf),

#           ('svc-w', svc_w_clf),

          ('xt-w', xt_w_clf),

          ('rf-w', rf_w_clf),

          ('lg-w', lg_w_clf),

]
results = pd.DataFrame(columns = ['Model',

                                  'Accuracy',

                                  'Fit time',

                                  'Predict test set time',

                                  'Predict train set time'])



for name, model in models:

    print(name)



    # Fit

    t0 = time.time()

    model.fit(X_train, y_train)

    t1 = time.time()

    t_fit = (t1 - t0)

    

    # Predict test set

    t0 = time.time()

    y_test_pred = pd.Series(model.predict(X_test), index=X_test.index)

    t1 = time.time()

    t_test_pred = (t1 - t0)



    # Predict train set

    t0 = time.time()

    y_train_pred = pd.Series(get_prediction(model, X_train, y_train), index=X_train.index)

    accuracy = accuracy_score(y_train, y_train_pred)

    t1 = time.time()

    t_train_pred = (t1 - t0)



    # Submit

    y_train_pred.to_csv('train_' + name + '.csv', header=['Cover_Type'], index=True, index_label='Id')

    y_test_pred.to_csv('submission_' + name + '.csv', header=['Cover_Type'], index=True, index_label='Id')

    print('\n')

    

    results = results.append({

        'Model': name,

        'Accuracy': accuracy,

        'Fit time': t_fit,

        'Predict test set time': t_test_pred,

        'Predict train set time': t_train_pred

    }, ignore_index = True)
results = results.sort_values('Accuracy', ascending=False).reset_index(drop=True)

results.to_csv('results.csv', index=True, index_label='Id')

results
tactic_03_results = pd.read_csv('../input/tactic-03-hyperparameter-optimization/results.csv', index_col='Id', engine='python')

tactic_03_results
comparison = pd.DataFrame(columns = ['Model',

                                     'Accuracy',

                                     'Fit time',

                                     'Predict test set time',

                                     'Predict train set time'])



def get_increment(df1, df2, model, column):

    model1 = model.split('-', 1)[0]

    v1 = float(df1[df1['Model'] == model1][column])

    v2 = float(df2[df2['Model'] == model][column])

    return '{:.2%}'.format((v2 - v1) / v1)



for model in results['Model']:

    accuracy = get_increment(tactic_03_results, results, model, 'Accuracy')

    fit_time = get_increment(tactic_03_results, results, model, 'Fit time')

    predict_test_set_time = get_increment(tactic_03_results, results, model, 'Predict test set time')

    predict_train_set_time = get_increment(tactic_03_results, results, model, 'Predict train set time')

    comparison = comparison.append({

        'Model': model,

        'Accuracy': accuracy,

        'Fit time': fit_time,

        'Predict test set time': predict_test_set_time,

        'Predict train set time': predict_train_set_time

    }, ignore_index = True)    



comparison