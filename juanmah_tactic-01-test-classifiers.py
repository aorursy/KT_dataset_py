import pip._internal as pip

pip.main(['install', '--upgrade', 'numpy==1.17.3'])

import numpy as np

import pandas as pd



import warnings

warnings.filterwarnings('ignore')



from lightgbm import LGBMClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.experimental import enable_hist_gradient_boosting

from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.ensemble import IsolationForest

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomTreesEmbedding

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.model_selection import cross_val_predict, cross_val_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.utils.multiclass import unique_labels

from xgboost import XGBClassifier



import time



from lwoku import get_prediction
N_ESTIMATORS = 2000

MIN_SAMPLE_LEAFS = 100

RANDOM_STATE = 42

N_JOBS = -1

VERBOSE = 0
# Read training and test files

X_train = pd.read_csv('../input/learn-together/train.csv', index_col='Id', engine='python')

X_test = pd.read_csv('../input/learn-together/test.csv', index_col='Id', engine='python')



# Define the dependent variable

y_train = X_train['Cover_Type'].copy()



# Define a training set

X_train = X_train.drop(['Cover_Type'], axis='columns')
lr_clf = LogisticRegression(verbose=VERBOSE,

                            random_state=RANDOM_STATE,

                            n_jobs=1)
lda_clf = LinearDiscriminantAnalysis()
knn_clf = KNeighborsClassifier(n_jobs=N_JOBS)
gnb_clf = GaussianNB()
svc_clf = SVC(random_state=RANDOM_STATE,

              verbose=True)
bg_clf = BaggingClassifier(n_estimators=N_ESTIMATORS,

                           verbose=VERBOSE,

                           random_state=RANDOM_STATE)
xt_clf = ExtraTreesClassifier(n_estimators=N_ESTIMATORS,

                              min_samples_leaf=MIN_SAMPLE_LEAFS,

                              verbose=VERBOSE,

                              random_state=RANDOM_STATE,

                              n_jobs=N_JOBS)
rf_clf = RandomForestClassifier(n_estimators=N_ESTIMATORS,

                                min_samples_leaf=MIN_SAMPLE_LEAFS,

                                verbose=VERBOSE,

                                random_state=RANDOM_STATE,

                                n_jobs=N_JOBS)
ab_clf = AdaBoostClassifier(n_estimators=N_ESTIMATORS,

                            random_state=RANDOM_STATE)
gb_clf = GradientBoostingClassifier(n_estimators=N_ESTIMATORS,

                              min_samples_leaf=MIN_SAMPLE_LEAFS,

                              verbose=VERBOSE,

                              random_state=RANDOM_STATE)
lg_clf = LGBMClassifier(n_estimators=N_ESTIMATORS,

                        num_leaves=MIN_SAMPLE_LEAFS,

                        verbosity=VERBOSE,

                        random_state=RANDOM_STATE,

                        n_jobs=N_JOBS)
xg_clf = XGBClassifier(random_state=RANDOM_STATE,

                       n_jobs=-N_JOBS,

                       learning_rate=0.1,

                       n_estimators=100,

                       max_depth=3)
models = [

          ('lr', lr_clf),

          ('lda', lda_clf),

          ('knn', knn_clf),

          ('gnb', gnb_clf),

          ('svc', svc_clf),

          ('bg', bg_clf),

          ('xt', xt_clf),

          ('rf', rf_clf),

          ('ab', ab_clf),

          ('gb', gb_clf),

          ('lg', lg_clf),

          ('xg', xg_clf)

]
results = pd.DataFrame(columns = ['Model',

                                  'Accuracy',

                                  'Fit time',

                                  'Predict test set time',

                                  'Predict train set time'])



for name, model in models:



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