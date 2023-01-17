import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Gradient Boosting
import lightgbm as lgb
import xgboost as xgb

# Scikit-learn
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Graphics
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Dataset
from sklearn.datasets import load_boston
# Uploading the Boston dataset
X, y = load_boston(return_X_y=True)

# Transforming the problem into a classification (unbalanced)
y_bin = (y > np.percentile(y, 90)).astype(int)
#CRIM - per capita crime rate by town
#ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
#INDUS - proportion of non-retail business acres per town.
#CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
#NOX - nitric oxides concentration (parts per 10 million)
#RM - average number of rooms per dwelling
#AGE - proportion of owner-occupied units built prior to 1940
#DIS - weighted distances to five Boston employment centres
#RAD - index of accessibility to radial highways
#TAX - full-value property-tax rate per $10,000
#PTRATIO - pupil-teacher ratio by town
#B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
#LSTAT - % lower status of the population
#MEDV - Median value of owner-occupied homes in $1000's this is our target variable
# Histogram highlighting the top 10% we use as a target
plt.hist(y[y <= np.percentile(y, 90)], bins='auto', alpha=0.7, label='0', color='b')
plt.hist(y[y > np.percentile(y, 90)], bins=8, alpha=0.7, label='1', color='r')
plt.title("Histogram of MEDV")
plt.legend(loc='upper right')
plt.show()
# For convenience, we will create a Pandas dataframe from X
train = pd.DataFrame(X)
train = train.add_prefix('var_')
# Checking about the shape of our training set
print(train.shape)
# Setting a 5-fold stratified cross-validation (note: shuffle=True)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=8)
MAX_ROUNDS = 2000
lgb_iter1 = []
sklearn_gbm_iter1 = []
xgb_gbm_iter1 = []

lgb_ap1 = []
sklearn_gbm_ap1 = []
xgb_gbm_ap1 = []
# Set up the classifier with standard configuration
# Later we will more performing parameters with Bayesian Optimization
params = {
    'learning_rate':  0.06, 
    'max_depth': 6, 
    #'lambda_l1': 16.7,
    'min_data_in_leaf':5,
    'boosting': 'gbdt', 
    'objective': 'binary', 
    'metric': 'auc',
    'feature_fraction': .9,
    'is_training_metric': False, 
    'seed': 1
}
for i, (train_index, test_index) in enumerate(skf.split(train,y_bin)):
    
    # Create data for this fold
    y_train, y_valid = y_bin[train_index], y_bin[test_index]
    X_train, X_valid = train.iloc[train_index,:], train.iloc[test_index,:]
        
    print( "\nFold ", i)

    # Running models for this fold
    
    # ->LightGBM
    lgb_gbm = lgb.train(params, 
                          lgb.Dataset(X_train, label=y_train), 
                          MAX_ROUNDS, 
                          lgb.Dataset(X_valid, label=y_valid), 
                          verbose_eval=False, 
                          #feval= auc, 
                          early_stopping_rounds=50)
    
    print( " Best iteration lgb = ", lgb_gbm.best_iteration)
    
    # ->Scikit-learn GBM
    sklearn_gbm = GradientBoostingClassifier(n_estimators=MAX_ROUNDS, 
                                    learning_rate = 0.06,
                                    max_features=2, 
                                    max_depth = 6, 
                                    n_iter_no_change=50, 
                                    tol=0.01,
                                    random_state = 0)
    
    sklearn_gbm.fit(X_train, y_train)
    print( " Best iteration sklearn_gbm = ", sklearn_gbm.n_estimators_)
    
    # ->XGBoost
    xgb_gbm = xgb.XGBClassifier(max_depth=6, 
                                n_estimators=MAX_ROUNDS,
                                eval_set=[(X_train, y_train), (X_valid, y_valid)],
                                learning_rate=0.06,
                                early_stopping_rounds=50)

    xgb_gbm.fit(X_train, y_train)
    
    print( " Best iteration xgboost_gbm = ", xgb_gbm.get_booster().best_iteration)
    
    # Storing and reporting results of the fold
    lgb_iter1 = np.append(lgb_iter1, lgb_gbm.best_iteration)
    sklearn_gbm_iter1 = np.append(sklearn_gbm_iter1, sklearn_gbm.n_estimators_)
    xgb_gbm_iter1 = np.append(xgb_gbm_iter1, xgb_gbm.get_booster().best_iteration)
   
    pred = lgb_gbm.predict(X_valid, num_iteration=lgb_gbm.best_iteration)
    ap = average_precision_score(y_valid, pred, average='macro', pos_label=1, sample_weight=None)
    print('lgb ', ap)
    lgb_ap1 = np.append(lgb_ap1, ap)
    
    pred = sklearn_gbm.predict(X_valid)
    ap = average_precision_score(y_valid, pred, average='macro', pos_label=1, sample_weight=None)
    print('sklearn_gbn ', ap)
    sklearn_gbm_ap1 = np.append(sklearn_gbm_ap1, ap)
    
    pred  = xgb_gbm.predict(X_valid)
    ap = average_precision_score(y_valid, pred, average='macro', pos_label=1, sample_weight=None)
    print('xgboost ', ap)
    xgb_gbm_ap1 = np.append(xgb_gbm_ap1, ap)
print('lgb_iter1: ', np.mean(lgb_iter1))
print('sklearn_gbm_iter1: ', np.mean(sklearn_gbm_iter1))
print('xgb_gbm_iter1: ',np.mean(xgb_gbm_iter1))

print('lgb_ap1: ', np.mean(lgb_ap1))
print('sklearn_gbm_ap1: ', np.mean(sklearn_gbm_ap1))
print('xgb_gbm_ap1: ', np.mean(xgb_gbm_ap1))
poly = PolynomialFeatures(2)
poly_train = poly.fit_transform(train)
poly_train = pd.DataFrame(poly_train)
poly_train.head()
poly_train = poly_train.add_prefix('poly_')
train = pd.concat([train,poly_train], axis=1)
train.head()
MAX_ROUNDS = 2000
lgb_iter2 = []
sklearn_gbm_iter2 = []
xgb_gbm_iter2 = []

lgb_ap2 = []
sklearn_gbm_ap2 = []
xgb_gbm_ap2 = []
for i, (train_index, test_index) in enumerate(skf.split(train,y_bin)):
    
    # Create data for this fold
    y_train, y_valid = y_bin[train_index], y_bin[test_index]
    X_train, X_valid = train.iloc[train_index,:], train.iloc[test_index,:]
        
    print( "\nFold ", i)

    # Run model for this fold

    lgb_gbm = lgb.train(params, 
                          lgb.Dataset(X_train, label=y_train), 
                          MAX_ROUNDS, 
                          lgb.Dataset(X_valid, label=y_valid), 
                          verbose_eval=False, 
                          #feval= auc, 
                          early_stopping_rounds=50)
    
    print( " Best iteration lgb = ", lgb_gbm.best_iteration)
    
    sklearn_gbm = GradientBoostingClassifier(n_estimators=MAX_ROUNDS, 
                                    learning_rate = 0.06,
                                    max_features=2, 
                                    max_depth = 6, 
                                    n_iter_no_change=50, 
                                    tol=0.01,
                                    random_state = 0)
    
    sklearn_gbm.fit(X_train, y_train)
    print( " Best iteration sklearn_gbm = ", sklearn_gbm.n_estimators_)
    
    
    xgb_gbm = xgb.XGBClassifier(max_depth=6, 
                                n_estimators=MAX_ROUNDS,
                                eval_set=[(X_train, y_train), (X_valid, y_valid)],
                                learning_rate=0.06,
                                early_stopping_rounds=50)

    xgb_gbm.fit(X_train, y_train)
    
    print( " Best iteration xgboost_gbm = ", xgb_gbm.get_booster().best_iteration)
        
    lgb_iter2 = np.append(lgb_iter2, lgb_gbm.best_iteration)
    sklearn_gbm_iter2 = np.append(sklearn_gbm_iter2, sklearn_gbm.n_estimators_)
    xgb_gbm_iter2 = np.append(xgb_gbm_iter2, xgb_gbm.get_booster().best_iteration)
    
    pred = lgb_gbm.predict(X_valid, num_iteration=lgb_gbm.best_iteration)
    ap = average_precision_score(y_valid, pred, average='macro', pos_label=1, sample_weight=None)
    print('lgb ', ap)
    lgb_ap2 = np.append(lgb_ap2, ap)
    
    pred = sklearn_gbm.predict(X_valid)
    ap = average_precision_score(y_valid, pred, average='macro', pos_label=1, sample_weight=None)
    print('sklearn_gbn ', ap)
    sklearn_gbm_ap2 = np.append(sklearn_gbm_ap2, ap)
    
    pred  = xgb_gbm.predict(X_valid)
    ap = average_precision_score(y_valid, pred, average='macro', pos_label=1, sample_weight=None)
    print('xgboost ', ap)
    xgb_gbm_ap2 = np.append(xgb_gbm_ap2, ap)
print('lgb_iter1: ', np.mean(lgb_iter1),' lgb_iter2: ', np.mean(lgb_iter2))
print('sklearn_gbm_iter1: ', np.mean(sklearn_gbm_iter1), ' sklearn_gbm_iter2: ', np.mean(sklearn_gbm_iter2))
print('xgb_gbm_iter1: ',np.mean(xgb_gbm_iter1), ' xgb_gbm_iter2: ',np.mean(xgb_gbm_iter2) )

print('lgb_ap1: ', np.mean(lgb_ap1), ' lgb_ap2: ', np.mean(lgb_ap2))
print('sklearn_gbm_ap1: ', np.mean(sklearn_gbm_ap1), ' sklearn_gbm_ap2: ', np.mean(sklearn_gbm_ap2))
print('xgb_gbm_ap1: ', np.mean(xgb_gbm_ap1), ' xgb_gbm_ap2: ', np.mean(xgb_gbm_ap2))
# Installing the most recent version of skopt directly from Github
!pip install git+https://github.com/scikit-optimize/scikit-optimize.git
# Assuring you have the most recent CatBoost release
!pip install catboost -U
# Importing core libraries
import numpy as np
import pandas as pd
from time import time
import pprint
import joblib

# Suppressing warnings because of skopt verbosity
import warnings
warnings.filterwarnings("ignore")

# Our example dataset
from sklearn.datasets import load_boston

# Classifiers
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# Hyperparameters distributions
from scipy.stats import randint
from scipy.stats import uniform

# Model selection
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score

# Metrics
from sklearn.metrics import average_precision_score, roc_auc_score, mean_absolute_error
from sklearn.metrics import make_scorer

# Skopt functions
from skopt import BayesSearchCV
from skopt import gp_minimize # Bayesian optimization using Gaussian Processes
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args # decorator to convert a list of parameters to named arguments
from skopt.callbacks import DeadlineStopper # Stop the optimization before running out of a fixed budget of time.
from skopt.callbacks import VerboseCallback # Callback to control the verbosity
from skopt.callbacks import DeltaXStopper # Stop the optimization If the last two positions at which the objective has been evaluated are less than delta
# Uploading the Boston dataset
X, y = load_boston(return_X_y=True)
# Transforming the problem into a classification (unbalanced)
y_bin = (y > np.percentile(y, 90)).astype(int)
# Reporting util for different optimizers
def report_perf(optimizer, X, y, title, callbacks=None):
    """
    A wrapper for measuring time and performances of different optmizers
    
    optimizer = a sklearn or a skopt optimizer
    X = the training set 
    y = our target
    title = a string label for the experiment
    """
    start = time()
    if callbacks:
        optimizer.fit(X, y, callback=callbacks)
    else:
        optimizer.fit(X, y)
    best_score = optimizer.best_score_
    best_score_std = optimizer.cv_results_['std_test_score'][optimizer.best_index_]
    best_params = optimizer.best_params_
    print((title + " took %.2f seconds,  candidates checked: %d, best CV score: %.3f "
           +u"\u00B1"+" %.3f") % (time() - start, 
                                  len(optimizer.cv_results_['params']),
                                  best_score,
                                  best_score_std))    
    print('Best parameters:')
    pprint.pprint(best_params)
    print()
    return best_params
# Converting average precision score into a scorer suitable for model selection
avg_prec = make_scorer(average_precision_score, greater_is_better=True, needs_proba=True)
# Setting a 5-fold stratified cross-validation (note: shuffle=True)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
# A Scikit-learn GBM classifier
clf = GradientBoostingClassifier(n_estimators=20, random_state=0)
# GridSearchCV needs a predefined plan of the experiments
grid_search = GridSearchCV(clf, 
                           param_grid={"learning_rate": [0.01, 1.0],
                                       "n_estimators": [10, 500],
                                       "subsample": [1.0, 0.5],
                                       "min_samples_split": [2, 10],
                                       "min_samples_leaf": [1, 10],
                                       "max_features": ['sqrt', 'log2', None]
                                       },
                           n_jobs=-1,
                           cv=skf,
                           scoring=avg_prec,
                           iid=False, # just return the average score across folds
                           return_train_score=False)

best_params = report_perf(grid_search, X, y_bin,'GridSearchCV')
# RandomizedSearchCV needs the distribution of the experiments to be tested
# If you can provide the right distribution, the sampling will lead to faster and better results.

random_search = RandomizedSearchCV(clf, 
                           param_distributions={"learning_rate": uniform(0.01, 1.0),
                                                "n_estimators": randint(10, 500),
                                                "subsample": uniform(0.5, 0.5),
                                                "min_samples_split": randint(2, 10),
                                                "min_samples_leaf": randint(1, 10),
                                                "max_features": ['sqrt', 'log2', None]
                                       },
                                   n_iter=40,
                                   n_jobs=-1,
                                   cv=skf,
                                   scoring=avg_prec,
                                   iid=False, # just return the average score across folds
                                   return_train_score=False,
                                   random_state=0)

best_params = report_perf(random_search, X, y_bin, 'RandomizedSearchCV')
# also BayesSearchCV needs to work on the distributions of the experiments but it is less sensible to them

search_spaces = {"learning_rate": Real(0.01, 1.0),
                 "n_estimators": Integer(10, 500),
                 "subsample": Real(0.5, 1.0),
                 "min_samples_split": Integer(2, 10),
                 "min_samples_leaf": Integer(1, 10),
                 "max_features": Categorical(categories=['sqrt', 'log2', None])}

for baseEstimator in ['GP', 'RF', 'ET', 'GBRT']:
    opt = BayesSearchCV(clf,
                        search_spaces,
                        scoring=avg_prec,
                        cv=skf,
                        n_iter=40,
                        n_jobs=-1,
                        return_train_score=False,
                        optimizer_kwargs={'base_estimator': baseEstimator},
                        random_state=4)
    
    best_params = report_perf(opt, X, y_bin,'BayesSearchCV_'+baseEstimator)
# Initialize a pipeline with a model
pipe = Pipeline([('model', GradientBoostingClassifier(n_estimators=20, random_state=0))])

# Define search space for GBM;
search_space_GBM = {"model": Categorical([GradientBoostingClassifier(n_estimators=20, random_state=0)]),
                    "model__learning_rate": Real(0.01, 1.0),
                    "model__n_estimators": Integer(10, 500),
                    "model__subsample": Real(0.5, 1.0),
                    "model__min_samples_split": Integer(2, 10),
                    "model__min_samples_leaf": Integer(1, 10),
                    "model__max_features": Categorical(categories=['sqrt', 'log2', None])}

# Define search space for RF
search_space_RF  = {"model": Categorical([RandomForestClassifier(n_estimators=20, random_state=0)]),
                    "model__n_estimators": Integer(10, 200),
                    "model__min_samples_split": Integer(2, 10),
                    "model__min_samples_leaf": Integer(1, 10),
                    "model__max_features": Categorical(categories=['sqrt', 'log2', None])}

opt = BayesSearchCV(pipe,
                        search_spaces=[(search_space_GBM, 20), (search_space_RF, 20)],
                        scoring=avg_prec,
                        cv=skf,
                        n_jobs=-1,
                        return_train_score=False,
                        optimizer_kwargs={'base_estimator': 'GP'},
                        random_state=4)
    
best_params = report_perf(opt, X, y_bin,'BayesSearchCV_GP')
counter = 0
def onstep(res):
    global counter
    x0 = res.x_iters   # List of input points
    y0 = res.func_vals # Evaluation of input points
    print('Last eval: ', x0[-1], 
          ' - Score ', y0[-1])
    print('Current iter: ', counter, 
          ' - Score ', res.fun, 
          ' - Args: ', res.x)
    joblib.dump((x0, y0), 'checkpoint.pkl') # Saving a checkpoint to disk
    counter += 1

# Our search space
dimensions = [Real(0.01, 1.0, name="learning_rate"),
              Integer(10, 500, name="n_estimators"),
              Real(0.5, 1.0, name="subsample"),
              Integer(2, 10, name="min_samples_split"),
              Integer(1, 10, name="min_samples_leaf"),
              Categorical(categories=['sqrt', 'log2', None], name="max_features")]

# The objective function to be minimized
def make_objective(model, X, y, space, cv, scoring):
    # This decorator converts your objective function with named arguments into one that
    # accepts a list as argument, while doing the conversion automatically.
    @use_named_args(space) 
    def objective(**params):
        model.set_params(**params)
        return -np.mean(cross_val_score(model, 
                                        X, y, 
                                        cv=cv, 
                                        n_jobs=-1,
                                        scoring=scoring))

    return objective

objective = make_objective(clf,
                           X, y_bin,
                           space=dimensions,
                           cv=skf,
                           scoring=avg_prec)
gp_round = gp_minimize(func=objective,
                       dimensions=dimensions,
                       acq_func='gp_hedge', # Defining what to minimize 
                       n_calls=10,
                       callback=[onstep],
                       random_state=22)
x0, y0 = joblib.load('checkpoint.pkl')

gp_round = gp_minimize(func=objective,
                       x0=x0,              # already examined values for x
                       y0=y0,              # observed values for x0
                       dimensions=dimensions,
                       acq_func='gp_hedge', # Expected Improvement.
                       n_calls=10,
                       callback=[onstep],
                       random_state=0)
best_parameters = gp_round.x
best_result = gp_round.fun
print(best_parameters, best_result)
clf = lgb.LGBMClassifier(boosting_type='gbdt',
                         class_weight='balanced',
                         objective='binary',
                         n_jobs=1, 
                         verbose=0)

search_spaces = {
        'learning_rate': Real(0.01, 1.0, 'log-uniform'),
        'num_leaves': Integer(2, 500),
        'max_depth': Integer(0, 500),
        'min_child_samples': Integer(0, 200),
        'max_bin': Integer(100, 100000),
        'subsample': Real(0.01, 1.0, 'uniform'),
        'subsample_freq': Integer(0, 10),
        'colsample_bytree': Real(0.01, 1.0, 'uniform'),
        'min_child_weight': Integer(0, 10),
        'subsample_for_bin': Integer(100000, 500000),
        'reg_lambda': Real(1e-9, 1000, 'log-uniform'),
        'reg_alpha': Real(1e-9, 1.0, 'log-uniform'),
        'scale_pos_weight': Real(1e-6, 500, 'log-uniform'),
        'n_estimators': Integer(10, 10000)        
        }

opt = BayesSearchCV(clf,
                    search_spaces,
                    scoring=avg_prec,
                    cv=skf,
                    n_iter=40,
                    n_jobs=-1,
                    return_train_score=False,
                    refit=True,
                    optimizer_kwargs={'base_estimator': 'GP'},
                    random_state=22)
    
best_params = report_perf(opt, X, y_bin,'LightGBM', 
                          callbacks=[DeltaXStopper(0.001), 
                                     DeadlineStopper(60*5)])
counter = 0

clf = lgb.LGBMClassifier(boosting_type='gbdt',
                         class_weight='balanced',
                         objective='binary',
                         n_jobs=1, 
                         verbose=0)

dimensions = [Real(0.01, 1.0, 'log-uniform', name='learning_rate'),
              Integer(2, 500, name='num_leaves'),
              Integer(0, 500, name='max_depth'),
              Integer(0, 200, name='min_child_samples'),
              Integer(100, 100000, name='max_bin'),
              Real(0.01, 1.0, 'uniform', name='subsample'),
              Integer(0, 10, name='subsample_freq'),
              Real(0.01, 1.0, 'uniform', name='colsample_bytree'),
              Integer(0, 10, name='min_child_weight'),
              Integer(100000, 500000, name='subsample_for_bin'),
              Real(1e-9, 1000, 'log-uniform', name='reg_lambda'),
              Real(1e-9, 1.0, 'log-uniform', name='reg_alpha'),
              Real(1e-6, 500, 'log-uniform', name='scale_pos_weight'),
              Integer(10, 10000, name='n_estimators')]

objective = make_objective(clf,
                           X, y_bin,
                           space=dimensions,
                           cv=skf,
                           scoring=avg_prec)
gp_round = gp_minimize(func=objective,
                       dimensions=dimensions,
                       acq_func='gp_hedge',
                       n_calls=10, # Minimum is 10 calls
                       callback=[onstep],
                       random_state=7)
x0, y0 = joblib.load('checkpoint.pkl')

gp_round = gp_minimize(func=objective,
                       x0=x0,              # already examined values for x
                       y0=y0,              # observed values for x0
                       dimensions=dimensions,
                       acq_func='gp_hedge', # Expected Improvement.
                       n_calls=10,
                       #callback=[onstep],
                       random_state=3)

best_parameters = gp_round.x
best_result = gp_round.fun
print(best_parameters, best_result)
clf = xgb.XGBClassifier(
        n_jobs = 1,
        objective = 'binary:logistic',
        silent=1,
        tree_method='approx')
search_spaces = {'learning_rate': Real(0.01, 1.0, 'log-uniform'),
                 'min_child_weight': Integer(0, 10),
                 'max_depth': Integer(0, 50),
                 'max_delta_step': Integer(0, 20),
                 'subsample': Real(0.01, 1.0, 'uniform'),
                 'colsample_bytree': Real(0.01, 1.0, 'uniform'),
                 'colsample_bylevel': Real(0.01, 1.0, 'uniform'),
                 'reg_lambda': Real(1e-9, 1000, 'log-uniform'),
                 'reg_alpha': Real(1e-9, 1.0, 'log-uniform'),
                 'gamma': Real(1e-9, 0.5, 'log-uniform'),
                 'min_child_weight': Integer(0, 5),
                 'n_estimators': Integer(50, 100),
                 'scale_pos_weight': Real(1e-6, 500, 'log-uniform')}
opt = BayesSearchCV(clf,
                    search_spaces,
                    scoring=avg_prec,
                    cv=skf,
                    n_iter=40,
                    n_jobs=-1,
                    return_train_score=False,
                    refit=True,
                    optimizer_kwargs={'base_estimator': 'GP'},
                    random_state=22)
    
best_params = report_perf(opt, X, y_bin,'XGBoost',                           
                          callbacks=[DeltaXStopper(0.001), 
                                     DeadlineStopper(60*5)])
clf = CatBoostClassifier(loss_function='Logloss',
                         verbose = False)
search_spaces = {'iterations': Integer(10, 100),
                 'depth': Integer(1, 8),
                 'learning_rate': Real(0.01, 1.0, 'log-uniform'),
                 'random_strength': Real(1e-9, 10, 'log-uniform'),
                 'bagging_temperature': Real(0.0, 1.0),
                 'border_count': Integer(1, 255),
                 'ctr_border_count': Integer(1, 255),
                 'l2_leaf_reg': Integer(2, 30),
                 'scale_pos_weight':Real(0.01, 1.0, 'uniform')}
opt = BayesSearchCV(clf,
                    search_spaces,
                    scoring=avg_prec,
                    cv=skf,
                    n_iter=40,
                    n_jobs=1,  # use just 1 job with CatBoost in order to avoid segmentation fault
                    return_train_score=False,
                    refit=True,
                    optimizer_kwargs={'base_estimator': 'GP'},
                    random_state=22)

best_params = report_perf(opt, X, y_bin,'CatBoost', 
                          callbacks=[DeltaXStopper(0.001), 
                                     DeadlineStopper(60*5)])