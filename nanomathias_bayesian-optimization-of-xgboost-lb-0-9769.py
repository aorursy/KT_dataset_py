import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold

# SETTINGS - CHANGE THESE TO GET SOMETHING MEANINGFUL
ITERATIONS = 10 # 1000
TRAINING_SIZE = 100000 # 20000000
TEST_SIZE = 25000

# Load data
X = pd.read_csv(
    '../input/train.csv', 
    skiprows=range(1,184903891-TRAINING_SIZE), 
    nrows=TRAINING_SIZE,
    parse_dates=['click_time']
)

# Split into X and y
y = X['is_attributed']
X = X.drop(['click_time','is_attributed', 'attributed_time'], axis=1)
# Classifier
bayes_cv_tuner = BayesSearchCV(
    estimator = xgb.XGBClassifier(
        n_jobs = 1,
        objective = 'binary:logistic',
        eval_metric = 'auc',
        silent=1,
        tree_method='approx'
    ),
    search_spaces = {
        'learning_rate': (0.01, 1.0, 'log-uniform'),
        'min_child_weight': (0, 10),
        'max_depth': (0, 50),
        'max_delta_step': (0, 20),
        'subsample': (0.01, 1.0, 'uniform'),
        'colsample_bytree': (0.01, 1.0, 'uniform'),
        'colsample_bylevel': (0.01, 1.0, 'uniform'),
        'reg_lambda': (1e-9, 1000, 'log-uniform'),
        'reg_alpha': (1e-9, 1.0, 'log-uniform'),
        'gamma': (1e-9, 0.5, 'log-uniform'),
        'min_child_weight': (0, 5),
        'n_estimators': (50, 100),
        'scale_pos_weight': (1e-6, 500, 'log-uniform')
    },    
    scoring = 'roc_auc',
    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    ),
    n_jobs = 3,
    n_iter = ITERATIONS,   
    verbose = 0,
    refit = True,
    random_state = 42
)

def status_print(optim_result):
    """Status callback durring bayesian hyperparameter search"""
    
    # Get all the models tested so far in DataFrame format
    all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)    
    
    # Get current parameters and the best parameters    
    best_params = pd.Series(bayes_cv_tuner.best_params_)
    print('Model #{}\nBest ROC-AUC: {}\nBest params: {}\n'.format(
        len(all_models),
        np.round(bayes_cv_tuner.best_score_, 4),
        bayes_cv_tuner.best_params_
    ))
    
    # Save all model results
    clf_name = bayes_cv_tuner.estimator.__class__.__name__
    all_models.to_csv(clf_name+"_cv_results.csv")
# Fit the model
result = bayes_cv_tuner.fit(X.values, y.values, callback=status_print)
# Classifier
bayes_cv_tuner = BayesSearchCV(
    estimator = lgb.LGBMRegressor(
        objective='binary',
        metric='auc',
        n_jobs=1,
        verbose=0
    ),
    search_spaces = {
        'learning_rate': (0.01, 1.0, 'log-uniform'),
        'num_leaves': (1, 100),      
        'max_depth': (0, 50),
        'min_child_samples': (0, 50),
        'max_bin': (100, 1000),
        'subsample': (0.01, 1.0, 'uniform'),
        'subsample_freq': (0, 10),
        'colsample_bytree': (0.01, 1.0, 'uniform'),
        'min_child_weight': (0, 10),
        'subsample_for_bin': (100000, 500000),
        'reg_lambda': (1e-9, 1000, 'log-uniform'),
        'reg_alpha': (1e-9, 1.0, 'log-uniform'),
        'scale_pos_weight': (1e-6, 500, 'log-uniform'),
        'n_estimators': (50, 100),
    },    
    scoring = 'roc_auc',
    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    ),
    n_jobs = 3,
    n_iter = ITERATIONS,   
    verbose = 0,
    refit = True,
    random_state = 42
)

# Fit the model
result = bayes_cv_tuner.fit(X.values, y.values, callback=status_print)
from sklearn.model_selection import PredefinedSplit

# Training [index == -1], testing [index == 0])
test_fold = np.zeros(len(X))
test_fold[:(TRAINING_SIZE-TEST_SIZE)] = -1
cv = PredefinedSplit(test_fold)

# Check that we only have a single train-test split, and the size
train_idx, test_idx = next(cv.split())
print(f"Splits: {cv.get_n_splits()}, Train size: {len(train_idx)}, Test size: {len(test_idx)}")
from sklearn.model_selection import TimeSeriesSplit

# Here we just do 3-fold timeseries CV
cv = TimeSeriesSplit(max_train_size=None, n_splits=3)

# Let us check the sizes of the folds. Note that you can keep train size constant with max_train_size if needed
for i, (train_index, test_index) in enumerate(cv.split(X)):
    print(f"Split {i+1} / {cv.get_n_splits()}:, Train size: {len(train_index)}, Test size: {len(test_index)}")
{
    'colsample_bylevel': 0.1,
    'colsample_bytree': 1.0,
    'gamma': 5.103973694670875e-08,
    'learning_rate': 0.140626707498132,
    'max_delta_step': 20,
    'max_depth': 6,
    'min_child_weight': 4,
    'n_estimators': 100,
    'reg_alpha': 1e-09,
    'reg_lambda': 1000.0,
    'scale_pos_weight': 499.99999999999994,
    'subsample': 1.0
}