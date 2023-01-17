import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import datetime, time
import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import scale
from scipy.stats import ranksums
from bayes_opt import BayesianOptimization
def reduce_mem_usage(df_):
    start_mem = df_.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe: {:.2f} MB'.format(start_mem))
    
    for c in df_.columns[df_.dtypes != 'object']:
        col_type = df_[c].dtype
        
        c_min = df_[c].min()
        c_max = df_[c].max()
        if str(col_type)[:3] == 'int':
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df_[c] = df_[c].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df_[c] = df_[c].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df_[c] = df_[c].astype(np.int32)
            else:
                df_[c] = df_[c].astype(np.int64)  
        else:
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                df_[c] = df_[c].astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df_[c] = df_[c].astype(np.float32)
            else:
                df_[c] = df_[c].astype(np.float64)

    end_mem = df_.memory_usage().sum() / 1024**2
    print('Memory usage after optimization: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df_
# For exploring missing values across train and test sets
def get_missing_report(df_, target_name = 'TARGET'):
    # Divide in training/validation and test data
    df_train_ = df_[df_[target_name].notnull()].drop(target_name, axis = 1)
    df_test_ = df_[df_[target_name].isnull()].drop(target_name, axis = 1)
    
    count_missing_train = df_train_.isnull().sum().values
    ratio_missing_train = count_missing_train / df_train_.shape[0]
    count_missing_test = df_test_.isnull().sum().values
    ratio_missing_test = count_missing_test / df_test_.shape[0]
    
    return pd.DataFrame(data = {'count_missing_train': count_missing_train, 
                                'ratio_missing_train': ratio_missing_train,
                                'count_missing_test': count_missing_test, 
                                'ratio_missing_test': ratio_missing_test}, 
                        index = df_test_.columns.values)
# For comparing distributions of two features
def corr_feature_with_target(feature, target):
    c0 = feature[target == 0].dropna()
    c1 = feature[target == 1].dropna()
        
    if set(feature.unique()) == set([0, 1]):
        diff = abs(c0.mean(axis = 0) - c1.mean(axis = 0))
    else:
        diff = abs(c0.median(axis = 0) - c1.median(axis = 0))
        
    p = ranksums(c0, c1)[1] if ((len(c0) >= 20) & (len(c1) >= 20)) else 2
        
    return [diff / feature.mean(), p]
# For selecting the best model for train set by simple launching the several classic models
def plot_roc_curves(df_train_, target_name, random_state = 0):
    warnings.simplefilter('ignore')
    
    f_imp = pd.DataFrame(index = df_train_.columns.drop(target_name))
    
    X_trn, X_tst, y_trn, y_tst = train_test_split(df_train_.drop(target_name, axis = 1), 
                                                  df_train_[target_name], 
                                                  test_size = 0.2, random_state = random_state)

    plt.figure(figsize = (7, 7))
    plt.plot([0, 1], [0, 1], 'k--')

    estimator = LGBMClassifier(random_state = random_state)
    estimator.fit(X_trn, y_trn)
    y_pred_xgb = estimator.predict_proba(X_tst)[:, 1]
    fpr_xgb, tpr_xgb, _ = roc_curve(y_tst, y_pred_xgb)
    f_imp['LGBM'] = pd.Series(estimator.feature_importances_, index = X_trn.columns)
    plt.plot(fpr_xgb, tpr_xgb, label = 'LGBM: ' + str(roc_auc_score(y_tst, y_pred_xgb)))
    
    X_trn.fillna(X_trn.mean(axis = 0), inplace = True)
    X_tst.fillna(X_tst.mean(axis = 0), inplace = True)
        
    estimator = RandomForestClassifier(random_state = random_state)
    estimator.fit(X_trn, y_trn)
    y_pred_rf = estimator.predict_proba(X_tst)[:, 1]
    fpr_rf, tpr_rf, _ = roc_curve(y_tst, y_pred_rf)
    f_imp['RF'] = estimator.feature_importances_
    plt.plot(fpr_rf, tpr_rf, label = 'RF: ' + str(roc_auc_score(y_tst, y_pred_rf)))
    
    estimator = LogisticRegression(random_state = random_state)
    estimator.fit(X_trn, y_trn)
    y_pred_lrg = estimator.predict_proba(X_tst)[:, 1]
    fpr_lrg, tpr_lrg, _ = roc_curve(y_tst, y_pred_lrg)
    plt.plot(fpr_lrg, tpr_lrg, label = 'LogR: ' + str(roc_auc_score(y_tst, y_pred_lrg)))
    
    X_trn = pd.DataFrame(scale(X_trn), index = X_trn.index, columns = X_trn.columns)
    X_tst = pd.DataFrame(scale(X_tst), index = X_tst.index, columns = X_tst.columns)
    
    estimator = KNeighborsClassifier()
    estimator.fit(X_trn, y_trn)
    y_pred_knn = estimator.predict_proba(X_tst)[:, 1]
    fpr_knn, tpr_knn, _ = roc_curve(y_tst, y_pred_knn)
    plt.plot(fpr_knn, tpr_knn, label = 'KNN: ' + str(roc_auc_score(y_tst, y_pred_knn)))
    
    del X_trn, X_tst, y_trn, y_tst
    gc.collect()
    
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc = 'best')
    plt.show()
    
    f_imp['mean'] = f_imp.mean(axis = 1)
    return f_imp
# For saving scores and metrics
scores_index = [
        'roc_auc_train', 'roc_auc_test', 
        'precision_train_0', 'precision_test_0', 
        'precision_train_1', 'precision_test_1', 
        'recall_train_0', 'recall_test_0', 
        'recall_train_1', 'recall_test_1', 
        'LB'
]
# For visual analysis of the metrics
def display_scores(df_scores_):
    _, axes = plt.subplots(3, 2, figsize = (25, 10))
    df_scores_.T[[scores_index[0]]].plot(ax = axes[0, 0]); # roc-auc train
    df_scores_.T[[scores_index[1], scores_index[10]]].plot(ax = axes[0, 1]); # roc-auc test & LB
    df_scores_.T[[scores_index[2], scores_index[3]]].plot(ax = axes[1, 0]);  # precision class 0
    df_scores_.T[[scores_index[4], scores_index[5]]].plot(ax = axes[1, 1]);  # precision class 1
    df_scores_.T[[scores_index[6], scores_index[7]]].plot(ax = axes[2, 0]);  # recall class 0
    df_scores_.T[[scores_index[8], scores_index[9]]].plot(ax = axes[2, 1]);  # recall class 1
# For cleaning float LGBM parameters after Bayesian optimization
def int_lgbm_params(params):
    for p in params:
        if p in ['num_leaves', 'max_depth', 'n_estimators', 'subsample_for_bin', 'min_child_samples', 
                 'subsample_freq', 'random_state']:
            params[p] = int(np.round(params[p], decimals = 0))
    return params
# For cross-validation with LGBM classifier
def cv_lgbm_scores(df_, num_folds, params, 
                   target_name = 'TARGET', index_name = 'SK_ID_CURR',
                   stratified = False, rs = 1001, verbose = -1):
    
    warnings.simplefilter('ignore')
    
    # Cleaning and defining parameters for LGBM
    params = int_lgbm_params(params)
    clf = LGBMClassifier(**params, n_estimators = 20000, nthread = 4, n_jobs = -1)

    # Divide in training/validation and test data
    df_train_ = df_[df_[target_name].notnull()]
    df_test_ = df_[df_[target_name].isnull()]
    print("Starting LightGBM cross-validation at {}".format(time.ctime()))
    print("Train shape: {}, test shape: {}".format(df_train_.shape, df_test_.shape))

    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits = num_folds, shuffle = True, random_state = rs)
    else:
        folds = KFold(n_splits = num_folds, shuffle = True, random_state = rs)
        
    # Create arrays to store results
    train_pred = np.zeros(df_train_.shape[0])
    train_pred_proba = np.zeros(df_train_.shape[0])

    test_pred = np.zeros(df_train_.shape[0])
    test_pred_proba = np.zeros(df_train_.shape[0])
    
    prediction = np.zeros(df_test_.shape[0]) # prediction for test set
    
    feats = df_train_.columns.drop([target_name, index_name])
    
    df_feat_imp_ = pd.DataFrame(index = feats)
    
    # Cross-validation cycle
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(df_train_[feats], df_train_[target_name])):
        print('--- Fold {} started at {}'.format(n_fold, time.ctime()))
        
        train_x, train_y = df_train_[feats].iloc[train_idx], df_train_[target_name].iloc[train_idx]
        valid_x, valid_y = df_train_[feats].iloc[valid_idx], df_train_[target_name].iloc[valid_idx]

        clf.fit(train_x, train_y, 
                eval_set = [(valid_x, valid_y)], eval_metric = 'auc', 
                verbose = verbose, early_stopping_rounds = 100)

        train_pred[train_idx] = clf.predict(train_x, num_iteration = clf.best_iteration_)
        train_pred_proba[train_idx] = clf.predict_proba(train_x, num_iteration = clf.best_iteration_)[:, 1]
        test_pred[valid_idx] = clf.predict(valid_x, num_iteration = clf.best_iteration_)
        test_pred_proba[valid_idx] = clf.predict_proba(valid_x, num_iteration = clf.best_iteration_)[:, 1]
        
        prediction += clf.predict_proba(df_test_[feats], 
                                        num_iteration = clf.best_iteration_)[:, 1] / folds.n_splits

        df_feat_imp_[n_fold] = pd.Series(clf.feature_importances_, index = feats)
        
        del train_x, train_y, valid_x, valid_y
        gc.collect()

    # Computation of metrics
    roc_auc_train = roc_auc_score(df_train_[target_name], train_pred_proba)
    precision_train = precision_score(df_train_[target_name], train_pred, average = None)
    recall_train = recall_score(df_train_[target_name], train_pred, average = None)
    
    roc_auc_test = roc_auc_score(df_train_[target_name], test_pred_proba)
    precision_test = precision_score(df_train_[target_name], test_pred, average = None)
    recall_test = recall_score(df_train_[target_name], test_pred, average = None)

    print('Full AUC score {:.6f}'.format(roc_auc_test))
    
    # Filling the feature_importance table
    df_feat_imp_.fillna(0, inplace = True)
    df_feat_imp_['mean'] = df_feat_imp_.mean(axis = 1)
    
    # Preparing results of prediction for saving
    prediction_train = df_train_[[index_name]]
    prediction_train[target_name] = test_pred_proba
    prediction_test = df_test_[[index_name]]
    prediction_test[target_name] = prediction
    
    del df_train_, df_test_
    gc.collect()
    
    # Returning the results and metrics in format for scores' table
    return df_feat_imp_, prediction_train, prediction_test, \
           [roc_auc_train, roc_auc_test,
            precision_train[0], precision_test[0], precision_train[1], precision_test[1],
            recall_train[0], recall_test[0], recall_train[1], recall_test[1], 0]
# For visual analysis of the fearure importances
def display_feature_importances(df_feat_imp_):
    n_columns = 3
    n_rows = (df_feat_imp_.shape[1] + 1) // n_columns
    _, axes = plt.subplots(n_rows, n_columns, figsize=(8 * n_columns, 8 * n_rows))
    for i, c in enumerate(df_feat_imp_.columns):
        sns.barplot(x = c, y = 'index', 
                    data = df_feat_imp_.reset_index().sort_values(c, ascending = False).head(20), 
                    ax = axes[i // n_columns, i % n_columns] if n_rows > 1 else axes[i % n_columns])
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.show()
# For selection of parameters for LGBM with Bayesian optimization
def get_best_params_for_lgbm(df_train_, seed_cv_, seed_bo_, target_name = 'TARGET', 
                             init_points = 5, n_iter = 5):
    def lgbm_auc_evaluate(**params):
        warnings.simplefilter('ignore')
    
        params = int_lgbm_params(params)   
        clf = LGBMClassifier(**params, n_estimators = 10000, nthread = 4, n_jobs = -1)

        folds = KFold(n_splits = 2, shuffle = True, random_state = params['random_state'])
        
        test_pred_proba = np.zeros(df_train_.shape[0])
    
        feats = df_train_.columns.drop(target_name)
    
        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(df_train_[feats], df_train_[target_name])):
            train_x, train_y = df_train_[feats].iloc[train_idx], df_train_[target_name].iloc[train_idx]
            valid_x, valid_y = df_train_[feats].iloc[valid_idx], df_train_[target_name].iloc[valid_idx]

            clf.fit(train_x, train_y, 
                    eval_set = [(valid_x, valid_y)], eval_metric = 'auc', 
                    verbose = False, early_stopping_rounds = 100)

            test_pred_proba[valid_idx] = clf.predict_proba(valid_x, num_iteration = clf.best_iteration_)[:, 1]
        
            del train_x, train_y, valid_x, valid_y
            gc.collect()
        
        roc_auc_test = roc_auc_score(df_train_[target_name], test_pred_proba)
        
        return roc_auc_test
    
    params = {'learning_rate': (.001, .02), 
          'colsample_bytree': (0.3, 1),
          'subsample_for_bin' : (20000, 500000),
          'subsample': (0.3, 1), 
          'num_leaves': (2, 100), 
          'max_depth': (3, 9), 
          'reg_alpha': (.0, 1.), 
          'reg_lambda': (.0, 1.), 
          'min_split_gain': (.01, 1.),
          'min_child_weight': (1, 50),
          'min_child_samples': (10, 1000),
          'random_state': (seed_cv_, seed_cv_)}
    bo = BayesianOptimization(lgbm_auc_evaluate, params, random_state = seed_bo_)
    bo.maximize(init_points = init_points, n_iter = n_iter)

    return bo.res['max']['max_val'], bo.res['max']['max_params']
# For metadata about predictions
blending_index = ['date', 'to_blend', 'folder', 'file_name', 'auc_train', 'auc_test', 'auc_LB', 'Comments']
suffix_train = '_train.csv'
suffix_test = '_test.csv'
blending_folder = '../input/tmp-preds'
blending_file_name = 'predictions_for_blending.csv'

df_blend = pd.DataFrame(index = blending_index)
df_blend.index.name = 'index'
#df_blend.to_csv(blending_folder + '/' + blending_file_name) # Commented for Kaggle
# For saving prediction into file
def save_prediction(df_, file_name, index_col = 'SK_ID_CURR', prediction_col = 'TARGET'):
    df_.columns = [index_col, prediction_col]
    df_.to_csv(file_name, index = False) 
# For saving files and metadata about predictions for blending
def store_predictions_for_blending(df_train_, df_test_, file_name, scor, comments, 
                                   index_col = 'SK_ID_CURR', prediction_col = 'TARGET', 
                                   folder = blending_folder, b_file_name = blending_file_name):
    
    full_file_name = folder + '/' + file_name
    save_prediction(df_train_, full_file_name + suffix_train, index_col, prediction_col)
    save_prediction(df_test_, full_file_name + suffix_test, index_col, prediction_col)
    
    df_blending = pd.read_csv(b_file_name, index_col = 'index')
    df_blending[df_blending.shape[1]] = [
        datetime.datetime.today(),
        True,
        folder, file_name,
        scor[0], scor[1], scor[-1],
        comments
    ]
    df_blending.to_csv(b_file_name)
    del df_blending
    gc.collect()
# For loading previous prediction results
def load_predictions_for_blending(b_folder = blending_folder, b_file_name = blending_file_name):
    def load_prediction(file_name):
        tmp = pd.read_csv(file_name)
        return tmp.set_index(tmp.columns[0])

    df_blend_ = pd.read_csv(b_folder + '/' + b_file_name, index_col = 'index')
    df_train_ = []
    df_test_ = []
    for c in df_blend_.columns:
        #full_file_name = df_blend_.loc['folder', c] + '/' + df_blend_.loc['file_name', c]
        full_file_name = '../input/tmp-preds' + '/' + df_blend_.loc['file_name', c] # Only for Kaggle
        df_train_.append(load_prediction(full_file_name + suffix_train))
        df_test_.append(load_prediction(full_file_name + suffix_test))
        
    df_train_ = pd.concat(df_train_, axis = 1)
    df_train_.columns = df_blend_.columns
    df_test_ = pd.concat(df_test_, axis = 1)
    df_test_.columns = df_blend_.columns

    return df_train_, df_test_, df_blend_
# For blending flagged predictions
def get_blended_prediction(df_train_, flag, params_):
    warnings.simplefilter('ignore')
    
    test_pred_proba = pd.Series(np.zeros(df_train_.shape[0]), index = df_train_.index)
    
    for f in df_train_.columns[flag.values.astype(bool)]:
        test_pred_proba += df_train_[f] * params_[f]
        
    min_pr = test_pred_proba.min()
    max_pr = test_pred_proba.max()
    test_pred_proba = (test_pred_proba - min_pr) / (max_pr - min_pr)
    return test_pred_proba
# For selection of parameters with Bayesian optimization
def get_best_params_for_blending(df_train_, flag, target, seed_bo_, init_points = 10, n_iter = 10):
    def blend_auc_evaluate(**params):
        return roc_auc_score(target, get_blended_prediction(df_train_, flag, params))    
    
    params = {}
    for c in df_train_.columns[flag.values.astype(bool)]:
        params[c] = (0, 1)

    bo = BayesianOptimization(blend_auc_evaluate, params, random_state = seed_bo_)
    bo.maximize(init_points = init_points, n_iter = n_iter)

    return bo.res['max']['max_val'], bo.res['max']['max_params']
from sklearn.preprocessing import LabelEncoder
df_train = pd.read_csv('../input/home-credit-default-risk/application_train.csv')
df_test = pd.read_csv('../input/home-credit-default-risk/application_test.csv')
print('Train shape: {}, test shape: {}'.format(df_train.shape, df_test.shape))
df_all = pd.concat([df_train, df_test], axis = 0, ignore_index = True)
print('Сombined shape: {}'.format(df_all.shape))
train_index = df_all[df_all['TARGET'].notnull()].index
test_index = df_all[df_all['TARGET'].isnull()].index
print('Train shape: {}, test shape: {}'.format(df_all.loc[train_index].shape, df_all.loc[test_index].shape))
del df_train, df_test
gc.collect()
le = LabelEncoder()
categorical = df_all.columns[df_all.dtypes == 'object']
print('{} categorical features were'.format(len(categorical)))
for c in categorical:
    df_all[c].fillna('NaN', inplace = True)
    df_all[c] = le.fit_transform(df_all[c])
print('{} categorical features left'.format(len(df_all.columns[df_all.dtypes == 'object'])))
df_all = reduce_mem_usage(df_all)
missing_values = get_missing_report(df_all)
missing_values.sort_values('ratio_missing_train', ascending = False) \
                                    [['ratio_missing_train', 'ratio_missing_test']].plot(figsize = (25, 7));
missing_values[abs(missing_values['ratio_missing_train'] - missing_values['ratio_missing_test']) > .1]
corr_target = pd.DataFrame(index = ['diff', 'p'])
for c in df_all.columns.drop('TARGET'):
    corr_target[c] = corr_feature_with_target(df_all.loc[train_index, c], df_all.loc[train_index, 'TARGET'])
corr_target = corr_target.T
bad_features = corr_target[(corr_target['diff'] < 1e-5) & (corr_target['p'] > .05)].index
print('There are {} uninformative features'.format(len(bad_features)))
bad_features
df_all.drop(bad_features, axis = 1, inplace = True)
df_all.shape
target_test = (df_all['TARGET'].notnull()).astype(int)
corr_test = pd.DataFrame(index = ['diff', 'p'])
for c in df_all.columns.drop('TARGET'):
    corr_test[c] = corr_feature_with_target(df_all[c], target_test)
corr_test = corr_test.T
bad_features = corr_test[(corr_test['diff'] > 1) & (corr_test['p'] < .05)].index
print('There are {} features with different distribution on the train and test sets'.format(len(bad_features)))
bad_features
df_all.drop(bad_features, axis = 1, inplace = True)
df_all.shape
feature_importance = plot_roc_curves(df_all.loc[train_index], 'TARGET', random_state = 0)
feature_importance.head()
display_feature_importances(feature_importance)
step = 'first_prediction'
seed_cv = 0
score_table = pd.DataFrame(index = scores_index)
f_importance, pred_train, pred_test, score = cv_lgbm_scores(df_all, 
                                                            num_folds = 5, params = {'random_state': seed_cv}, 
                                                            rs = seed_cv)
display_feature_importances(f_importance)
#save_prediction(pred_test, 'pred_' + step + suffix_test) # Commented for Kaggle
score[-1] = .745
score_table[step] = score
score_table.T
#store_predictions_for_blending(pred_train, pred_test, step, score, # Commented for Kaggle
#                               comments = 'The first prediction without tuning')
step = 'bo_seed_0'
seed_bo = 0
best_score, best_params = get_best_params_for_lgbm(df_all.loc[train_index], seed_cv, seed_bo)
print('Best score:', best_score)
print('Best params:', best_params)
f_importance, pred_train, pred_test, score = cv_lgbm_scores(df_all, 
                                                            num_folds = 5, params = best_params, 
                                                            rs = seed_cv)
#save_prediction(pred_test, 'pred_' + step + suffix_test) # Commented for Kaggle
score[-1] = .748
score_table[step] = score
score_table.T
display_scores(score_table)
#store_predictions_for_blending(pred_train, pred_test, step, score, # Commented for Kaggle
#                               comments = 'Best params for LGBM (seed_bo = 0)')
df_train, df_test, df_blending = load_predictions_for_blending()
df_blending.T
df_train.head()
df_test.head()
best_score, best_params = get_best_params_for_blending(df_train, df_blending.loc['to_blend'], 
                                                       df_all.loc[train_index, 'TARGET'], seed_bo)
print('Best score:', best_score)
print('Best params:', best_params)
pred = get_blended_prediction(df_test, df_blending.loc['to_blend'], best_params)
pred.head()
#save_prediction(pred.reset_index(), 'final.csv') # Commented for Kaggle
