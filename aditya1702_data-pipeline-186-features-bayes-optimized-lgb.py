import numpy as np
import pandas as pd
import scipy
import gc
import pickle

from scipy.stats import ks_2samp, mode
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, ShuffleSplit
from sklearn.model_selection._split import check_cv
from sklearn.base import clone, is_classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, make_scorer
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion

from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras.models import Model, Sequential

import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from bayes_opt import BayesianOptimization

import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline
def read_data():
    print("############# Read the data #############")
    
    train = pd.read_csv("../input/train.csv", index_col = 0)
    test = pd.read_csv("../input/test.csv", index_col = 0)
    sub = pd.read_csv("../input/sample_submission.csv", index_col = 0)
    print("\nTrain shape: {}\nTest shape: {}".format(train.shape, test.shape))
    
    return train, test, sub
train_orig, test_orig, sub_orig = read_data()
train_orig.head()
y_train = train_orig['Target']
x_train = train_orig.drop('Target', 1)
x_test = test_orig.copy()
test_id = test_orig.index
class MissingValuesImputer(BaseEstimator, TransformerMixin):
    
    def __init__(self, impute_zero_columns):
        self.impute_zero_columns = impute_zero_columns
        
    def fit(self, X, y = None):
        print("Mean Values Imputer")
        return self
    
    def transform(self, X, y = None):
        
        # Fill missing values for v18q1, v2a1 and rez_esc
        for column in self.impute_zero_columns:
            X[column] = X[column].fillna(0)

        # For meaneduc we use the average schooling of household adults
        self.X_with_meaneduc_na = X[pd.isnull(X['meaneduc'])]
        self.mean_escolari_dict = dict(self.X_with_meaneduc_na.groupby('idhogar')['escolari'].apply(np.mean))
        for row_index in self.X_with_meaneduc_na.index:
            row_idhogar = X.at[row_index, 'idhogar']
            X.at[row_index, 'meaneduc'] = self.mean_escolari_dict[row_idhogar]
            X.at[row_index, 'SQBmeaned'] = np.square(self.mean_escolari_dict[row_idhogar])
        return X
class CategoricalVariableTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.categorical_columns = ['idhogar']
        
    def fit(self, X, y = None):
        print("Categorical Variables Transformer")
        return self
    
    def transform(self, X, y = None):
        X['dependency'] = np.sqrt(X['SQBdependency'])

        X.loc[X['edjefe'] == 'no', 'edjefe'] = '0'
        X.loc[X['edjefe'] == 'yes', 'edjefe'] = '1'
        
        X.loc[X['edjefa'] == 'no', 'edjefa'] = '0'
        X.loc[X['edjefa'] == 'yes', 'edjefa'] = '1'

        X['edjefa'] = X['edjefa'].astype(int)
        X['edjefe'] = X['edjefe'].astype(int)
    
        label_encoder = LabelEncoder()
        for column in self.categorical_columns:
            X[column] = label_encoder.fit_transform(X[column])

        return X
class UnnecessaryColumnsRemoverTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, axis = 1):
        print("Unnecessary Columns Remover Transformer")
        self.axis = axis
        self.unnecessary_columns = [
                                     'r4t3', 'tamhog', 'tamviv', 'hogar_total', 'v18q', 'v14a', 'agesq',
                                     'mobilephone', 'energcocinar1', 'sanitario6',
                                     'estadocivil7', 'lugar1', 'area1', 'female'
                                   ]
        
    def fit(self, X, y = None):
        unnecessary_columns_to_extend = [
            [col for col in X.columns.tolist() if 'SQB' in col],
            [col for col in X.columns.tolist() if 'epared' in col],
            [col for col in X.columns.tolist() if 'etecho' in col],
            [col for col in X.columns.tolist() if 'eviv' in col],
            [col for col in X.columns.tolist() if 'instlevel' in col],
            [col for col in X.columns.tolist() if 'pared' in col],
            [col for col in X.columns.tolist() if 'piso' in col],
            [col for col in X.columns.tolist() if 'techo' in col],
            [col for col in X.columns.tolist() if 'abastagua' in col],
            [col for col in X.columns.tolist() if 'elimbasu' in col],
            [col for col in X.columns.tolist() if 'tipoviv' in col]
        ]
        
        for col_list in unnecessary_columns_to_extend:
            self.unnecessary_columns.extend(col_list)
        return self
    
    def transform(self, X, y = None):
        X = X.drop(self.unnecessary_columns, axis = self.axis)
        return X
class ReverseOHETransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, axis = 1):
        self.axis = axis
        self.ohe_column_prefixes = ['epared', 'etecho', 'eviv', 'instlevel', 
                                    'pared', 'piso', 'techo', 'abastagua']
                                    
        
    def fit(self, X, y = None):
        print("Reverse OHE Transformer")
        
        self.ohe_columns_dict = dict()
        for column_prefix in self.ohe_column_prefixes:
            ohe_columns = [col for col in X.columns if col.startswith(column_prefix)]
            self.ohe_columns_dict[column_prefix] = ohe_columns
        return self
    
    def transform(self, X, y = None):
        for ohe_column_prefix, ohe_column_list in self.ohe_columns_dict.items():
            ohe_df = X[ohe_column_list]
            ohe_df.columns = list(range(ohe_df.shape[1]))
            ohe_numeric_df = ohe_df.idxmax(axis = self.axis)
            ohe_numeric_df.name = ohe_column_prefix
            X = pd.concat([X, ohe_numeric_df], axis = self.axis)
            
            # Remove the columns from the data
            X = X.drop(ohe_column_list, axis = self.axis)
        return X
class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, axis = 1):
        self.axis = axis
        
        # individual level boolean features
        self.individual_boolean_features = ['dis', 'male', 'estadocivil1', 'estadocivil2', 
                                            'estadocivil3', 'estadocivil4', 'estadocivil5', 'estadocivil6', 
                                            'parentesco1', 'parentesco2',  'parentesco3', 'parentesco4', 
                                            'parentesco5', 'parentesco6', 'parentesco7', 'parentesco8',  
                                            'parentesco9', 'parentesco10', 'parentesco11']

        # individual level ordered features
        self.individual_ordered_features = ['escolari', 'age']
        
    def fit(self, X, y = None):
        print("Feature Engineering Transformer")
        self.more_columns_to_drop = [
            [col for col in X.columns.tolist() if 'parentesco' in col and 'parentesco1' not in col],
            ['idhogar']
        ]
        
        f = lambda x: x.std(ddof = 0)
        f.__name__ = 'std_0'
        self.aggregate_features = (['mean', 'max', 'min', 'sum', f])
        return self
    
    def transform(self, X, y = None):
        
        # Rooms
        X['rent_per_room'] = X['v2a1']/X['rooms']
        X['adults_per_room'] = X['hogar_adul']/X['rooms']
        X['males_per_room'] = X['r4h3']/X['rooms']
        X['females_per_room'] = X['r4m3']/X['rooms']
        X['children_per_room'] = X['hogar_nin']/X['rooms']
        X['humans_per_room'] = X['hhsize']/X['rooms']
        X['beds_per_room'] = X['bedrooms']/X['rooms']
        
        # Bedroom
        X['adults_per_bedroom'] = X['hogar_adul']/X['bedrooms']
        X['males_per_bedroom'] = X['r4h3']/X['bedrooms']
        X['females_per_bedroom'] = X['r4m3']/X['bedrooms']
        X['children_per_bedroom'] = X['hogar_nin']/X['bedrooms']
        X['humans_per_bedroom'] = X['hhsize']/X['bedrooms']
        
        X['persons12less_fraction'] = (X['r4h1'] + X['r4m1'])/X['hhsize']
        X['males12plus_fraction'] = X['r4h2']/X['hhsize']
        X['total_males_fraction'] = X['r4h3']/X['hhsize']
        X['females12plus_fraction'] = X['r4m2']/X['hhsize']
        X['all_females_fraction'] = X['r4m3']/X['hhsize']
        X['rent_per_person'] = X['v2a1']/X['hhsize']
        X['mobiles_per_person'] = X['qmobilephone']/X['hhsize']
        X['tablets_per_person'] = X['v18q1']/X['hhsize']
        X['mobiles_per_male'] = X['qmobilephone']/X['r4h3']
        X['tablets_per_male'] = X['v18q1']/X['r4h3']
        
#         X['males_per_females'] = X['r4h3']/X['r4m3']
#         X['males12plus_per_females12plus'] = X['r4h2']/X['r4m2']
#         X['males12less_per_females12less'] = X['r4h1']/X['r4m1']
#         X['number_of_non_bedrooms'] = np.abs(X['rooms'] - X['bedrooms'])
        
        # Create individual-level features
        grouped_df = X.groupby('idhogar')[self.individual_boolean_features + self.individual_ordered_features]
        grouped_df = grouped_df.agg(self.aggregate_features)
        X = X.join(grouped_df, on = 'idhogar')
        
        # Finally remove the other parentesco columns since we are only going to use only heads of
        # households for our scoring
        for col in self.more_columns_to_drop:
            X = X.drop(col, axis = self.axis) 
        
        return X
class LGBClassifierCV(BaseEstimator, RegressorMixin):
    
    def __init__(self, axis = 0, lgb_params = None, fit_params = None, cv = 3, perform_bayes_search = False, perform_random_search = False, use_train_test_split = False, use_kfold_split = True):
        self.axis = axis
        self.lgb_params = lgb_params
        self.fit_params = fit_params
        self.cv = cv
        self.perform_random_search = perform_random_search
        self.perform_bayes_search = perform_bayes_search
        self.use_train_test_split = use_train_test_split
        self.use_kfold_split = use_kfold_split
    
    @property
    def feature_importances_(self):
        feature_importances = []
        for estimator in self.estimators_:
            feature_importances.append(
                estimator.feature_importances_
            )
        return np.mean(feature_importances, axis = 0)
    
    @property
    def evals_result_(self):
        evals_result = []
        for estimator in self.estimators_:
            evals_result.append(
                estimator.evals_result_
            )
        return np.array(evals_result)
    
    @property
    def best_scores_(self):
        best_scores = []
        for estimator in self.estimators_:
            best_scores.append(
                estimator.best_score_['validation']['macroF1']
            )
        return np.array(best_scores)
    
    @property
    def cv_scores_(self):
        return self.best_scores_ 
    
    @property
    def cv_score_(self):
        return np.mean(self.best_scores_)
    
    @property
    def best_iterations_(self):
        best_iterations = []
        for estimator in self.estimators_:
            best_iterations.append(
                estimator.best_iteration_
            )
        return np.array(best_iterations)
    
    @property
    def best_iteration_(self):
        return np.round(np.mean(self.best_iterations_))

    def find_best_params_(self, X, y):
        
        if self.perform_random_search:
            # Define a search space for the parameters
            lgb_search_params = {
                      'num_leaves': sp_randint(20, 100), 
                      'min_child_samples': sp_randint(40, 100), 
                      'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
                      'subsample': sp_uniform(loc = 0.75, scale = 0.25), 
                      'colsample_bytree': sp_uniform(loc = 0.8, scale = 0.15),
                      'reg_alpha': [0, 1e-3, 1e-1, 1, 10, 50, 100],
                      'reg_lambda': [0, 1e-3, 1e-1, 1, 10, 50, 100]
                }

            x_train, x_val, y_train, y_val = train_test_split(X, y, test_size = 0.10, random_state = 42, stratify = y)
            F1_scorer = make_scorer(f1_score, greater_is_better = True, average = 'macro')

            lgb_model = lgb.LGBMClassifier(**self.lgb_params)
            self.fit_params["eval_set"] = [(x_train, y_train), (x_val, y_val)]
            self.fit_params["verbose"] = 200

            rs = RandomizedSearchCV(estimator = lgb_model, 
                                    param_distributions = lgb_search_params, 
                                    n_iter = 100,
                                    scoring = F1_scorer,
                                    cv = 5,
                                    refit = True,
                                    random_state = 314,
                                    verbose = False,
                                    fit_params = self.fit_params)

            # Fit the random search
            _ = rs.fit(x_train, y_train)
            optimal_params = rs.best_params_
        
        if self.perform_bayes_search:
            
            init_round = 10 
            opt_roun = 10
            n_folds = 6
            random_seed = 42
            n_estimators = 500
            learning_rate = 0.02
            colsample_bytree = 0.93

            # prepare data
            train_data = lgb.Dataset(data = X, label = y)

            # parameters
            def lgb_eval(num_leaves, feature_fraction, bagging_fraction, max_depth, lambda_l1, lambda_l2, min_split_gain, min_child_weight, colsample_bytree, min_child_samples, subsample):
                params = {'application': 'multiclass',
                          'num_iterations': n_estimators, 
                          'learning_rate': learning_rate, 
                          'early_stopping_round': 300, 
                          'metric': 'macroF1'}
                
                params["num_leaves"] = int(round(num_leaves))
                params["num_class"] = 5
                params['feature_fraction'] = max(min(feature_fraction, 1), 0)
                params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
                params['max_depth'] = int(round(max_depth))
                params['lambda_l1'] = max(lambda_l1, 0)
                params['lambda_l2'] = max(lambda_l2, 0)
                params['min_split_gain'] = min_split_gain
                params['min_child_weight'] = min_child_weight
                params['colsample_bytree'] = 0.93
                params['min_child_samples'] = 56,
                params['subsample'] = 0.84
                cv_result = lgb.cv(params, train_data, nfold = n_folds, seed = random_seed, stratified = True, verbose_eval = 200, metrics = ['auc'])
                return max(cv_result['auc-mean'])

            # range 
            lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (19, 45),
                                                    'feature_fraction': (0.1, 0.9),
                                                    'bagging_fraction': (0.8, 1),
                                                    'max_depth': (5, 8.99),
                                                    'lambda_l1': (0, 5),
                                                    'lambda_l2': (0, 3),
                                                    'min_split_gain': (0.001, 0.1),
                                                    'min_child_weight': (5, 50),
                                                    'colsample_bytree' : (0.7,1.0),
                                                    'min_child_samples' : (40,65),
                                                    'subsample' : (0.7,1.0)
                                                   }, random_state = 0)
            # optimize
            lgbBO.maximize(init_points = init_round, n_iter = opt_roun)
            optimal_params = lgbBO.res['max']['max_params']
        
        print("Optimal LGB parameters:")
        print(optimal_params)
        with open("lgb_best_params.pickle", "wb") as lgb_best_params:
            pickle.dump(optimal_params, lgb_best_params)
            
        return optimal_params
    
    def fit(self, X, y, **fit_params):
        print("LGBClassifierCV")
        
        # Use only heads of households for scoring
        X.insert(0, 'Target', y)
        X = X.query('parentesco1 == 1')
        y = X['Target'] - 1
        X = X.drop(['Target', 'parentesco1'], 1)
        print("Number of columns in train - " + str(X.shape[1]))
        
        self.estimators_ = []
        
         # Find the best params using random search
        if self.perform_bayes_search or self.perform_random_search:
#             self.lgb_optimal_params = self.find_best_params_(X, y)
            self.lgb_optimal_params = {'num_leaves': 19, 
                                         'feature_fraction': 0.10676970062446138, 
                                         'bagging_fraction': 0.8, 
                                         'max_depth': 9, 
                                         'lambda_l1': 5.0, 
                                         'lambda_l2': 2.999999999999948, 
                                         'min_split_gain': 0.1, 
                                         'min_child_weight': 49.999999999999986, 
                                         'colsample_bytree': 0.7, 
                                         'min_child_samples': 65, 
                                         'subsample': 1.0}
            
        # Use a simple train-test split. I have found that this gives a better local CV score than
        # K folds.
        if self.use_train_test_split:
            x_train, x_val, y_train, y_val = train_test_split(X, y, test_size = 0.1, random_state = 0)
            
            lgb_model = lgb.LGBMClassifier(**self.lgb_params)
            if self.perform_random_search or self.perform_bayes_search:
                lgb_model.set_params(**self.lgb_optimal_params)
            
            lgb_model.fit(
                    x_train, y_train,
                    eval_set = [(x_train, y_train), (x_val, y_val)],
                    **self.fit_params
            )
            print("Train F1 - " + str(lgb_model.best_score_['train']['macroF1']) + "   " + "Validation F1 - " + str(lgb_model.best_score_['validation']['macroF1']))
            self.estimators_.append(lgb_model)
            
        # When not using random search to tune parameters, proceed with a simple Stratified Kfold CV
        if self.use_kfold_split:
            kf = StratifiedKFold(n_splits = self.cv, shuffle = True)
            for fold_index, (train, valid) in enumerate(kf.split(X, y)):
                print("Train Fold Index - " + str(fold_index))

                lgb_model = lgb.LGBMClassifier(**self.lgb_params)
                if self.perform_random_search:
                    lgb_model.set_params(**self.lgb_optimal_params)

                lgb_model.fit(
                        X.iloc[train], y.iloc[train],
                        eval_set = [(X.iloc[train], y.iloc[train]), (X.iloc[valid], y.iloc[valid])],
                        **self.fit_params
                )
                print("Train F1 - " + str(lgb_model.best_score_['train']['macroF1']) + "   " + "Validation F1 - " + str(lgb_model.best_score_['validation']['macroF1']))

                self.estimators_.append(lgb_model)
        return self
    
    def predict(self, X):
        # Remove this column since we are using only heads of households for scoring
        X = X.drop('parentesco1', 1)
        
        # When not using random search, use voting to get predictions from all CV estimators.
        y_pred = []
        for estimator_index, estimator in enumerate(self.estimators_):
            print("Estimator Index - " + str(estimator_index))
            y_pred.append(estimator.predict(X))
        return np.mean(y_pred, axis = self.axis).astype(int)
def get_lgb_params():
    
    def evaluate_macroF1_lgb(truth, predictions):  
        pred_labels = predictions.reshape(len(np.unique(truth)), -1).argmax(axis = 0)
        f1 = f1_score(truth, pred_labels, average = 'macro')
        return ('macroF1', f1, True)

    def learning_rate_power_0997(current_iter):
            base_learning_rate = 0.1
            min_learning_rate = 0.02
            lr = base_learning_rate  * np.power(.995, current_iter)
            return max(lr, min_learning_rate)

    lgb_params = {'boosting_type': 'dart',
                  'class_weight': 'balanced',
                  "objective": 'multiclassova',
                  'metric': None,
                  'silent': True,
                  'random_state': 0,
                  'n_jobs': -1}
    
    fit_params={"early_stopping_rounds": 400, 
                "eval_metric" : evaluate_macroF1_lgb, 
                'eval_names': ['train', 'validation'],
                'verbose': False,
                'categorical_feature': 'auto'}
    
    return lgb_params, fit_params
lgb_params, lgb_fit_params = get_lgb_params()

pipeline = Pipeline([
    ('na_imputer', MissingValuesImputer(impute_zero_columns = ['v18q1', 'v2a1', 'rez_esc'])),
    ('cat_transformer', CategoricalVariableTransformer()),
    ('unnecessary_columns_remover_transformer', UnnecessaryColumnsRemoverTransformer()),
    ('feature_engineering_transformer', FeatureEngineeringTransformer()),
    ('lgb', LGBClassifierCV(lgb_params = lgb_params,
                            fit_params = lgb_fit_params,
                            cv = 5,
                            perform_random_search = False,
                            perform_bayes_search = True,
                            use_train_test_split = True,
                            use_kfold_split = False)
    )
])


pipeline.fit(x_train.copy(), y_train.copy())
pred = pipeline.predict(x_test.copy())
print("Local CV Score - " + str(pipeline.named_steps['lgb'].cv_score_))
sub_orig['Target'] = pred + 1
sub_orig.to_csv('Pipeline_Base_LGB_'+ str(pipeline.named_steps['lgb'].cv_score_) + '.csv')
print(sub_orig.head())
