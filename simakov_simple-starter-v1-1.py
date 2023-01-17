import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

pd.set_option('max_columns',50)
train = pd.read_csv('../input/hse-practical-ml-1/car_loan_train.csv')

test = pd.read_csv('../input/hse-practical-ml-1/car_loan_test.csv')



target = train.target

del train['target']



submission = pd.DataFrame()

submission.loc[:, 'ID'] = test.index

submission.loc[:, 'Predicted'] = 0
train.head()
def preproc_date(df, cols, remove=True):

    for col in cols:

        time = pd.to_datetime(df[col])

        df.loc[:, col + '_year'] = [i.year if i.year<=2020 else i.year - 100 for i in time] # If ‘unix’ (or POSIX) time; origin is set to 1970-01-01.

        df.loc[:, col + '_month'] = [i.month for i in time]

        df.loc[:, col + '_day'] = [i.day for i in time]

        if remove:

            del df[col]

    return df



def parse_date(df, cols, remove=True):

    for col in cols:

        time = df[col].str.split(' ')

        df.loc[:, col + '_year'] = [int(i[0].split('yrs')[0]) for i in time]

        df.loc[:, col + '_month'] = [int(i[1].split('mon')[0]) for i in time]

        if remove:

            del df[col]

    return df



class PreprocText(object):

    def __init__(self, columns):

        self.columns = columns

        self.res = {}

        

    def fit(self, X, y=None):

        for col in self.columns:

            encoder = LabelEncoder()

            encoder.fit(X.loc[:, col].astype(str))

            self.res[col] = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

            

    def transform(self, X, y=None):

        for col in self.columns:

            X.loc[:, col] = X.loc[:, col].map(self.res[col])

        return X

    

    def fit_transform(self, X, y=None):

        self.fit(X)

        return self.transform(X)

            

def data_preproc(df):

    df = preproc_date(df, ['Date.of.Birth', 'DisbursalDate'])

    df = parse_date(df, ['AVERAGE.ACCT.AGE', 'CREDIT.HISTORY.LENGTH'])

    return df



class DataPreproc(object):

    def __init__(self):

        pass

        

    def fit(self, X):

        self.text_prep = PreprocText(columns=['Employment.Type', 'PERFORM_CNS.SCORE.DESCRIPTION'])

        self.text_prep.fit(X)

        

            

    def transform(self, X, y=None):

        X = self.text_prep.transform(X)

        X = data_preproc(X)

        return X

    

    def fit_transform(self, X, y=None):

        self.fit(X)

        return self.transform(X)    
%%time



prep = DataPreproc()

train = prep.fit_transform(train)

test = prep.transform(test)
columns_for_del = []
columns_for_del.extend([i for i in train.columns if i.startswith('magic')])

columns_for_del.extend([i for i in train.columns if i.startswith('f')])
columns_for_del.extend(['UniqueID'])

columns_for_del.extend(['MobileNo_Avl_Flag', 'DisbursalDate_year'])
def create_features(df):

    df.loc[:, 'ltv_round'] = df.loc[:, 'ltv'] - df.loc[:, 'ltv'] % 1 # целая часть

    df.loc[:, 'ltv_%1'] = df.loc[:, 'ltv'] % 1 # кратны одному / дробная часть

    df.loc[:, 'ltv_round_%5'] = df.loc[:, 'ltv_round'] % 5 # кратны 5

    df.loc[:, 'ltv_round_%10'] = df.loc[:, 'ltv_round'] % 10 # кратны 10

    return df



class CreateCounts(object):

    def __init__(self, columns):

        self.columns = columns

        self.res = {}

        

    def fit(self, X, y=None):

        for col in self.columns:

            _dict = X[col].value_counts().to_dict()

            self.res[col] = _dict

    

    def transform(self, X, y=None):

        for col in self.columns:

            X.loc[:, col + '_counts'] = X.loc[:, col].map(self.res[col])

        return X

    

    def fit_transform(self, X, y=None):

        self.fit(X)

        return self.transform(X)

    

class CreateFeats(object):

    def __init__(self):

        self.counts = None

        

    def fit(self, X):

        self.counts = CreateCounts(columns=['disbursed_amount', 'asset_cost', 'branch_id', 'supplier_id',

                                              'manufacturer_id', 'Current_pincode_ID', 'State_ID',

                                              'Employee_code_ID', 'PERFORM_CNS.SCORE.DESCRIPTION'])

        self.counts.fit(X)

        

            

    def transform(self, X, y=None):

        X = self.counts.transform(X)

        X = create_features(X)

        return X

    

    def fit_transform(self, X, y=None):

        self.fit(X)

        return self.transform(X)    
%%time



fgen = CreateFeats()

train = fgen.fit_transform(train)

test = fgen.transform(test)
train.head()
for col in columns_for_del:

    del train[col]

    del test[col]
from sklearn.metrics import roc_auc_score as auc

from sklearn.model_selection import KFold

import lightgbm as lgb

from copy import deepcopy, copy



class LGBMWrapper(object):

    def __init__(self, params, meta_params):

        self.params = params

        self.meta_params = meta_params

        

    def fit(self, X, y, X_val, y_val):

        train_data = lgb.Dataset(X, label=y)

        val_data = lgb.Dataset(X_val, label=y_val)

        self.model = lgb.train(self.params,

                               train_data,

                               valid_sets=val_data,

                               **self.meta_params,

                              )

        

    def predict(self, X):

        return self.model.predict(X)

    

class Experiment(object):

    def __init__(self, model, params):

        self.model = model

        self.params = params

        self.models = []

        self.metrics = []

        

    def cv(self, X, y, imp=True, verbose=True):

        self.models = []

        oof = np.zeros(len(X))

        

        if imp:

            self.metrics = []

            self.gain_imp = np.zeros(X.shape[1])

            self.split_imp = np.zeros(X.shape[1])

        

        for n, (tr, vl) in enumerate(self.params['folds']):

            X_tr, X_vl = X.loc[tr], X.loc[vl]

            y_tr, y_vl = y.loc[tr], y.loc[vl]

            self.model.fit(X_tr, y_tr, X_vl, y_vl)

            if imp:

                self.gain_imp += self.model.model.feature_importance(importance_type='gain')

                self.split_imp += self.model.model.feature_importance(importance_type='split')                

            

            self.models.append(copy(self.model))

            oof[vl] = self.model.predict(X_vl)

            

            

            if verbose:  

                self.metrics.append(self.params['metric'](y_vl, oof[vl]))

                print('Fold {}, metric {:f}'.format(n, self.metrics[-1]))

                

        

        if verbose: 

            self.oof_metric = self.params['metric'](y, oof)

            print('==========')

            print('OOF metric {:f}'.format(self.oof_metric))

            print('Mean folds metric {:f}'.format(np.mean(self.metrics)))

        

        if imp:

            self.gain_imp /= (n + 1)

            self.split_imp /= (n + 1)

        

        return oof

    

    def fit(self, X, y, imp=True, verbose=True):

        self.columns = list(X.columns)

        oof = self.cv(X, y, imp, verbose)

        self.imp = pd.DataFrame(index=self.columns)

        self.imp.loc[:, 'gain'] = self.gain_imp

        self.imp.loc[:, 'split'] = self.split_imp

    

    def predict(self, X):

        preds = np.zeros(len(X))

        for n, model in enumerate(self.models):

            preds += self.model.predict(X)

            

        return preds / (n+1)

    

    def get_importances(self, sort_by='gain'):

        return self.imp.sort_values(sort_by, ascending=False)

    

    def forward(self, X, y, imp='gain', tol=4):

        imps = self.get_importances(sort_by=imp)

        sorted_cols = list(imps.index)

        

        c = 0

        best_metric = 0

        usefull_columns = []

        for col in sorted_cols:

            usefull_columns.append(col)

            oof = self.cv(X[usefull_columns], y, False, False)

            metric = self.params['metric'](y, oof)

            if metric > best_metric:

                print('Column {} added, {:f} -> {:f}'.format(col, best_metric, metric))

                best_metric = metric

                c = 0

            else:

                usefull_columns.pop()

                c += 1

                

            if c >= tol:

                break

        

        print('========')

        print('Best metric {:f}'.format(best_metric))

                

        return usefull_columns

    

    def backward(self, X, y, imp='gain', tol=4):

        imps = self.get_importances(sort_by=imp)

        sorted_cols = list(reversed(imps.index))

        

        c = 0

        best_metric = self.oof_metric

        usefull_columns = sorted_cols

        for col in sorted_cols:

            usefull_columns = sorted(list(set(usefull_columns) - {col}))

            oof = self.cv(X[usefull_columns], y, False, False)

            metric = self.params['metric'](y, oof)

            if metric > best_metric:

                print('Column {} removed, {:f} -> {:f}'.format(col, best_metric, metric))

                best_metric = metric

                c = 0

            else:

                usefull_columns.append(col)

                c += 1

                

            if c >= tol:

                break

        print('========')

        print('Best metric {:f}'.format(best_metric))

                

        return usefull_columns

                

            

    

    
params = {'application': 'binary',

          'objective': 'binary',

          'metric': 'auc'

         }



meta_params = {'num_boost_round': 5000,

               'early_stopping_rounds': 100,

               'verbose_eval': None

              }



model = LGBMWrapper(params, meta_params)



expparams = {'folds': list(KFold(n_splits=5, random_state=0, shuffle=True).split(train)),

             'metric': auc}



exp = Experiment(model, expparams)

exp.fit(train, target)
exp.get_importances().head(10)
ff_split = exp.forward(train, target, imp='split')
params = {'application': 'binary',

          'objective': 'binary',

          'metric': 'auc',

          'learning_rate': 0.02,

          'min_data_in_leaf': 200,

          'bagging_fraction': 0.8,

          'bagging_freq': 1,

          'feature_fraction': 0.3,

         }



meta_params = {'num_boost_round': 5000,

               'early_stopping_rounds': 200,

               'verbose_eval': None

              }



model = LGBMWrapper(params, meta_params)



expparams = {'folds': list(KFold(n_splits=5, random_state=0, shuffle=True).split(train)),

             'metric': auc}



exp = Experiment(model, expparams)

exp.fit(train[ff_split], target)
predict = exp.predict(test[ff_split])
submission.loc[:, 'Predicted'] = predict

submission.to_csv('submission.csv', index=None)