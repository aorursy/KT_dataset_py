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
train.head()
columns_for_del = []
plt.hist(train['disbursed_amount'], bins=100, label='disbursed_amount')

plt.hist(train['magic_0'], bins=100, label='magic_0', alpha=0.5)

plt.legend()

plt.show()
columns_for_del.extend([i for i in train.columns if i.startswith('magic')])

columns_for_del.extend([i for i in train.columns if i.startswith('f')])
for col in ['ltv', 'PERFORM_CNS.SCORE']:

    plt.hist(train[col], bins=100)

    plt.title(col)

    plt.show()
def check_coverage(x1, x2):

    return len(set(x1) & set(x2)) / len(set(x1))
for col in train.columns:

    print('Column: {}, coverage: {}'.format(col, check_coverage(train[col].fillna(-999), test[col].fillna(-999))))
columns_for_del.extend(['UniqueID'])

for col in train.columns:

    if len(train[col].value_counts()) <= 1:

        print(col, 'train')

    if len(test[col].value_counts()) <= 1:

        print(col, 'test')
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

from copy import deepcopy



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

        

    def cv(self, X, y, verbose=True):

        self.models = []

        self.metrics = []

        oof = np.zeros(len(X))

        for n, (tr, vl) in enumerate(self.params['folds']):

            X_tr, X_vl = X.loc[tr], X.loc[vl]

            y_tr, y_vl = y.loc[tr], y.loc[vl]

            self.model.fit(X_tr, y_tr, X_vl, y_vl)

            self.models.append(deepcopy(self.model))

            oof[vl] = self.model.predict(X_vl)

            

            self.metrics.append(self.params['metric'](y_vl, oof[vl]))

            if verbose:            

                print('Fold {}, metric {:f}'.format(n, self.metrics[-1]))



        if verbose: 

            print('==========')

            print('OOF metric {:f}'.format(self.params['metric'](y, oof)))

            print('Mean folds metric {:f}'.format(np.mean(self.metrics)))

        return oof

    

    def fit(self, X, y):

        oof = self.cv(X, y, verbose=True)

    

    def predict(self, X):

        preds = np.zeros(len(X))

        for n, model in enumerate(self.models):

            preds += self.model.predict(X)

            

        return preds / (n+1)

    

    
params = {'application': 'binary',

          'objective': 'binary',

          'metric': 'auc'

         }



meta_params = {'num_boost_round': 5000,

               'early_stopping_rounds': 100,

               'verbose_eval': 100

              }



model = LGBMWrapper(params, meta_params)



expparams = {'folds': list(KFold(n_splits=5, random_state=0, shuffle=True).split(train)),

             'metric': auc}



exp = Experiment(model, expparams)

exp.fit(train, target)
predict = exp.predict(test)
submission.loc[:, 'Predicted'] = predict

submission.to_csv('submission.csv', index=None)