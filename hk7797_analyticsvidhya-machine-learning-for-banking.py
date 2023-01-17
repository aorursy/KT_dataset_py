import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import lightgbm as lgb

import requests

from urllib.request import urlretrieve

import shutil

from io import StringIO, BytesIO

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

import statsmodels.api as sm

from sklearn.metrics import mean_absolute_error, roc_auc_score, log_loss, f1_score

import warnings

from termcolor import colored

from datetime import timedelta, datetime

from tqdm.notebook import tqdm



from functools import partial

import numpy as np

import scipy as sp



tqdm.pandas()



warnings.filterwarnings("ignore")
def data_from_link(link):

    re = requests.get(link)

    assert re.status_code == 200, 'Download Failed'

    return pd.read_csv(BytesIO(re.content))
train_link = 'https://datahack-prod.s3.amazonaws.com/train_file/train_fNxu4vz.csv'

test_link = 'https://datahack-prod.s3.amazonaws.com/test_file/test_fjtUOL8.csv'

submission_link = 'https://datahack-prod.s3.amazonaws.com/sample_submission/sample_submission_HSqiq1Q.csv'
%%time

train = data_from_link(train_link)

print('train file have {} rows and {} columns'.format(train.shape[0], train.shape[1]))

test = data_from_link(test_link)

print('test file have {} rows and {} columns'.format(test.shape[0], test.shape[1]))

submission = data_from_link(submission_link)

print('submission file have {} rows and {} columns'.format(submission.shape[0], submission.shape[1]))



train['Loan_Amount_Requested'] = train['Loan_Amount_Requested'].str.replace(',', '').astype('float')

test['Loan_Amount_Requested'] = test['Loan_Amount_Requested'].str.replace(',', '').astype('float')

train['Length_Employed'] = train['Length_Employed'].replace({'< 1 year': '0', '1 year': '1', '10+ years': '10'}).str.replace(' years', '').astype('float')

test['Length_Employed'] = test['Length_Employed'].replace({'< 1 year': '0', '1 year': '1', '10+ years': '10'}).str.replace(' years', '').astype('float')
train.head(2)
test.head(2)
submission.head(2)
class OptimizedRounder_v2(object):

    def __init__(self):

        self.coef_ = 0

    

    def _kappa_loss(self, coef, X, y):

        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [1, 2, 3])

        return - f1_score(y, preds, average='weighted')

    

    def fit(self, X, y):

        loss_partial = partial(self._kappa_loss, X = X, y = y)

        initial_coef = [1.5, 2.5]

        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method = 'nelder-mead')['x']

    

    def predict(self, X, coef=None):

        if coef is None:

            coef = self.coefficients()

        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [1, 2, 3])

        return preds

    

    def coefficients(self):

        return self.coef_
class SimpleModel:

    def __init__(self, train, n_splits, params, categorical):

        self.feature = [col for col in train.columns if col not in ['Loan_ID', 'Interest_Rate']]

        self.categorical = categorical

        self.target = 'Interest_Rate'

        self.n_splits = n_splits

        self.params = params

        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        train = train.copy()

        train[self.categorical] = train[self.categorical].astype('category')

        self.models = []

        self.optimized_rounders = []

        oof_pred = np.zeros((len(train),))

        oof_pred_class = np.zeros((len(train),))

        

        for fold, (train_idx, val_idx) in enumerate(cv.split(train, train[self.target])):

            x_train, x_val = train[self.feature].iloc[train_idx], train[self.feature].iloc[val_idx]

            y_train, y_val = train[self.target][train_idx], train[self.target][val_idx]

            train_set, val_set = self.convert_dataset(x_train, y_train, x_val, y_val)

            model = self.train_model(train_set, val_set)

            oof_pred[val_idx] = model.predict(x_val)

            self.models.append(model)

            print('\n\n optimized rounder')

            pred = model.predict(x_train)

            opt_rounder = OptimizedRounder_v2()

            opt_rounder.fit(pred, y_train)

            self.optimized_rounders.append(opt_rounder)

            print(f'coef: {opt_rounder.coefficients()}')

            oof_pred_class[val_idx] = opt_rounder.predict(oof_pred[val_idx])



            print('Partial score of fold {} is: {}'.format(fold, f1_score(y_val, opt_rounder.predict(oof_pred_class[val_idx]), average='weighted')))

        

        

        loss_score = f1_score(train[self.target], oof_pred_class, average='weighted')

        print('Our f1 score is: ', loss_score)



        opt_rounder = OptimizedRounder_v2()

        for i in range(len(self.optimized_rounders)):

            if i == 0:

                opt_rounder.coef_ = self.optimized_rounders[i].coef_/self.n_splits

            else:

                opt_rounder.coef_ += self.optimized_rounders[i].coef_/self.n_splits

        print(f'coef: {opt_rounder.coefficients()}')

        self.opt_rounder_avg = opt_rounder



        loss_score = f1_score(train[self.target], self.opt_rounder_avg.predict(oof_pred), average='weighted')

        print('Our f1 score is (opt avg): ', loss_score)





    

    def convert_dataset(self, x_train, y_train, x_val, y_val):

        train_set = lgb.Dataset(x_train, y_train)

        val_set = lgb.Dataset(x_val, y_val)

        return train_set, val_set

    

    def train_model(self, train_set, val_set):

        verbosity = 100

        return lgb.train(self.params, train_set, valid_sets=[train_set, val_set], 

                         categorical_feature=self.categorical, verbose_eval=verbosity)

    

    def predict(self, test_df):

        x_test = test_df[self.feature].copy()

        x_test[self.categorical] = x_test[self.categorical].astype('category')

        y_pred = np.zeros((len(test_df), ))

        for model_ in self.models:

            y_pred += model_.predict(x_test) / self.n_splits

        return self.opt_rounder_avg.predict(y_pred)
categorical_1 = ['Home_Owner', 'Income_Verified', 'Purpose_Of_Loan', 'Gender']



params = {'n_estimators':2500,

            'boosting_type': 'gbdt',

            'objective': 'fair',

            'metric': ['l2', 'l1'],

            'subsample': 0.70,

            'subsample_freq': 1,

            'learning_rate': 0.02,

            'feature_fraction': 0.75,

            'max_depth': 13,

            'lambda_l1': 0.5,  

            'lambda_l2': 0.5,

            'early_stopping_rounds': 100,

            'seed': 42

            }



model_train_1 = SimpleModel(train, n_splits=5, params=params, categorical=categorical_1)
train['Interest_Rate'].value_counts(dropna=False, normalize=True)
np.all(submission['Loan_ID'] == test['Loan_ID'])
submission['Interest_Rate'] = model_train_1.predict(test)

submission.head()
submission['Interest_Rate'].value_counts(dropna=False, normalize=True)
submission.to_csv('submission_fair_loss.csv', index=False)