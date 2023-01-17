import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier as rfc

from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold

from sklearn.metrics import roc_auc_score

from datetime import timedelta, datetime

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.linear_model import LogisticRegression as lr

import sys

from sklearn.svm import SVC

from joblib import Parallel, delayed

from sklearn.feature_selection import RFE

from xgboost import XGBClassifier

from catboost import CatBoostClassifier

import optuna

from lightgbm import LGBMClassifier

import shap
class StrangeTransformer(BaseEstimator, TransformerMixin):

    def process_strange_columns(self, X):

        df = X.copy()

        for col in df.columns.tolist():

            df[col] = df[col].apply(

                lambda x: int(x[: x.find('y')]) * 12

                + int(x[x.find('s') + 1: x.find('m')])

            )

        return df





    def fit(self, X, y=None):

        return self





    def transform(self, X):

        self.X = self.process_strange_columns(X)

        return self.X



    

class DateTransformer(BaseEstimator, TransformerMixin):

    def process_date(self, X):

        df = X.copy()

        for col in X.columns.tolist():

            df[col] = pd.to_numeric((

                pd.to_datetime(df[col])

                - pd.to_datetime(datetime(2000, 1, 1))

            ).dt.days)

        return df





    def fit(self, X, y=None):

        return self





    def transform(self, X):

        self.X = self.process_date(X)

        return self.X





class TargetEncoder():

    def __init__(self, feats, target, alpha):

        self.feats = feats

        self.target = target

        self.alpha = alpha

        self.agg = []





    def calc_agg(self, df1, feat, target):

        df = df1.copy()

        mean = df[target].mean()

        agg = df.groupby(feat)[target].agg(['count', 'mean'])

        mean_i = agg['mean']

        count = agg['count']

        aggregated = (mean_i * count + self.alpha * mean) / (count + self.alpha)

        return {feat: aggregated}





    def fit(self, X, y=None, n_jobs=4):

        self.agg = Parallel(n_jobs)(

            delayed(self.calc_agg)(X, feat, self.target)

            for feat in self.feats

        )

        self.agg = {k: v for i in self.agg for k, v in i.items()}





    def transform(self, X, n_jobs=4):

        new_columns = Parallel(n_jobs)(

            delayed(X[feat].map)(self.agg[feat])

            for feat in self.feats

        )

        new_columns = [

            column.rename('te_' + column.name)

            for column in new_columns

        ]

        return pd.concat(new_columns, axis=1)





class CounterTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):

        self.agg = {}





    def calc_agg(self, df, feat):

        agg = df.groupby(feat)['tmp'].agg('count') / df.shape[0]

        return {feat: agg}





    def fit(self, X, y=None, n_jobs=4):

        df = X.copy()

        df['tmp'] = 1

        self.agg = Parallel(n_jobs)(

            delayed(self.calc_agg)(df, feat) for feat in X.columns.tolist()

        )

        self.agg = {k: v for i in self.agg for k, v in i.items()}

        return self





    def transform(self, X, n_jobs=4):

        new_columns = Parallel(n_jobs)(

            delayed(X[feat].map)(self.agg[feat]) for feat in X.columns.tolist()

        )

        new_columns = [col.rename('ce_' + col.name) for col in new_columns]

        return pd.concat(new_columns, axis=1)
raw_train_data = pd.read_csv('/kaggle/input/hse-practical-ml-1/car_loan_train.csv')

raw_test_data = pd.read_csv('/kaggle/input/hse-practical-ml-1//car_loan_test.csv')
categorical_columns = []

for col in raw_train_data.columns.tolist():

    #print(col)

    if len(raw_train_data[col].unique()) < 50:

        categorical_columns.append(col)

        #print("CAT?", raw_train_data[col].unique())

    #else:

        #print(raw_train_data[col].unique()[:20])
train = raw_train_data.drop('UniqueID', axis=1)

#print('NANs:', train.columns[train.isna().any()])

train = train.fillna('0')



X = train.drop('target', axis=1)

y = train['target'].values

strange_columns = ['AVERAGE.ACCT.AGE', 'CREDIT.HISTORY.LENGTH']

date_columns = ['Date.of.Birth', 'DisbursalDate']

small_numeric = [

    'NO.OF_INQUIRIES', 'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS',

    'NEW.ACCTS.IN.LAST.SIX.MONTHS', 'SEC.NO.OF.ACCTS',

    'PRI.ACTIVE.ACCTS', 'SEC.ACTIVE.ACCTS',

    'PRI.OVERDUE.ACCTS', 'SEC.OVERDUE.ACCTS'

    

]

categorical_columns = list(

    set(categorical_columns)

    - set(small_numeric)

    - set(date_columns)

    - set(['target'])

)



numeric_columns = list(

    set(X.columns.tolist())

    - set(categorical_columns)

    - set(strange_columns)

    - set(date_columns)

)
folds = list(StratifiedKFold(3, True, random_state=0).split(train, np.ones(train.shape[0])))

target_encoder = TargetEncoder(categorical_columns, 'target', 0)

counter_encoder = CounterTransformer()

te_columns = {}

params = {

    'lambda_l1': 5.11118721135484e-08,

    'lambda_l2': 0.0028568361304161603,

    'num_leaves': 34,

    'feature_fraction': 0.4009450288266103,

    'bagging_fraction': 0.9396605105017192,

    'bagging_freq': 5,

    'min_child_samples': 76

}

for i, (otr, ote) in enumerate(folds):

    xtr, xte = train.iloc[otr, :], train.iloc[ote, :]

    te_columns[i] = {}

    te_columns[i]['train'] = xtr[categorical_columns].copy()

    te_columns[i]['train'] = te_columns[i]['train'].rename(

        columns={i: 'te_' + str(i) for i in categorical_columns}

    )



    internal_folds = list(StratifiedKFold(3, True, random_state=0).split(xtr, np.ones(xtr.shape[0])))

    for itr, ite in internal_folds:

        target_encoder.fit(xtr.iloc[itr])

        te_columns[i]['train'].iloc[ite] = target_encoder.transform(xtr.iloc[ite])

    target_encoder.fit(xtr)

    te_columns[i]['test'] = target_encoder.transform(xte)

    X_train = pd.concat(

        [xtr[numeric_columns]]

        + [DateTransformer().fit(xtr[date_columns]).transform(xtr[date_columns])]

        + [StrangeTransformer().fit(xtr[strange_columns]).transform(xtr[strange_columns])]

        + [CounterTransformer().fit(xtr[categorical_columns]).transform(xtr[categorical_columns])]

        + [te_columns[i]['train']],

        axis=1

    ).astype('float64')

    y_train = xtr['target'].values

    X_test = pd.concat(

        [xte[numeric_columns]]

        + [DateTransformer().fit(xte[date_columns]).transform(xte[date_columns])]

        + [StrangeTransformer().fit(xte[strange_columns]).transform(xte[strange_columns])]

        + [CounterTransformer().fit(xte[categorical_columns]).transform(xte[categorical_columns])]

        + [te_columns[i]['test']],

        axis=1

    ).astype('float64')

    y_test = xte['target'].values

    clf = LGBMClassifier(**params)

    model = clf.fit(X_train, y_train)

    yhat = model.predict_proba(X_test)[:, 1].reshape(-1, 1)

    print(

        f"Test ROC_AUC "

        f"{roc_auc_score(y_test.reshape(-1, 1), yhat)}"

    )

    print(

        f"Train ROC_AUC "

        f"{roc_auc_score(y_train.reshape(-1, 1), model.predict_proba(X_train)[:, 1].reshape(-1, 1))}"

    )
def objective(trial):

    yhats = np.ones(train.shape[0]).reshape(-1, 1)

    folds = list(StratifiedKFold(2, True, random_state=0).split(train, yhats))

    target_encoder = TargetEncoder(categorical_columns, 'target', 0)

    param = {

        'objective': 'binary',

        'metric': 'binary_logloss',

        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),

        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),

        'num_leaves': trial.suggest_int('num_leaves', 2, 256),

        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),

        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),

        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),

        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),

    }

    for i, (otr, ote) in enumerate(folds):

        xtr, xte = train.iloc[otr, :], train.iloc[ote, :]

        te_columns[i] = {}

        te_columns[i]['train'] = xtr[categorical_columns].copy()

        te_columns[i]['train'] = te_columns[i]['train'].rename(

            columns={i: 'te_' + str(i) for i in categorical_columns}

        )



        internal_folds = list(StratifiedKFold(3, True, random_state=0).split(xtr, np.ones(xtr.shape[0])))

        for itr, ite in internal_folds:

            target_encoder.fit(xtr.iloc[itr])

            te_columns[i]['train'].iloc[ite] = target_encoder.transform(xtr.iloc[ite])

        target_encoder.fit(xtr)

        te_columns[i]['test'] = target_encoder.transform(xte)

        X_train = pd.concat(

            [xtr[numeric_columns]]

            + [DateTransformer().fit(xtr[date_columns]).transform(xtr[date_columns])]

            + [StrangeTransformer().fit(xtr[strange_columns]).transform(xtr[strange_columns])]

            + [te_columns[i]['train']],

            axis=1

        ).astype('float64')

        y_train = xtr['target'].values

        X_test = pd.concat(

            [xte[numeric_columns]]

            + [DateTransformer().fit(xte[date_columns]).transform(xte[date_columns])]

            + [StrangeTransformer().fit(xte[strange_columns]).transform(xte[strange_columns])]

            + [te_columns[i]['test']],

            axis=1

        ).astype('float64')

        y_test = xte['target'].values

        clf = LGBMClassifier(**param)

        model = clf.fit(X_train, y_train)

        yhat = model.predict_proba(X_test)[:, 1].reshape(-1, 1)

        print(

            f"Test ROC_AUC "

            f"{roc_auc_score(y_test.reshape(-1, 1), yhat)}"

        )

        print(

            f"Train ROC_AUC "

            f"{roc_auc_score(y_train.reshape(-1, 1), model.predict_proba(X_train)[:, 1].reshape(-1, 1))}"

        )

        yhats[ote] = yhat

    return roc_auc_score(train['target'].values.reshape(-1, 1), yhats)

 

study = optuna.create_study(direction='maximize')

study.optimize(objective, n_trials=10)

 

print('Number of finished trials:', len(study.trials))

print('Best trial:', study.best_trial.params)
params = study.best_trial.params

X_train = pd.concat(

    [train[numeric_columns]]

    + [DateTransformer().fit(train[date_columns]).transform(train[date_columns])]

    + [StrangeTransformer().fit(train[strange_columns]).transform(train[strange_columns])]

    + [CounterTransformer().fit(train[categorical_columns]).transform(train[categorical_columns])],

    axis=1

).astype('float64')

y_train = train['target'].values

clf = LGBMClassifier(**params)

model = clf.fit(X_train, y_train)

shap_values = shap.TreeExplainer(model).shap_values(X_train)

shap.summary_plot(shap_values, X_train, max_display=100)
folds = list(StratifiedKFold(3, True, random_state=0).split(train, np.ones(train.shape[0])))

target_encoder = TargetEncoder(categorical_columns, 'target', 0)

counter_encoder = CounterTransformer()

te_columns = {}

yhats = []

params = study.best_trial.params

for i, (otr, ote) in enumerate(folds):

    xtr, xte = train.iloc[otr, :], raw_test_data.fillna('0')

    te_columns[i] = {}

    te_columns[i]['train'] = xtr[categorical_columns].copy()

    te_columns[i]['train'] = te_columns[i]['train'].rename(

        columns={i: 'te_' + str(i) for i in categorical_columns}

    )



    internal_folds = list(StratifiedKFold(3, True, random_state=0).split(xtr, np.ones(xtr.shape[0])))

    for itr, ite in internal_folds:

        target_encoder.fit(xtr.iloc[itr])

        te_columns[i]['train'].iloc[ite] = target_encoder.transform(xtr.iloc[ite])

    target_encoder.fit(xtr)

    te_columns[i]['test'] = target_encoder.transform(xte)

    X_train = pd.concat(

        [xtr[numeric_columns]]

        + [DateTransformer().fit(xtr[date_columns]).transform(xtr[date_columns])]

        + [StrangeTransformer().fit(xtr[strange_columns]).transform(xtr[strange_columns])]

        + [CounterTransformer().fit(xtr[categorical_columns]).transform(xtr[categorical_columns])]

        + [te_columns[i]['train']],

        axis=1

    ).astype('float64')

    y_train = xtr['target'].values

    X_test = pd.concat(

        [xte[numeric_columns]]

        + [DateTransformer().fit(xte[date_columns]).transform(xte[date_columns])]

        + [StrangeTransformer().fit(xte[strange_columns]).transform(xte[strange_columns])]

        + [CounterTransformer().fit(xte[categorical_columns]).transform(xte[categorical_columns])]

        + [te_columns[i]['test']],

        axis=1

    ).astype('float64')

    clf = LGBMClassifier(**params)

    model = clf.fit(X_train, y_train)

    yhats.append(model.predict_proba(X_test)[:, 1].reshape(-1, 1))
submission = raw_test_data.rename(columns={'UniqueID': 'ID'})

submission['Predicted'] = np.mean(np.hstack(yhats), axis=1).reshape(-1, 1)

submission['ID'] = np.arange(submission.shape[0])

submission = submission[['ID', 'Predicted']]



print(submission.head())



submission.to_csv('submission.csv', index=None)
