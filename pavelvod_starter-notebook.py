# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_validate, cross_val_predict, cross_val_score

from IPython.display import display

from sklearn.metrics import roc_auc_score

from pathlib import Path



root_path = Path('/kaggle/input/fight-bias-part-1')

list(root_path.glob('*'))



# Any results you write to the current directory are saved as output.
train = pd.read_csv(root_path / 'train.csv').set_index('index')

test = pd.read_csv(root_path / 'test.csv').set_index('index')

sample_submission = pd.read_csv(root_path / 'sample_submission.csv')


X = train.drop(columns=['label'])

y = train['label']





models = dict(LogisticRegression=(LogisticRegression, dict(C=1)),

              RandomForestClassifier=(RandomForestClassifier, dict(min_samples_leaf=1000, min_samples_split=1000, n_estimators=30))

             )
from sklearn.model_selection import StratifiedKFold

from tqdm import tqdm



n_splits = 5



cv_cls = StratifiedKFold

cv_params = dict(shuffle=True, random_state=42)



def cross_validate_model(model_name, 

                         model_cls, 

                         model_params, 

                         X,

                         y,

                         test=pd.DataFrame(),

                         cv_cls=StratifiedKFold, 

                         n_splits = 5, 

                         cv_params = dict(shuffle=True, random_state=42)

                        ):

    skf = cv_cls(n_splits=n_splits, **cv_params)



    oof_trn = np.zeros(shape=y.shape)

    oof_val = np.zeros(shape=y.shape)

    

    oof_tst = np.zeros(shape=test.values.shape[0])

    for fold_idx, (trn_idx, val_idx) in tqdm(enumerate(skf.split(X, y)), total = n_splits):

        x_trn, x_val = X.iloc[trn_idx, :].values, X.iloc[val_idx, :].values

        y_trn, y_val = y.iloc[trn_idx].values, y.iloc[val_idx].values



        model_obj = model_cls(**model_params).fit(x_trn, y_trn)

        oof_trn[trn_idx] += model_obj.predict_proba(x_trn)[:, 1] / (n_splits - 1)

        oof_val[val_idx] = model_obj.predict_proba(x_val)[:, 1]

        if test.index.size > 0:

            oof_tst += model_obj.predict_proba(test)[:, 1] / n_splits

    print(model_name, dict(train_score=roc_auc_score(y, oof_trn), validation_score=roc_auc_score(y, oof_val)))

    return oof_tst, roc_auc_score(y, oof_val)

# data_prop = pd.concat([X.assign(label=1),test.assign(label=0)])

# X_prop = data_prop.drop(columns=['label'])

# y_prop = data_prop['label']



# model_outputs = {}

# model_val_scores = {}

# for model_name, (model_cls, model_params) in models.items():

#     model_outputs[model_name], model_val_scores[model_name] = cross_validate_model(model_name, 

#                                    model_cls, 

#                                    model_params, 

#                                    X_prop,

#                                    y_prop)
model_outputs = {}

model_val_scores = {}

for model_name, (model_cls, model_params) in models.items():

    model_outputs[model_name], model_val_scores[model_name] = cross_validate_model(model_name, 

                                   model_cls, 

                                   model_params, 

                                   X,

                                   y,

                                   test)
best_model = pd.Series(model_val_scores).idxmax()

best_model
sample_submission['label'] = model_outputs[best_model]

sample_submission.to_csv('submission.csv', index=False)