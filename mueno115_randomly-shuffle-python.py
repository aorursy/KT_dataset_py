# Python version of:

# https://www.kaggle.com/brandenkmurray/randomly-shuffled-data-also-works



import os



import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from tqdm import tqdm

import catboost



print(os.listdir("../input"))
df_train = pd.read_csv('../input/train.csv', index_col='ID_code')

df_test = pd.read_csv('../input/test.csv', index_col='ID_code')
# split the train data by the target

target0 = df_train[df_train.target == 0]

target1 = df_train[df_train.target == 1]
((target0.shape, target1.shape))
# shuffle each feature

for c in tqdm(df_test.columns):

    target0.loc[:, c] = target0[c].sample(frac=1).values

    target1.loc[:, c] = target1[c].sample(frac=1).values
# verify shuffled
df_train.head(3)
df_train_shuffled = pd.concat([target0, target1], axis=0)

df_train_shuffled.head(3)
# Train a model



del df_train, target0, target1



Xy_train, Xy_valid = train_test_split(df_train_shuffled, test_size=.2, random_state=0, stratify=df_train_shuffled.target)

del df_train_shuffled



xtr = Xy_train.drop('target', axis=1)

ytr = Xy_train.target

xvalid = Xy_valid.drop('target', axis=1)

yvalid = Xy_valid.target



def make_model():

    return catboost.CatBoostClassifier(

        early_stopping_rounds=100,

        iterations=2000,

        random_seed=0,

        task_type="GPU",



        depth=8,

        eval_metric='Logloss',

        learning_rate=3e-2,

        loss_function='Logloss',

        metric_period=50,   # maintain training speed

        od_pval=1e-6,

        od_type='IncToDec',

        use_best_model=True,

        verbose=200,        # set the logging period

    )



def train(clf, xtr, ytr, xte, yte):

  clf.fit(

      X=xtr,

      y=ytr,

      eval_set=(xte, yte),

      use_best_model=True)

  

def validate(clf, xvalid, yvalid):

  y_pred = clf.predict_proba(xvalid)[:, 1]

  score = roc_auc_score(yvalid, y_pred)

  return score
model = make_model()

train(model, xtr, ytr, xvalid, yvalid)

score = validate(model, xvalid, yvalid)



print('Validation score: {}'.format(score))
# Predict on the test set.

y_pred = model.predict_proba(df_test)[:, 1]
df_submit = pd.DataFrame(dict(target=y_pred), index=df_test.index)

df_submit.to_csv('randomly_shuffled.csv')  # => Public 0.892