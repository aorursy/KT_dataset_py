import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import BayesianRidge

from sklearn.model_selection import RepeatedKFold

import os

print(os.listdir("../input/"))
#Import the Validation output and submissions



oof = pd.read_csv("../input/santander-outputs/Validation_Skfold.csv")['0']

oof_2 = pd.read_csv("../input/santander-outputs/Validation_kfold.csv")['0']



predictions = pd.read_csv("../input/santander-outputs/submission26_skfold.csv")["target"]

predictions_2 = pd.read_csv("../input/santander-outputs/submission26_kfold.csv")["target"]
train = pd.read_csv('../input/santander-customer-transaction-prediction/train.csv')

features = [c for c in train.columns if c not in ['ID_code', 'target']]
target = train['target']

train = train.drop(["ID_code", "target"], axis=1)
train_stack = np.vstack([oof,oof_2]).transpose()

test_stack = np.vstack([predictions, predictions_2]).transpose()



folds_stack = RepeatedKFold(n_splits=5, n_repeats=2, random_state=15)

oof_stack = np.zeros(train_stack.shape[0])

predictions_3 = np.zeros(test_stack.shape[0])



for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack,target)):

    print("fold {}".format(fold_))

    trn_data, trn_y = train_stack[trn_idx], target.iloc[trn_idx].values

    val_data, val_y = train_stack[val_idx], target.iloc[val_idx].values

    

    clf_3 = BayesianRidge()

    clf_3.fit(trn_data, trn_y)

    

    oof_stack[val_idx] = clf_3.predict(val_data)

    predictions_3 += clf_3.predict(test_stack) / 10
sample_submission = pd.read_csv('../input/santander-customer-transaction-prediction/sample_submission.csv')

sample_submission['target'] = predictions_3

sample_submission.to_csv('submission_ashish.csv', index=False)