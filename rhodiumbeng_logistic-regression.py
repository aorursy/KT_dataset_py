import numpy as np

import pandas as pd

import os

print(os.listdir("../input"))
# load train & test set

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

sub = pd.read_csv('../input/sample_submission.csv')

print(train_df.shape, test_df.shape, sub.shape)
predict = pd.DataFrame(columns=['index', 'pred'])

print(predict.shape)
from sklearn.linear_model import LogisticRegression

from sklearn import metrics



for i in range(512):

    print('Processing data set # '+str(i))

    train = train_df[train_df['wheezy-copper-turtle-magic']==i]

    test = test_df[test_df['wheezy-copper-turtle-magic']==i]

    index = test.index

    X_train = train.drop(['id', 'wheezy-copper-turtle-magic', 'target'], axis=1).values

    Y_train = train['target'].values

    X_test = test.drop(['id', 'wheezy-copper-turtle-magic'], axis=1).values

    model = LogisticRegression(solver='liblinear', penalty='l1', C=0.05)

    model.fit(X_train, Y_train)

    Y_pred_train = model.predict(X_train)

    print('AUC is ', metrics.roc_auc_score(Y_train, Y_pred_train))

    preds = model.predict_proba(X_test)

    for i in range(len(index)):

        predict = predict.append({'index':index[i], 'pred':preds[i,1]}, ignore_index=True)
temp = predict

temp.sort_values(by=['index'], inplace=True)

temp.head()
temp = temp.set_index('index')

temp.head()
sub['target'] = temp['pred']

sub.head(20)
sub.to_csv('submission.csv',index=False)