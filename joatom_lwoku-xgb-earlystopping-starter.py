# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



from xgboost import XGBClassifier

from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score, confusion_matrix

import seaborn as sns
train=pd.read_csv('/kaggle/input/learn-together/train.csv')

test=pd.read_csv('/kaggle/input/learn-together/test.csv')



train.drop('Id', axis=1, inplace=True)

y_train = train.pop('Cover_Type')

test_ids = test.pop('Id')
n_splits = 3



kf = KFold(n_splits=n_splits, shuffle=True, random_state=2019)

yoof = np.zeros(len(train))

yhat = np.zeros(len(test))



fold = 0

n_est = 2000

lr = 0.2

for in_index, oof_index in kf.split(train, y_train):

    fold += 1

    print(f'fold {fold} of {n_splits}')

    X_in, X_oof = train.values[in_index], train.values[oof_index]

    y_in, y_oof = y_train.values[in_index], y_train.values[oof_index]



    model = XGBClassifier(n_estimators=n_est, learning_rate=lr, random_state=2019, tree_method = 'gpu_exact')



    model.fit(X_in, y_in, early_stopping_rounds=20, eval_set=[(X_oof, y_oof)], verbose=100,eval_metric=['merror', 'mlogloss'])



    print('## lr:',lr,'n_est:',n_est)

    print('Best iteration: '+ str(model.best_iteration), 'Best ntree_limit: '+ str(model.best_ntree_limit), 'Best score:', str(model.best_score))



    yoof[oof_index] = model.predict(X_oof)

    yhat += model.predict(test.values)



yhat /= n_splits

cm=confusion_matrix(y_train, yoof)

sns.heatmap(cm, annot=True, fmt="d", xticklabels=range(1,8), yticklabels=range(1,8))

print('Accuracy:', accuracy_score(y_train, yoof))


