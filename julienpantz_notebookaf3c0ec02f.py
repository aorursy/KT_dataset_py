# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import xgboost as xgb



from sklearn.metrics import mean_absolute_error
train = pd.read_csv('../input/train.csv')[0:10000]

test = pd.read_csv('../input/test.csv')[0:10000]
test['loss'] = np.nan

joined = pd.concat([train, test])





def evalerror(preds, dtrain):

    labels = dtrain.get_label()

    return 'mae', mean_absolute_error(np.exp(preds), np.exp(labels))



for column in list(train.select_dtypes(include=['object']).columns):

    if train[column].nunique() != test[column].nunique():

        set_train = set(train[column].unique())

        set_test = set(test[column].unique())

        remove_train = set_train - set_test

        remove_test = set_test - set_train



        remove = remove_train.union(remove_test)

        def filter_cat(x):

            if x in remove:

                return np.nan

            return x



        joined[column] = joined[column].apply(lambda x: filter_cat(x), 1)

            

    joined[column] = pd.factorize(joined[column].values, sort=True)[0]



train = joined[joined['loss'].notnull()]

test = joined[joined['loss'].isnull()]



shift = 200

y = np.log(train['loss'] + shift)

ids = test['id']

X = train.drop(['loss', 'id'], 1)

X_test = test.drop(['loss', 'id'], 1)
def evalerror(preds, dtrain):

    labels = dtrain.get_label()

    return 'mae', mean_absolute_error(np.exp(preds), np.exp(labels))





RANDOM_STATE = 2016



params = {

        'min_child_weight': 1,

        'eta': 0.01,

        'colsample_bytree': 0.5,

        'max_depth': 12,

        'subsample': 0.8,

        'alpha': 1,

        'gamma': 1,

        'silent': 1,

        'verbose_eval': True,

        'seed': RANDOM_STATE

    }



xgtrain = xgb.DMatrix(X, label=y)

xgtest = xgb.DMatrix(X_test)



cvresult = xgb.cv(params, xgtrain, nfold=3,feval=evalerror,num_boost_round=25, early_stopping_rounds=50)

print(cvresult)





model = xgb.train(params, xgtrain, int(2012 / 0.9), feval=evalerror)



prediction = np.exp(model.predict(xgtest)) - shift



submission = pd.DataFrame()

submission['loss'] = prediction

submission['id'] = ids

submission.to_csv('sub_v.csv', index=False)