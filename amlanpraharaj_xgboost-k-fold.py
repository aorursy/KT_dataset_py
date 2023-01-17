#importing Libraries

import pandas as pd

from sklearn import metrics

import numpy as np

from sklearn.model_selection import KFold

import xgboost as xgb
train = pd.read_csv('../input/train.tsv',delimiter='\t')
test = pd.read_csv('../input/test.tsv',delimiter='\t')
train = train.rename(columns = {'train_id':'id'})

test = test.rename(columns = {'test_id':'id'})
#Using Data Pre Processing from Naive Catboost kernel

#https://www.kaggle.com/bguberfain/naive-catboost



def factorize(train, test, col, min_count):

    train_cat_count = train[col].value_counts()

    test_cat_count = test[col].value_counts()

    

    train_cat = set(train_cat_count[(train_cat_count >= min_count)].index)



    cat_ids = {k:i for i, k in enumerate(sorted(train_cat))}

    cat_ids[np.nan] = -1

    

    train[col] = train[col].map(cat_ids)

    train[col] = train[col].fillna(len(cat_ids))  # Create 'other' category



    test[col] = test[col].map(cat_ids)

    test[col] = test[col].fillna(len(cat_ids))



# Factorize string columns

factorize(train, test, 'category_name', min_count=50)

factorize(train, test, 'brand_name', min_count=50)
train.head()
test.head()
train1 = train.drop(['price', 'name', 'item_description'], axis=1).copy()

y = np.log(train['price'] + 1)
test1 = test.drop(['name', 'item_description'], axis=1).copy()
KF = KFold(n_splits = 5, shuffle = True, random_state=1)
preds = np.zeros([test1.shape[0],])
#K-Fold

params = {'eta': 0.4, 'max_depth': 6, 'objective': 'reg:linear',

          'eval_metric': 'rmse', 'seed': 99, 'alpha':2}



for i,(train_ind, test_ind) in enumerate(KF.split(train1)):

    print('========Fold',i)

    Xtrain, XCV, ytrain, yCV = train1.iloc[train_ind], train1.iloc[test_ind], y.values[train_ind], y.values[test_ind]

    

    model = xgb.train(params, xgb.DMatrix(Xtrain, ytrain), 200, maximize=False)



    preds += np.exp(model.predict(xgb.DMatrix(test1)))/5



    pred = model.predict(xgb.DMatrix(XCV))

    print('logloss :', np.sqrt(metrics.mean_squared_error(pred, yCV)))
submission = pd.concat([test.id,pd.DataFrame(preds)],axis=1)
submission.columns = ['test_id','price']
submission.head()
submission.to_csv('sub_kf_xgb.csv',index=False)