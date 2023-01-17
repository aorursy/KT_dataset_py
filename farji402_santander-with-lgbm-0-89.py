# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Import libraries

import pandas as pd

import numpy as np

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

import lightgbm as lgb

import gc

gc.enable()
# Import data

df_train = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')

df_test = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')
df_train.head()
# Cross validation splits

skfold = StratifiedKFold(n_splits= 5, shuffle= True, random_state= True)

splits = skfold.split(df_train.drop(['ID_code', 'target'], axis= 1), df_train['target'])
# Training

cv_val = np.zeros(df_train.shape[0])



for counter, idx in enumerate(splits):

    

    X_train, y_train = df_train.drop(['ID_code', 'target'], axis= 1).values[idx[0]],\

    df_train['target'].values[idx[0]]

    X_val, y_val = df_train.drop(['ID_code', 'target'], axis= 1).values[idx[1]],\

    df_train['target'].values[idx[1]]

    

    lgb_model = lgb.LGBMClassifier(max_depth= -1,

                                   n_estimators= 999999,

                                   learning_rate = 0.02,

                                   colsample_bytree= 0.3,

                                   num_leaves= 2,

                                   metric= 'auc',

                                   objective= 'binary',

                                   device= 'gpu',

                                   gpu_platform_id= 0,

                                   gpu_device_id= 0

                                  )

    

    lgb_model.fit(X_train, y_train, eval_set= [(X_val, y_val)], early_stopping_rounds= 1000)

    

    cv_val[idx[1]] += lgb_model.predict_proba(X_val)[:,1]

    

    del X_train, X_val, y_train, y_val

    gc.collect()

    

    save_to = 'Fold{}.txt'.format(counter+1)

    lgb_model.booster_.save_model(save_to)
# Evaluation

lgb_auc = round(roc_auc_score(df_train['target'], cv_val), 4)

print('AUC score with LGBM: {}'.format(lgb_auc))
# Predicting

lgb_result = np.zeros(df_test.shape[0])



for i in np.arange(1,6,1):

    model = lgb.Booster(model_file='Fold{}.txt'.format(i))

    

    lgb_result += model.predict(df_test.drop(['ID_code'], axis= 1))

    

lgb_result /= 5
submission = pd.DataFrame({

    'ID_code': df_test['ID_code'],

    'target': lgb_result

})



submission.to_csv('submission.csv', index= False)
submission.shape