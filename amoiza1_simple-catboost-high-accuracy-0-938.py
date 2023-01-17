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
import numpy as np

import pandas as pd

from catboost import cv, Pool
#read data

train = pd.read_csv('../input/data-without-drift/train_clean.csv', dtype={'time':np.float32,'signal':np.float32,'open_channels':np.int32})

test = pd.read_csv('../input/data-without-drift/test_clean.csv', dtype={'time':np.float32,'signal':np.float32})

sub = pd.read_csv('../input/liverpool-ion-switching/sample_submission.csv', dtype={'time':np.float32,'open_channels':np.int32})

Y_train_proba = np.load("../input/ion-shifted-rfc-proba/Y_train_proba.npy")

Y_test_proba = np.load("../input/ion-shifted-rfc-proba/Y_test_proba.npy")



for i in range(11):

  train[f"proba_{i}"] = Y_train_proba[:,i]

  test[f"proba_{i}"] = Y_test_proba[:,i]



#Lets also get hard votes from "ion-shifted-rfc-proba" data because this categorical information might be useful in CatboostClassifier

train['proba'] = np.argmax(Y_train_proba,axis=1)

test['proba'] = np.argmax(Y_test_proba,axis=1)
train.loc[train.index % 2 ==0,'id'] = 1

train.loc[train.index % 2 !=0,'id'] = 2



x_train = train[train.id ==1].drop(['open_channels','id'],axis=1)

y_train = train[train.id ==1]['open_channels']



x_val = train[train.id ==2].drop(['open_channels','id'],axis=1)

y_val = train[train.id ==2]['open_channels']



print(x_train.shape)

print(y_train.shape)

print(x_val.shape)

print(y_val.shape)
# conda install -c conda-forge catboost
from catboost import CatBoostClassifier



cat = CatBoostClassifier(task_type='GPU',iterations=10000,loss_function='MultiClass',od_type='Iter',od_wait=500)



cat.fit(x_train,y_train,verbose=25, eval_set = (x_val,y_val),cat_features=[13])

y_preds = cat.predict(test)

sub['open_channels'] = y_preds

sub.to_csv('submission_cat.csv',index=False,float_format='%.4f')