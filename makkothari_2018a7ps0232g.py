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
df = pd.read_csv("/kaggle/input/minor-project-2020/train.csv")

df2 = pd.read_csv("/kaggle/input/minor-project-2020/test.csv")
!pip install xgboost==0.90
!pip freeze
import xgboost

#!pip install xgboost==0.90

print(xgboost.__version__)
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SVMSMOTE

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import StandardScaler

#import xgboost as xgb

#import xgboost

xgboost.__version__
X_train = df.iloc[ : , 1:-1].values

y_train = df.iloc [ : , -1].values



X_test = df2.iloc [ : , 1:].values

one = 0

zero = 0

X_zeros = []

X_ones = []

for i in range(len(X_train)):

  if (y_train[i] == 0):

    zero += 1

    X_zeros.append(i);

  else:

    one += 1

    X_ones.append(i)

print(one, zero)
zero/one
t = 0

for i in range(531):  



  xgb_model = xgboost.XGBClassifier(objective="binary:logistic", random_state=42)

  X_rand = np.append(X_zeros[0+t:1501+t], X_ones)  

  y_rand = np.append(np.zeros(1501),np.ones(1501))



  X_cust = X_train[X_rand, : ]

  y_cust = y_rand

  

  xgb_model.fit(X_cust, y_cust)



  pred = xgb_model.predict_proba(X_test)[ : , 1]

  pred = pred.reshape(1, -1)    

  if (i == 0):

    preds = pred

  else:

    preds = np.append(preds, pred, axis = 0)

    

  t += 1501

  print(i)  

ans = np.mean(preds, axis = 0)

ans.shape
df_res = pd.DataFrame(ans, index = None, columns = ['target'])

df_res_fin = pd.concat([df2.iloc[ : , 0], df_res], axis = 1, ignore_index = True)

df_res_fin.columns = ['id','target']

df_res_fin.to_csv ('submission2.csv', index = False, header=True)