# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import warnings

warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
df_train = pd.read_csv("../input/train.csv")
df_train.head()
len(df_train)
len(df_train[df_train['target'] == 1])
len(df_train[df_train['target'] == 0])
from sklearn.utils import resample



# Separate majority and minority classes

df_train_1 = df_train[df_train['target']==1]

df_train_0 = df_train[df_train['target']==0]
# Upsample minority class

df_train_1_upsampled = resample(df_train_1, 

                                 replace=True,     # sample with replacement

                                 n_samples=179902,    # to match majority class

                                 random_state=123456) # reproducible results
df_train = pd.concat([df_train_0, df_train_1_upsampled])
len(df_train)
len(df_train[df_train['target'] == 1])
len(df_train[df_train['target'] == 0])
y_train = df_train['target']
train_id = df_train['ID_code']
df_train = df_train.drop(['ID_code','target'], axis=1)
df_train.head()
#from sklearn.ensemble import AdaBoostClassifier

#model = AdaBoostClassifier(n_estimators=300)
# Gradient Boosting 

#from sklearn.ensemble import GradientBoostingClassifier

#model = GradientBoostingClassifier(learning_rate=0.25,n_estimators=300,max_features=150)

# n_estimators = 100 (default)

# loss function = deviance(default) used in Logistic Regression
#from xgboost import XGBClassifier

#model = XGBClassifier(learning_rate=0.35,n_estimators = 300)

#  (default)

#  (default)
from sklearn.linear_model import SGDClassifier

model = SGDClassifier(loss='log', shuffle=True, random_state=101, n_jobs=8, learning_rate='optimal', 

                      alpha=0.1, l1_ratio=0.3)
model.fit(df_train, y_train)
#from scipy.stats import pearsonr

#df_train['var_0'].corr(df_train['var_6'])
df_test = pd.read_csv("../input/test.csv")
df_test.head()
test_id = df_test['ID_code']
df_test = df_test.drop(['ID_code'], axis=1)
df_test.head()
predictions = model.predict(df_test)
test = test_id.to_frame(name='ID_code')
test['target'] = pd.Series(predictions)
test.head()
test.to_csv("submission.csv", columns = test.columns, index=False)