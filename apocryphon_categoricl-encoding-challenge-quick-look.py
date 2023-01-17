# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import xgboost as xgb

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OrdinalEncoder



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def is_null(df):

    dic={}

    dic['count'] = df.isnull().sum()

    dic['fraction'] = df.isnull().sum()/len(train)

    return pd.DataFrame(dic)
train = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv')

test = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv')
train.shape
train.head()
train.info()
train.describe()
from sklearn.preprocessing import OrdinalEncoder

enc = OrdinalEncoder()
train.bin_0.value_counts()
train.bin_2.value_counts()
train.bin_3.value_counts()
train.bin_4.value_counts()
train.bin_1.value_counts()
train.ord_0.value_counts()
train.ord_1.value_counts()
train.ord_2.value_counts()
train.ord_3.value_counts()
train.ord_4.value_counts()
train.ord_5.value_counts()
is_null(train)
#bin_3 ---- 'Y', 'N'

#bin_4 ---- 'Y', 'N'

#ord_1 ---- 'Novice', 'Expert', 'Contributor' , 'Master','Grandmaster'

#ord_2 ---- 'Freezing', 'Warm', 'Cold', 'Boiling Hot', 'Hot', 'Lava Hot'

#ord_3 ---- 'n', 'a','m','c','h','o','b','e','k','i','d','f','g','j','l'

#ord_4 ----'N' ,'P' ,'Y'  ,'A'  ,'R'  ,'U' ,'M'  ,'X'   ,'C'   ,'H'  ,'Q'   ,'T'  ,'O'  ,'B'  ,'E'  ,'K'  ,'I' ,'D'  ,'F'  ,'W' ,'Z'  ,'S' ,'G','V','J','L'

#ord_5 ----- 'Fl','DN','Sz' ........
test.shape
test.info()
test.describe()
Y = train['target']

train =train.astype('category')

train['target'] = Y



test["target"] = -1

data = pd.concat([train, test]).reset_index(drop=True)



features = [x for x in train.columns if x not in ["id", "target",'ord_1','ord_2']]



for feat in features:

    lbl_enc = LabelEncoder()

    data[feat] = lbl_enc.fit_transform(data[feat].fillna("-1").astype(str))

    



o_enc = OrdinalEncoder()

data[['ord_1','ord_2']] = o_enc.fit_transform(data[['ord_1','ord_2']].fillna("-1").astype(str))



test = data[data.target == -1]

train = data[data.target != -1]
target = train['target']

train.drop(['id','target'],axis = 1,inplace = True)



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(train,target,train_size = .85,random_state = 42,stratify = Y)
xg_reg1 = xgb.XGBClassifier( objective='reg:logistic',learning_rate = 0.1,

                max_depth = 20, reg_alpha = 15, n_estimators = 1000,booster = 'dart',

            tree_method = 'hist')

xg_reg1.fit(x_train,y_train,eval_set= [(x_train,y_train), (x_test, y_test)],

    eval_metric="auc",early_stopping_rounds=5,verbose=True)

xg_reg1.save_model('xgb.model')
key = test['id'].reset_index(drop = True)

test.drop(['id','target'],axis = 1,inplace = True)
key.head()
pred = pd.Series(xg_reg1.predict_proba(test)[:,-1])



sub = pd.concat([key,pred],axis = 1,ignore_index = True)



sub.columns = ['id','target']

 

sub.to_csv('submission.csv',index = False)