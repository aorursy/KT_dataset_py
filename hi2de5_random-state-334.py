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
train=pd.read_csv("/kaggle/input/cat-in-the-dat-ii/train.csv")

test=pd.read_csv("/kaggle/input/cat-in-the-dat-ii/test.csv")



train.iloc[:,4:22].head()

#test.head()

#train["ord_5"].unique()
train2=train.drop("id",axis=1)

test2=test.drop("id",axis=1)



# ord1_dic

ord1_dic={"Novice":1, "Contributor":2, "Expert":3, "Master":4, "Grandmaster":5}

train2["ord_1"] = train2["ord_1"].replace(ord1_dic)

test2["ord_1"] = test2["ord_1"].replace(ord1_dic)



# ord2_dic

ord2_dic={"Freezing":1, "Cold":2, "Warm":3, "Hot":4, "Lava Hot":5, "Boiling Hot":6}

train2["ord_2"] = train2["ord_2"].replace(ord2_dic)

test2["ord_2"] = test2["ord_2"].replace(ord2_dic)



# apply lc

from sklearn import preprocessing

for c in ["bin_3","bin_4","nom_0","nom_1","nom_2","nom_3","nom_4","nom_5",

          "nom_6","nom_7","nom_8","nom_9","ord_3","ord_4","ord_5"]:

    le = preprocessing.LabelEncoder()

    le.fit(pd.concat([train2[c].fillna("NA"),test2[c].fillna("NA")])) 

    train2[c] = le.transform(train2[c].fillna("NA"))    

    test2[c] = le.transform(test2[c].fillna("NA"))

    

train2
# split to X and y

label="target"

train_y = train2[label]

train_X = train2.drop(label, axis=1)

test_X = test2



train_X.head()

#test2.isnull().sum()
import lightgbm as lgb

lgb=lgb.LGBMClassifier(random_state=334)

gbm=lgb.fit(train_X,train_y)

test_y_lgb=gbm.predict_proba(test_X)[:,1]



import xgboost as xgb

xgb = xgb.XGBClassifier(random_state=334)

gbs = xgb.fit(train_X, train_y)

test_y_xgb = gbs.predict_proba(test_X)[:,1]



w=0.8

test_y=w*test_y_lgb+(1-w)*test_y_xgb

submission = pd.DataFrame({'id': test.id, 'target': test_y})

submission.to_csv('submission_lxgb82_allattrs_ord12.csv', index=False)

submission
#sample=pd.read_csv("/kaggle/input/cat-in-the-dat-ii/sample_submission.csv")

#sample