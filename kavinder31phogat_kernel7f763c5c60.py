# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings('ignore')

from xgboost import XGBClassifier

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train=pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')

df_test=pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')

df_train.head()
features=df_train.columns[2:]

for f in features:

    v= df_train[f].value_counts()

    d=dict(v)

    df_train[f +'count']=df_train[f].map(d)

df_train.head()
from sklearn.model_selection import train_test_split

X = df_train.drop(['ID_code','target'],axis=1)

y = df_train['target']

train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
from datetime import datetime





print(datetime.now())



xgb1 = XGBClassifier(objective = "binary:logistic").fit(train_X,train_y)

print(datetime.now())
y_pred = xgb1.predict_proba(val_X)[:,1]
from sklearn.metrics import roc_auc_score

print("Training score :" + str(roc_auc_score(train_y,xgb1.predict(train_X))))

print("validation score :" + str(roc_auc_score(val_y,y_pred)))

 # plot feature importance

from xgboost import plot_importance

plt.rcParams["figure.figsize"] = (20,40)

plot_importance(xgb1,importance_type='gain')

plt.show()
feature_important=xgb1.get_booster().get_score(importance_type='gain')

feature_important
keys = list(feature_important.keys())

values = list(feature_important.values())

data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)

data
a=list((data[data['score']>60]).index)

print(len(a))

a
train_X2=train_X[a]

val_X2=val_X[a]
from datetime import datetime





print(datetime.now())



xgb2 = XGBClassifier(objective = "binary:logistic",

                    subsample= 0.5, 

                    reg_lambda= 1, 

                    reg_alpha= 0, 

                    n_estimators= 2500, 

                    min_child_weight=12, 

                    max_depth= 10, 

                    learning_rate= 0.02, 

                    gamma= 0.3,

                    colsample_bytree= 0.5,

                    eval_metric ="auc").fit(train_X2,train_y)

print(datetime.now())
y_pred2 = xgb2.predict_proba(val_X2)[:,1]

from sklearn.metrics import roc_auc_score

print("Training score :" + str(roc_auc_score(train_y,xgb2.predict(train_X2))))

print("validation score :" + str(roc_auc_score(val_y,y_pred2)))

features1=df_test.columns[1:]

for f in features1:

    v= df_test[f].value_counts()

    d=dict(v)

    df_test[f +'count']=df_test[f].map(d)

df_test.head()
test_X = df_test.drop(['ID_code'],axis=1)

test_X = test_X[a]

Test_Prediction = xgb2.predict_proba(test_X)[:,1]



sub_df = pd.DataFrame({"ID_code":df_test["ID_code"].values})

sub_df["target"] = Test_Prediction

sub_df.to_csv("submission_final4.csv", index=False)
