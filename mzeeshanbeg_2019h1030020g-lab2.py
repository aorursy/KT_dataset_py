#import the libs

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

df = pd.read_csv('../input/eval-lab-2-f464/train.csv')
df.info()
df.describe()
df.head(10)
df.dtypes
df.isnull().head(10)
# missing_count = df.isnull().sum()

# missing_count[missing_count > 0]

num_features = df.select_dtypes(include=[np.number])

corr = num_features.corr()

corr['class'].sort_values(ascending= False)
#replace 0 values with means



# df['chem_5'].replace(0,np.nan,inplace= True)

# df['chem_6'].replace(0,np.nan,inplace= True)

# df["chem_1"].fillna(value=df.mean(),inplace=True)



# df.fillna(value=df.mean(),inplace=True)

df['chem_1'].replace(0, np.nan, inplace = True)

df['chem_2'].replace(0, np.nan, inplace = True)

df["chem_1"]=df["chem_1"].fillna(df["chem_1"].mean())

df["chem_2"]=df["chem_2"].fillna(df["chem_2"].mean())
df.isnull().any()
df.describe()
df.corr()
import xgboost as xgb
X = df[["chem_0","chem_1","chem_2","chem_3","chem_4","chem_5","chem_6","chem_7","attribute"]].copy()

y = df["class"].copy()

y.head()
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

from sklearn.metrics import accuracy_score



X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.05,random_state=42)
X_train.info()

X_val.info()
# from sklearn.model_selection import GridSearchCV



# clf = xgb.XGBClassifier()

# parameters = {

#      "eta"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,

#      "max_depth"        : [ 1,2,3, 4, 5, 6, 8, 10, 12, 15,50,55, 100, 110],

#      "min_child_weight" : [ 1, 3, 5, 7, 30, 35 ],

#      "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4, 4.0,5.0 ],

#      "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ],

#      "booster" :['gbtree','gblinear','dart'],

#      "n_estimators"     :[10, 15 , 30, 45, 80, 160 ,300]

#      }



# grid = GridSearchCV(clf,

#                     parameters, n_jobs=4,

#                     scoring="neg_log_loss",

#                     cv=3)



# grid.fit(X_train, y_train)

# clf.fit(X_train,y_train)

# ypred = clf.predict(X_val)

# print(metrics.accuracy_score(y_val,ypred))
#this is final

#-----------------------------------------



from xgboost import XGBClassifier

clf = XGBClassifier(base_score=0.5, booster='gbtree',

                                     colsample_bylevel=1, colsample_bynode=1,

                                     colsample_bytree=1, gamma=0,

                                     learning_rate=0.1, max_delta_step=0,

                                     max_depth=3, min_child_weight=1,

                                     missing=None, n_estimators=100, n_jobs=1,

                                     nthread=None, objective='binary:logistic',

                                     random_state=0, reg_alpha=0,

                                     scale_pos_weight=1, seed=None, silent=None,

                                     subsample=1, verbosity=1)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)

print(metrics.accuracy_score(y_val,y_pred))
predict = pd.read_csv('../input/eval-lab-2-f464/test.csv')

predict.isnull().any()
X_test_predict = predict[["chem_0","chem_1","chem_2","chem_3","chem_4","chem_5","chem_6","chem_7","attribute"]].copy()
y_pred_lr_test = clf.predict(X_test_predict)
predict['class'] = y_pred_lr_test
predict.head()
ans = predict[["id","class"]].copy()

ans.head(20)
ans.to_csv('ans.csv',index=False,encoding ='utf-8' )