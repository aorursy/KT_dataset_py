import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

from sklearn import datasets, linear_model
df=pd.read_csv('../input/eval-lab-2-f464/train.csv')
df.head()
df.info()
df.isnull().sum()
120-df.astype(bool).sum(axis=0)
df['chem_1'].replace(0, np.nan, inplace = True)

df['chem_2'].replace(0, np.nan, inplace = True)
df.isnull().sum()
df["chem_1"]=df["chem_1"].fillna(df["chem_1"].mean())

df["chem_2"]=df["chem_2"].fillna(df["chem_2"].mean())
df.isnull().sum()

df.astype(bool).sum(axis=0)
df.describe()
df.astype(bool).sum(axis=0)
features=['chem_0','chem_1','chem_2','chem_3','chem_4','chem_5','chem_6','chem_7','attribute']

x=df[features]

y=df['class']
from sklearn.model_selection import train_test_split

X_train,X_val,y_train,y_val = train_test_split(x,y,test_size=0.1,random_state=42)
#GridSearchCV(cv=3, error_score='raise-deprecating',

#                                    colsample_bylevel=1, colsample_bynode=1,

#                                      colsample_bytree=1, gamma=0,

#                                      learning_rate=0.1, max_delta_step=0,

#                                      max_depth=3, min_child_weight=1,

#                                      missing=None, n_estimators=100, n_jobs=1,

#                                      nthread=None, objective='binary:logistic',

#                                      random_state=0, reg_alpha=0, reg_l...

#                                      scale_pos_weight=1, seed=None, silent=None,

#                                      subsample=1, verbosity=1),

#              iid='warn', n_jobs=4,

#              param_grid={'colsample_bytree': [0.3, 0.4, 0.5, 0.7],

#                          'eta': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3],

#                          'gamma': [0.0, 0.1, 0.2, 0.3, 0.4],

#                          'max_depth': [3, 4, 5, 6, 8, 10, 12, 15],

#                          'min_child_weight': [1, 3, 5, 7]},

#              pre_dispatch='2*n_jobs', refit=True, return_train_score=False,

#              scoring='neg_log_loss', verbose=0)

# 
from sklearn import metrics
# from sklearn.linear_model import LogisticRegression

# modellr=LogisticRegression(C=2.0,random_state=0,solver='sag',multi_class='multinomial')

# modellr.fit(X_train,y_train)

# y_pred=modellr.predict(X_val)

# print(metrics.accuracy_score(y_val,y_pred))

# from sklearn.ensemble import RandomForestClassifier

#  

# modelrf = RandomForestClassifier(n_estimators = 19).fit(X_train,y_train)



# from sklearn.metrics import mean_absolute_error



# y_pred_lr = modelrf.predict(X_val)

# print(metrics.accuracy_score(y_val,y_pred_lr))



# mae_lr = mean_absolute_error(y_pred_lr,y_val)





# from sklearn.ensemble import VotingClassifier

# modelvote = VotingClassifier(estimators=[('rf',modelrf),('xg',modelxg)], voting='hard')
# from sklearn.neighbors import KNeighborsClassifier

# model2=KNeighborsClassifier(n_neighbors=7).fit(X_train,y_train)



# from sklearn.metrics import mean_absolute_error



# y_pred_ll = model2.predict(X_val)

# print(metrics.accuracy_score(y_val,y_pred_ll))

# from sklearn.ensemble import VotingClassifier

# modelka = VotingClassifier(estimators=[('kn', model2), ('rf',clf2),('lr',model)], voting='hard').fit(X_train,y_train)



# yy=modelka.predict(X_val)

# from sklearn.metrics import mean_absolute_error

# print(metrics.accuracy_score(y_val,yy))
predict = pd.read_csv('../input/eval-lab-2-f464/test.csv')

# predict = pd.get_dummies(predict, columns=["type"])

X_test_predict = predict[['chem_0','chem_1','chem_2','chem_3','chem_4','chem_5','chem_6','chem_7','attribute']].copy()
from xgboost import XGBClassifier

modelxg = XGBClassifier(base_score=0.5, booster='gbtree',

                                     colsample_bylevel=1, colsample_bynode=1,

                                     colsample_bytree=1, gamma=0,

                                     learning_rate=0.1, max_delta_step=0,

                                     max_depth=3, min_child_weight=1,

                                     missing=None, n_estimators=100, n_jobs=1,

                                     nthread=None, objective='binary:logistic',

                                     random_state=0, reg_alpha=0,

                                     scale_pos_weight=1, seed=None, silent=None,

                                     subsample=1, verbosity=1)

modelxg.fit(X_train, y_train)

y_predxg = modelxg.predict(X_val)

print(metrics.accuracy_score(y_val,y_predxg))
#from sklearn.ensemble import VotingClassifier

# modelvote = VotingClassifier(estimators=[('rf',modelrf),('xg',modelxg)], voting='hard')

# modelvote.fit(X_train, y_train)

# predvote=modelvote.predict(X_val)

# print(metrics.accuracy_score(y_val,predvote))
X_test_predict["chem_1"]=X_test_predict["chem_1"].fillna(df["chem_1"].mean())

X_test_predict["chem_2"]=X_test_predict["chem_2"].fillna(df["chem_2"].mean())
y_pred_lr_test = modelxg.predict(X_test_predict)
predict['class'] = y_pred_lr_test
predict.head()

# print(metrics.accuracy_score(y_val,y_pred_lr_test))
ans = predict[["id","class"]].copy()
ans.to_csv('ans.csv',index=False,encoding ='utf-8' )
from xgboost import XGBClassifier

modelxg = XGBClassifier(base_score=0.5, booster='gbtree',

                                     colsample_bylevel=1, colsample_bynode=1,

                                     colsample_bytree=1, gamma=0,

                                     learning_rate=0.1, max_delta_step=0,

                                     max_depth=3, min_child_weight=1,

                                     missing=None, n_estimators=100, n_jobs=1,

                                     nthread=None, objective='binary:logistic',

                                     random_state=0, reg_alpha=0,

                                     scale_pos_weight=1, seed=None, silent=None,

                                     subsample=1, verbosity=1)

modelxg.fit(X_train, y_train)

#y_predxg = model2.predict(X_val)

#print(metrics.accuracy_score(y_val,y_pred_ll))
