import pandas as pd

import numpy as np

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

import matplotlib.pyplot as plt

%matplotlib inline
df=pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")
df.head()
df.info()
df.describe()
df.sum().unique()
x=df.iloc[:,0:8].values

y=df.iloc[:,-1].values
df.shape
#y.head()

#x.head()
import seaborn as sns

corrmat = df.corr()

top_corr_features=corrmat.index

plt.figure(figsize=(20,20))

g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
import matplotlib.pyplot as plt

import seaborn as sns

plot = sns.pairplot(df, hue='Outcome', diag_kind = 'kde')
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=.3,random_state=0)
#now using random forest algorithm
from xgboost.sklearn import XGBClassifier

import xgboost as xgb

params={

 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,

 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],

 "min_child_weight" : [ 1, 3, 5, 7 ],

 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],

 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]  

}

classifer=xgb.XGBClassifier()

random_search=RandomizedSearchCV(classifer,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)
random_search.fit(x_train,y_train)
random_search.best_estimator_
classifer=xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=0.3, gamma=0.4, gpu_id=-1,

              importance_type='gain', interaction_constraints='',

              learning_rate=0.2, max_delta_step=0, max_depth=15,

              min_child_weight=5, missing='NaN', monotone_constraints='()',

              n_estimators=100, n_jobs=0, num_parallel_tree=1, random_state=0,

              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,

              tree_method='exact', validate_parameters=1, verbosity=None)
y_pred=random_search.predict(x_test)
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred)
from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred)