import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



%matplotlib inline
data = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")

data.shape
data.head(5)
data.info()
data.isnull().sum()
import seaborn as sns

corr_ds = data.corr()

top_corr = corr_ds.index

plt.figure(figsize=(20,20))

g = sns.heatmap(data[top_corr].corr(), annot = True)
data.corr()
sns.countplot(data['Outcome'])
X = data.drop(['Outcome'], axis = 1)

y = data['Outcome']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)
from sklearn.metrics import confusion_matrix, accuracy_score
import xgboost

from sklearn.model_selection import RandomizedSearchCV



xgb_model = xgboost.XGBClassifier()
param = {

    'learning_rate':[0.05,0.1,0.15,0.2,0.25,0.3],

    'max_depth':[3,4,5,6,8,10,12],

    'min_child_weight':[1,3,5,7],

    'gamma':[0.0,0.1,0.2,0.3,0.4],

    'colsample_bytree':[0.3,0.4,0.5,0.7]

}
random_search = RandomizedSearchCV(xgb_model, param_distributions = param, n_iter = 5,

                                     scoring = 'roc_auc', n_jobs = -1, cv = 5, verbose = 3)

random_search.fit(X_train,y_train)
random_search.best_estimator_
xgb_model = xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=0.4, gamma=0.1, gpu_id=-1,

              importance_type='gain', interaction_constraints='',

              learning_rate=0.05, max_delta_step=0, max_depth=12,

              min_child_weight=5, monotone_constraints='()',

              n_estimators=100, n_jobs=0, num_parallel_tree=1, random_state=0,

              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,

              tree_method='exact', validate_parameters=1, verbosity=None)
xgb_model.fit(X_train,y_train)
pred_xgb = xgb_model.predict(X_test)



acc_xgb = accuracy_score(y_test,pred_xgb)

print("Accuracy XGB:", acc_xgb)
cm_xgb = confusion_matrix(y_test,pred_xgb)

sns.heatmap(cm_xgb, annot=True)
from sklearn.svm import SVC

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler



svc_model = make_pipeline(StandardScaler(), SVC(gamma='auto'))



svc_model.fit(X_train, y_train)
pred_svc = svc_model.predict(X_test)



acc_svc = accuracy_score(y_test,pred_svc)

print("Accuracy SVC:", acc_svc)
cm_svc = confusion_matrix(y_test,pred_svc)

sns.heatmap(cm_svc, annot=True)