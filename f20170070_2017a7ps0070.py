import warnings

warnings.filterwarnings("ignore")

import numpy as np

import pandas as pd

import sklearn

from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns

import xgboost as xgb
df = pd.read_csv('train.csv')

df.info()
df.head()




missing_count = df.isnull().sum()

missing_count[missing_count > 0]
df.isnull().any().any()
sns.heatmap(df.corr())
X = df.drop(['class','id'],axis=1)

y = df['class']



print (X.shape, y.shape)
# from sklearn import preprocessing



# min_max_scaler = preprocessing.MinMaxScaler()



# X_minmax = min_max_scaler.fit_transform(X)

# X_minmax
df.corr()
from sklearn.model_selection import train_test_split



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1)
from sklearn.tree import DecisionTreeClassifier

# clf = DecisionTreeClassifier().fit(X_train,y_train)
from sklearn.metrics import accuracy_score

# y_pred = clf.predict(X_val)

# accuracy = accuracy_score(y_val,y_pred)



# print(accuracy)
#GradientBoostingClassifier

from sklearn.ensemble import GradientBoostingClassifier



# gb_clf = GradientBoostingClassifier().fit(X_train,y_train)

# y_pred_gb = gb_clf.predict(X_val)

# gb_acc = accuracy_score(y_val,y_pred_gb)



# print(gb_acc)
#XGBClassifier

from xgboost import XGBClassifier



xgb_clf = XGBClassifier().fit(X_train,y_train)

y_pred_xgb = xgb_clf.predict(X_val)

xgb_acc = accuracy_score(y_val,y_pred_xgb)



print(xgb_acc)
# n = len(X_train)

# X_A = X_train[:n//2]

# y_A = y_train[:n//2]

# X_B = X_train[n//2:]

# y_B = y_train[n//2:]
#RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier

# clf_1 = DecisionTreeClassifier().fit(X_A, y_A)

# y_pred_1 = clf_1.predict(X_B)

# clf_2 = RandomForestClassifier(n_estimators=100).fit(X_A, y_A)

# y_pred_2 = clf_2.predict(X_B)

# clf_3 = GradientBoostingClassifier().fit(X_A, y_A)

# y_pred_3 = clf_3.predict(X_B)
# X_C = pd.DataFrame({'RandomForest': y_pred_2, 'DeccisionTrees': y_pred_1, 'GradientBoost': y_pred_3})

# y_C = y_B

# X_C.head()
# X_D = pd.DataFrame({'RandomForest': clf_2.predict(X_val), 'DeccisionTrees': clf_1.predict(X_val), 'GradientBoost': clf_3.predict(X_val)})

# y_D = y_val
from xgboost import XGBClassifier



# xgb_clf = XGBClassifier().fit(X_C,y_C)

# y_pred_xgb = xgb_clf.predict(X_D)

# xgb_acc = accuracy_score(y_D,y_pred_xgb)



# print(xgb_acc)
# #BaggingClassifier

# from sklearn.ensemble import BaggingClassifier



# from sklearn.ensemble import VotingClassifier



# estimators = [('rf', RandomForestClassifier()), ('bag', BaggingClassifier()), ('xgb', XGBClassifier())]



# soft_voter = VotingClassifier(estimators=estimators, voting='soft').fit(X_train,y_train)

# hard_voter = VotingClassifier(estimators=estimators, voting='hard').fit(X_train,y_train)
# soft_acc = accuracy_score(y_val,soft_voter.predict(X_val))

# hard_acc = accuracy_score(y_val,hard_voter.predict(X_val))



# print("Acc of soft voting classifier:{}".format(soft_acc))

# print("Acc of hard voting classifier:{}".format(hard_acc))
# from sklearn.model_selection import GridSearchCV

# from sklearn.metrics import make_scorer

# from sklearn.model_selection import StratifiedKFold



# parameters = {'learning_rate': [0.01], #so called `eta` value

#               'max_depth': range(3,10,1),

#               'min_child_weight': range(1,6,2),

#               'silent': [1],

#               'subsample': [0.9],

#               'colsample_bytree': [0.5],

#               'n_estimators': [400], #number of trees, change it to 1000 for better results

#               'missing':[-500],

#               'seed': [1660]}#Dictionary of parameters

# scorer = make_scorer(accuracy_score)         #Initialize the scorer using make_scorer

# grid_obj = GridSearchCV(xgb_clf,parameters,n_jobs=5,scoring=scorer,)         #Initialize a GridSearchCV object with above parameters,scorer and classifier

# grid_fit = grid_obj.fit(X_train,y_train)        #Fit the gridsearch object with X_train,y_train

# best_clf_sv = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object

# unoptimized_predictions = xgb_clf.predict(X_val)      #Using the unoptimized classifiers, generate predictions

# optimized_predictions = best_clf_sv.predict(X_val)        #Same, but use the best estimator



# acc_unop = accuracy_score(y_val, unoptimized_predictions)*100       #Calculate accuracy for unoptimized model

# acc_op = accuracy_score(y_val, optimized_predictions)*100         #Calculate accuracy for optimized model



# print("Accuracy score on unoptimized model:{}".format(acc_unop))

# print("Accuracy score on optimized model:{}".format(acc_op))
scorer = make_scorer(accuracy_score)
param_test1 = {

 'max_depth':range(3,10,2),

 'min_child_weight':range(1,10,2)

}

gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140,

gamma=0, subsample=0.8, colsample_bytree=0.8,

objective= 'multi:softmax', nthread=4, scale_pos_weight=1, seed=27), 

 param_grid = param_test1, scoring=scorer,n_jobs=4,iid=False, cv=5)

gsearch1.fit(X,y)

gsearch1.best_params_, gsearch1.best_score_
param_test1 = {

 'max_depth':[8,9,10],

 'min_child_weight':[1,2,3]

}

gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=9,

min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,

objective= 'multi:softmax', nthread=4, scale_pos_weight=1, seed=27), 

 param_grid = param_test1, scoring=scorer,n_jobs=4,iid=False, cv=5)

gsearch1.fit(X,y)

gsearch1.best_params_, gsearch1.best_score_
# param_test2b = {

#  'min_child_weight':[1,2]

# }

# gsearch2b = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140,

# gamma=0, subsample=0.8, colsample_bytree=0.8,

#  nthread=4, scale_pos_weight=1, seed=27), 

#  param_grid = param_test2b, scoring=scorer,n_jobs=4,iid=False, cv=5)

# gsearch2b.fit(X_val,y_val)

# gsearch2b.best_params_, gsearch2b.best_score_
param_test3 = {

 'gamma':[i/10.0 for i in range(0,5)]

}

gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=9,

 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,

 nthread=4, scale_pos_weight=1, seed=27), 

 param_grid = param_test3, scoring=scorer,n_jobs=4,iid=False, cv=5)

gsearch3.fit(X,y)

gsearch3.best_params_, gsearch3.best_score_
xgb2 = XGBClassifier(

 learning_rate =0.1,

 n_estimators=1000,

 max_depth=9,

 min_child_weight=1,

 gamma=0.0,

 subsample=0.8,

 colsample_bytree=0.8,

 nthread=4,

 scale_pos_weight=1,

 seed=27)
param_test4 = {

 'subsample':[i/10.0 for i in range(6,10)],

 'colsample_bytree':[i/10.0 for i in range(6,10)]

}

gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=9,

 min_child_weight=1, gamma=0.0, subsample=0.8, colsample_bytree=0.8,

 objective= 'multi:softmax', nthread=4, scale_pos_weight=1,seed=27), 

 param_grid = param_test4, scoring=scorer,n_jobs=4,iid=False, cv=5)

gsearch4.fit(X,y)

gsearch4.best_params_, gsearch4.best_score_
param_test5 = {

 'subsample':[i/100.0 for i in range(85,100,5)],

 'colsample_bytree':[i/100.0 for i in range(55,70,5)]

}

gsearch5 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=9,

 min_child_weight=1, gamma=0.0, subsample=0.8, colsample_bytree=0.8,

 objective= 'multi:softmax', nthread=4, scale_pos_weight=1,seed=27), 

 param_grid = param_test5, scoring=scorer,n_jobs=4,iid=False, cv=5)

gsearch5.fit(X,y)

gsearch5.best_params_, gsearch5.best_score_
param_test6 = {

 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]

}

gsearch6 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=9,

 min_child_weight=1, gamma=0.0, subsample=0.95, colsample_bytree=0.55,

 objective= 'multi:softmax', nthread=4, scale_pos_weight=1,seed=27), 

 param_grid = param_test6, scoring=scorer,n_jobs=4,iid=False, cv=5)

gsearch6.fit(X,y)

gsearch6.best_params_, gsearch6.best_score_
param_test6 = {

 'reg_alpha':[0.1,0.05,0.5]

}

gsearch6 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=9,

 min_child_weight=1, gamma=0.0, subsample=0.95, colsample_bytree=0.55,

 objective= 'multi:softmax', nthread=4, scale_pos_weight=1,seed=27), 

 param_grid = param_test6, scoring=scorer,n_jobs=4,iid=False, cv=5)

gsearch6.fit(X,y)

gsearch6.best_params_, gsearch6.best_score_
xgb3 = XGBClassifier(

 learning_rate =0.01,

 n_estimators=1000,

 max_depth=9,

 min_child_weight=1,

 gamma=0.0,

 subsample=0.95,

 colsample_bytree=0.55,

 reg_alpha=0.1,

 objective= 'multi:softmax',

 nthread=2,

 scale_pos_weight=1,

 seed=27)
d = pd.read_csv('test.csv')

d.info()
d.isnull().any().any()
X_val = d.drop(['id'],axis=1)



# min_max_scaler = preprocessing.MinMaxScaler()



# X_val_minmax = min_max_scaler.fit_transform(X_val)



# print (X_val_minmax.shape)
X_val.head()
# X_val_minmax
# y_pred=xgb_clf.predict(X_val)
y_pred= xgb3.fit(X,y).predict(X_val)
y_pred
submission = pd.DataFrame({'id':d['id'],'class':y_pred})

submission.to_csv('long4.csv',index=False)