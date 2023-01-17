import pandas as pd

import numpy as np



pd.pandas.set_option('display.max_columns',None)

pd.pandas.set_option('display.max_rows',None)



train = pd.read_csv('../input/covid19-symptoms-checker/Cleaned-Data.csv')

train.head()
data = train.copy()

data = data.drop(['Severity_None','None_Sympton','None_Experiencing','Contact_Dont-Know','Country','Contact_No'],axis = 1)

data.head()
data1 = data.copy()

data1 = data.drop(['Severity_Moderate','Severity_Mild'],axis = 1)

y_data = data1['Severity_Severe']

x_data = data1.drop(['Severity_Severe'],axis = 1)

SEED = 42

from sklearn.model_selection import train_test_split

X_train,X_val,Y_train,Y_val = train_test_split(x_data,y_data,test_size = 0.3,random_state = SEED)
X_train.head()
Y_train.head()


from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

rf = RandomForestClassifier()

params = {"max_depth":[15,20,25], "n_estimators":[27,30,33]}

rf_reg = GridSearchCV(rf, params, cv = 10, n_jobs =10)

rf_reg.fit(X_train, Y_train)

print(rf_reg.best_estimator_)

best_estimator=rf_reg.best_estimator_

y_pred_train = best_estimator.predict(X_train)

y_pred_val = best_estimator.predict(X_val)

# rf.fit(X_train,Y_train)
from sklearn.metrics import confusion_matrix

confusion_matrix(Y_val,y_pred_val)
scoring = 'accuracy'

score = cross_val_score(rf,X_val,Y_val,cv = k_fold,n_jobs=1,scoring=scoring)

print(score)

type(score)
score.mean()
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

import xgboost 
params={'min_child_weight':[4,5], 'gamma':[i/10.0 for i in range(3,6)],  'subsample':[i/10.0 for i in range(6,11)],

'colsample_bytree':[i/10.0 for i in range(6,11)], 'max_depth': [2,3,4]}
Clf = xgboost.XGBClassifier()
# random_search = RandomizedSearchCV(regressor,param_distributions = params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)

random_search = RandomizedSearchCV(Clf

                                   , params,n_iter=5, n_jobs=1, cv=5)
random_search.fit(X_train,Y_train)

random_search.best_estimator_

xgbo = xgboost.XGBClassifier(base_score=0.5, booster=None, colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=1.0, gamma=0.3, gpu_id=-1,

              importance_type='gain', interaction_constraints=None,

              learning_rate=0.300000012, max_delta_step=0, max_depth=2,

              min_child_weight=5,monotone_constraints=None,

              n_estimators=100, n_jobs=0, num_parallel_tree=1,

              objective='binary:logistic', random_state=0, reg_alpha=0,

              reg_lambda=1, scale_pos_weight=1, subsample=0.9, tree_method=None,

              validate_parameters=False, verbosity=None)
xgbo.fit(X_train,Y_train)
pred_xgb = xgbo.predict(X_val) 

confusion_matrix(Y_val,pred_xgb)
# from sklearn.model_selection import KFold



# ntrain = X_train.shape[0]

# ntest = X_val.shape[0]

# NFOLDS = 5

# kf = KFold(n_splits = NFOLDS, random_state = SEED)







# class SklearnHelper(object):

#     def __init__(self, clf, seed=0, params=None):

#         params['random_state'] = seed

#         self.clf = clf(**params)



#     def train(self, x_train, y_train):

#         self.clf.fit(x_train, y_train)



#     def predict(self, x):

#         return self.clf.predict(x)

    

#     def fit(self,x,y):

#         return self.clf.fit(x,y)

    

#     def feature_importances(self,x,y):

#         print(self.clf.fit(x,y).feature_importances_)

    



# def get_oof(clf, x_train, y_train, x_test):

#     oof_train = np.zeros((ntrain,))

#     oof_test = np.zeros((ntest,))

#     oof_test_skf = np.empty((NFOLDS,ntest))

    

#     for i, (train_index, test_index) in enumerate(kf.split(x_train)):

#         x_tr = x_train[train_index]

#         x_te = x_train[test_index]

#         y_tr = y_train[train_index]

        

#         clf.train(x_tr,y_tr)

#         oof_train[test_index] = clf.predict(x_te)

#         oof_test_skf[i,:] = clf.predict(x_test)

        

#     oof_test[:] = oof_test_skf.mean(axis=0)

#     return oof_train.reshape(-1,1),oof_test.reshape(-1,1)

        
# from sklearn.ensemble import RandomForestClassifier

# from sklearn.ensemble import AdaBoostClassifier

# from sklearn.ensemble import GradientBoostingClassifier

# from sklearn.ensemble import ExtraTreesClassifier

# from sklearn.svm import SVC
# #Random Forest Parameters

# rf_params = {

#     'n_jobs': -1,

#     'n_estimators': 500,

#      'warm_start': True, 

#      #'max_features': 0.2,

#     'max_depth': 6,

#     'min_samples_leaf': 2,

#     'max_features' : 'sqrt',

#     'verbose': 0

# }

# #extraTree Parameters

# et_params = {

#     'n_jobs': -1,

#     'n_estimators':500,

#     #'max_features': 0.5,

#     'max_depth': 8,

#     'min_samples_leaf': 2,

#     'verbose': 0

# }



# # AdaBoost parameters

# ada_params = {

#     'n_estimators': 500,

#     'learning_rate' : 0.75

# }



# # Gradient Boosting parameters

# gb_params = {

#     'n_estimators': 500,

#      #'max_features': 0.2,

#     'max_depth': 5,

#     'min_samples_leaf': 2,

#     'verbose': 0

# }



# # Support Vector Classifier parameters 

# svc_params = {

#     'kernel' : 'linear',

#     'C' : 0.025

#     }
# rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)

# et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)

# ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)

# gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)

# svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)

# # #X['Survived'].ravel()

# # y_train = y

# # train = X

# # x_train = train.values # Creates an array of the train data

# # x_test = X1.values #





# y_train = y_data

# X = X_train

# X_train_values = X.values

# X_val_values = X_val.values
# et_oof_train, et_oof_test = get_oof(et, X_train_values, y_train, X_val_values) # Extra Trees

# rf_oof_train, rf_oof_test = get_oof(rf,X_train_values, y_train, X_val_values) # Random Forest

# ada_oof_train, ada_oof_test = get_oof(ada, X_train_values, y_train, X_val_values) # AdaBoost 

# gb_oof_train, gb_oof_test = get_oof(gb,X_train_values, y_train, X_val_values) # Gradient Boost

# svc_oof_train, svc_oof_test = get_oof(svc,X_train_values, y_train, X_val_values) # Support Vector Classifier



# print("Training is complete")
# rf_feature = rf.feature_importances(x_train,y_train)

# et_feature = et.feature_importances(x_train, y_train)

# ada_feature = ada.feature_importances(x_train, y_train)

# gb_feature = gb.feature_importances(x_train,y_train)
# rf_features = [0.3116732 , 0.56566957, 0.05057659, 0.05226303, 0.01981761]

# et_features = [0.26047002, 0.64797022, 0.03636626, 0.03371078, 0.02148271]

# ada_features = [0.29  ,0.106 ,0.076 ,0.066 ,0.102]

# gb_features = [0.24384334 ,0.66760243 ,0.03514145 ,0.0271241 , 0.02628868]