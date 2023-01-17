# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv('/kaggle/input/minor-project-2020/train.csv')

testf=pd.read_csv('/kaggle/input/minor-project-2020/test.csv')

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df1=df.drop(['id'],axis=1)

df1=df1.drop_duplicates()
y=df1['target']

X=df1.drop(['target'],axis=1)
from imblearn.over_sampling import SMOTE

oversample = SMOTE()

X_trainres,y_trainres = oversample.fit_resample(X_trainres, y_trainres)


X_train=X_trainres

y_train=y_trainres


from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.cluster import KMeans

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import precision_recall_fscore_support

from sklearn.metrics import mean_squared_error as mse

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_validate

from sklearn.model_selection import train_test_split

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.feature_selection import SelectFromModel

from sklearn.svm import LinearSVC

from sklearn.ensemble import VotingClassifier

from sklearn.feature_selection import RFECV

from sklearn.ensemble import AdaBoostClassifier

from sklearn import model_selection

import xgboost as xgb

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV

from sklearn.ensemble import StackingClassifier

from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import RandomUnderSampler

from sklearn.model_selection import cross_val_score,KFold, RandomizedSearchCV

from scipy.stats import uniform, randint

from sklearn.ensemble import IsolationForest

from imblearn.pipeline import make_pipeline

from imblearn.under_sampling import NearMiss

sc = StandardScaler()

from sklearn.model_selection import RepeatedStratifiedKFold
# decision_tree = DecisionTreeClassifier()



# scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

# scores = cross_validate(decision_tree, X_train, y_train, scoring=scoring, cv=20)



# sorted(scores.keys())

# dtree_fit_time = scores['fit_time'].mean()

# dtree_score_time = scores['score_time'].mean()

# dtree_accuracy = scores['test_accuracy'].mean()

# dtree_precision = scores['test_precision_macro'].mean()

# dtree_recall = scores['test_recall_macro'].mean()

# dtree_f1 = scores['test_f1_weighted'].mean()

# dtree_roc = scores['test_roc_auc'].mean()
# decision_tree.fit(X_train,y_train)

# roc_auc_score(y_test,decision_tree.predict(X_test))
# SVM = SVC(probability = True)



# scoring = ['accuracy','precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

# scores = cross_validate(SVM, X_train, y_train, scoring=scoring, cv=20)



# sorted(scores.keys())

# SVM_fit_time = scores['fit_time'].mean()

# SVM_score_time = scores['score_time'].mean()

# SVM_accuracy = scores['test_accuracy'].mean()

# SVM_precision = scores['test_precision_macro'].mean()

# SVM_recall = scores['test_recall_macro'].mean()

# SVM_f1 = scores['test_f1_weighted'].mean()

# SVM_roc = scores['test_roc_auc'].mean()
# LDA = LinearDiscriminantAnalysis()



# scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

# scores = cross_validate(LDA, X_train, y_train, scoring=scoring, cv=20)



# sorted(scores.keys())

# LDA_fit_time = scores['fit_time'].mean()

# LDA_score_time = scores['score_time'].mean()

# LDA_accuracy = scores['test_accuracy'].mean()

# LDA_precision = scores['test_precision_macro'].mean()

# LDA_recall = scores['test_recall_macro'].mean()

# LDA_f1 = scores['test_f1_weighted'].mean()

# LDA_roc = scores['test_roc_auc'].mean()
# QDA = QuadraticDiscriminantAnalysis()



# scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

# scores = cross_validate(QDA, X_train, y_train, scoring=scoring, cv=5)



# sorted(scores.keys())

# QDA_fit_time = scores['fit_time'].mean()

# QDA_score_time = scores['score_time'].mean()

# QDA_accuracy = scores['test_accuracy'].mean()

# QDA_precision = scores['test_precision_macro'].mean()

# QDA_recall = scores['test_recall_macro'].mean()

# QDA_f1 = scores['test_f1_weighted'].mean()

# QDA_roc = scores['test_roc_auc'].mean()
# random_forest = RandomForestClassifier()



# scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

# scores = cross_validate(random_forest, X_train, y_train, scoring=scoring, cv=5)



# sorted(scores.keys())

# forest_fit_time = scores['fit_time'].mean()

# forest_score_time = scores['score_time'].mean()

# forest_accuracy = scores['test_accuracy'].mean()

# forest_precision = scores['test_precision_macro'].mean()

# forest_recall = scores['test_recall_macro'].mean()

# forest_f1 = scores['test_f1_weighted'].mean()

# forest_roc = scores['test_roc_auc'].mean()
# random_forest.feature_importances_
# random_forest.fit(X_train,y_train)

# roc_auc_score(y_test,random_forest.predict(X_test))
# KNN = KNeighborsClassifier()



# scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

# scores = cross_validate(KNN, X_train, y_train, scoring=scoring, cv=5)



# sorted(scores.keys())

# KNN_fit_time = scores['fit_time'].mean()

# KNN_score_time = scores['score_time'].mean()

# KNN_accuracy = scores['test_accuracy'].mean()

# KNN_precision = scores['test_precision_macro'].mean()

# KNN_recall = scores['test_recall_macro'].mean()

# KNN_f1 = scores['test_f1_weighted'].mean()

# KNN_roc = scores['test_roc_auc'].mean()
# KNN.fit(X_train,y_train)

# roc_auc_score(y_test,KNN.predict(X_test))
# bayes = GaussianNB()



# scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

# scores = cross_validate(bayes, X_train, y_train, scoring=scoring, cv=5)



# sorted(scores.keys())

# bayes_fit_time = scores['fit_time'].mean()

# bayes_score_time = scores['score_time'].mean()

# bayes_accuracy = scores['test_accuracy'].mean()

# bayes_precision = scores['test_precision_macro'].mean()

# bayes_recall = scores['test_recall_macro'].mean()

# bayes_f1 = scores['test_f1_weighted'].mean()

# bayes_roc = scores['test_roc_auc'].mean()
# models_initial = pd.DataFrame({

#     'Model'       : ['Decision Tree', 'Support Vector Machine', 'Linear Discriminant Analysis', 'Quadratic Discriminant Analysis', 'Random Forest', 'K-Nearest Neighbors', 'Bayes'],

#     'Fitting time': [dtree_fit_time, SVM_fit_time, LDA_fit_time, QDA_fit_time, forest_fit_time, KNN_fit_time, bayes_fit_time],

#     'Scoring time': [dtree_score_time, SVM_score_time, LDA_score_time, QDA_score_time, forest_score_time, KNN_score_time, bayes_score_time],

#     'Accuracy'    : [dtree_accuracy, SVM_accuracy, LDA_accuracy, QDA_accuracy, forest_accuracy, KNN_accuracy, bayes_accuracy],

#     'Precision'   : [dtree_precision, SVM_precision, LDA_precision, QDA_precision, forest_precision, KNN_precision, bayes_precision],

#     'Recall'      : [dtree_recall, SVM_recall, LDA_recall, QDA_recall, forest_recall, KNN_recall, bayes_recall],

#     'F1_score'    : [dtree_f1, SVM_f1, LDA_f1, QDA_f1, forest_f1, KNN_f1, bayes_f1],

#     'AUC_ROC'     : [dtree_roc, SVM_roc, LDA_roc, QDA_roc, forest_roc, KNN_roc, bayes_roc],

#     }, columns = ['Model', 'Fitting time', 'Scoring time', 'Accuracy', 'Precision', 'Recall', 'F1_score', 'AUC_ROC'])



# models_initial.sort_values(by='Accuracy', ascending=False)
# from sklearn.ensemble import AdaBoostClassifier

# from sklearn import model_selection

# seed = 7

# num_trees = 30

# kfold = model_selection.KFold(n_splits=10, random_state=seed)

# model_ada = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)

# scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

# scores = cross_validate(model, X_train, y_train, scoring=scoring, cv=5)

# ada_fit_time = scores['fit_time'].mean()

# ada_score_time = scores['score_time'].mean()

# ada_accuracy = scores['test_accuracy'].mean()

# ada_precision = scores['test_precision_macro'].mean()

# ada_recall = scores['test_recall_macro'].mean()

# ada_f1 = scores['test_f1_weighted'].mean()

# ada_roc = scores['test_roc_auc'].mean()

# print(ada_roc)
# xgb_model = xgb.XGBRegressor(objective="reg:logistic", random_state=42)



# xgb_model.fit(X_train, y_train)



# y_predxg = xgb_model.predict(X_train)
# roc_auc_score(y_train, y_predxg)
# print(X_train.shape)
# print(X_test.shape, y_test.shape,type(y_test.to_numpy()),X_test[0:3,0:6])
# y_predtestxg=xgb_model.predict(X_test)

# roc_auc_score(y_test,xgb_model.predict(X_test))
# type(y_test)
# def evaluate_model(model, X, y):

# 	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# 	scores = cross_validate(model, X, y, scoring='auc', cv=cv, n_jobs=-1, error_score='raise')

# 	return scores
# smt3=SMOTE(random_state=14,sampling_strategy={1: 7000})

# RUS3=RandomUnderSampler(sampling_strategy={0: 21000},random_state=9)

# pipe = make_pipeline(smt3,RUS3)



# X_smt3, y_smt3 = pipe.fit_resample(X_train,y_train)
# level0 = list()

# # level0.append(('decision_tree', decision_tree))

# # level0.append(('bayes', bayes))

# # level0.append(('KNN', KNN))

# # level0.append(('random_forest', random_forest))

# level0.append(('LDR', LinearDiscriminantAnalysis()))

# # define meta learner model

# level1 = RandomForestClassifier(n_estimators=150,class_weight={1:5,0:1})

# # define the stacking ensemble

# model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)

# # fit the model on all available data

# model.fit(X_smt3, y_smt3)
# print(roc_auc_score(y_train,model.predict(X_train)),roc_auc_score(y_test,model.predict(X_test)))
# level0 = list()

# level0.append(('decision_tree', decision_tree))

# # level0.append(('bayes', bayes))

# # level0.append(('KNN', KNN))

# # level0.append(('random_forest', random_forest))

# # define meta learner model

# level1 = LogisticRegression()

# # define the stacking ensemble

# dec_log = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)

# # fit the model on all available data

# dec_log.fit(X_train, y_train)
# yhatensemble=dec_log.predict(X_train)
# y_train2=y_trainres
# roc_auc_score(y_test,model.predict(X_test))
# level20 = list()

# level20.append(('decision_tree', decision_tree))

# # level0.append(('bayes', bayes))

# # level0.append(('KNN', KNN))

# # level0.append(('random_forest', random_forest))

# # define meta learner model

# level21 = LogisticRegression()

# # define the stacking ensemble

# ensemblee2 = StackingClassifier(estimators=level20, final_estimator=level21, cv=5)

# # fit the model on all available data

# ensemblee2.fit(X_train, y_train)

# print(roc_auc_score(y_train,ensemblee2.predict(X_train)),roc_auc_score(y_test,ensemblee2.predict(X_test)))


# iso = IsolationForest(contamination=0.2)

# yhat = iso.fit_predict(X_train)

# mask = yhat != -1

# X_trainno_outlier, y_trainno_outlier = X_train[mask, :], y_train[mask]

# # summarize the shape of the updated training dataset

# print(X_trainno_outlier.shape, y_trainno_outlier.shape)
# ensemblee2.fit(X_trainno_outlier, y_trainno_outlier)

# print(roc_auc_score(y_trainno_outlier,ensemblee2.predict(X_trainno_outlier)),roc_auc_score(y_test,ensemblee2.predict(X_test)))
# params = {

#     "colsample_bytree": uniform(0.7, 0.3),

#     "gamma": [60,70,80,100,200],

#     "learning_rate": uniform(0.03, 0.3), # default 0.1 

#     "max_depth": randint(2, 6), # default 3

#     "n_estimators": randint(100, 150), # default 100

#     "subsample": uniform(0.6, 0.4)

#     "scale_pos_weight" 

# }



# grid = GridSearchCV(xgb_model, param_distributions=params, random_state=42, n_iter=200, cv=3, verbose=1, n_jobs=1, return_train_score=True)



# search.fit(X_train, y_train)



# gs_xb=GridSearchCV(xgb_model,params)

# search.fit(X_train, y_train)
# y_cvgrid=search.predict(X_train)

# y_cvgridtest=search.predict(X_test)

# print(roc_auc_score(y_train,y_cvgrid),roc_auc_score(y_test,y_cvgridtest))
# search_no_outlier = RandomizedSearchCV(xgb_model, param_distributions=params, random_state=42, n_iter=150, cv=3, verbose=1, n_jobs=1, return_train_score=True)

# search_no_outlier.fit(X_trainno_outlier, y_trainno_outlier)

# y_cvgridno_outlier=search.predict(X_trainno_outlier)

# y_cvgridtest=search.predict(X_test)

# print(roc_auc_score(y_trainno_outlier,y_cvgridno_outlier),roc_auc_score(y_test,y_cvgridtest))
# print(len(y_trainL[y_trainL==1]),len(y_trainL[y_trainL==0]))
# count_class_0 = 250000



# count_class_1 = 200000

# smt=SMOTE(random_state=14,sampling_strategy={1: count_class_1})

# nmLt=NearMiss(version=1,sampling_strategy={0: count_class_0})

# pipe = make_pipeline(smt,nmLt)



# X_smt, y_smt = pipe.fit_resample(X_trainL,y_trainL)
# # count_class_0 = 250000



# # count_class_1 = 200000

# smt=SMOTE(random_state=14,sampling_strategy={1: 300000})

# nmLt=NearMiss(version=1,sampling_strategy={0: 300000})

# pipe = make_pipeline(smt,nmLt)



# X_smt1, y_smt1 = pipe.fit_resample(X_trainL,y_trainL)
# # count_class_0 = 250000



# # count_class_1 = 200000

# smt=SMOTE(random_state=14,sampling_strategy={1: 45000})

# nmLt=NearMiss(version=1,sampling_strategy={0: 50000})

# pipe = make_pipeline(smt,nmLt)



# X_smt2, y_smt2 = pipe.fit_resample(X_trainL,y_trainL)
# isoL = IsolationForest(contamination=0.2)

# yhatL = isoL.fit_predict(X_trainL)

# maskL = yhatL != -1

# X_trainno_outlierL, y_trainno_outlierL = X_trainL[maskL, :], y_trainL[maskL]

# # summarize the shape of the updated training dataset

# xgb_modelL = xgb.XGBClassifier(objective="binary:logistic",tree_method ='gpu_hist', random_state=42,eval_metric='auc')

# xgb_modelL.fit(X_trainL,  y_trainL)
# params = {

#     "colsample_bytree": uniform(0.7, 0.3),

#     "gamma": uniform(0, 0.5),

#     "learning_rate": uniform(0.03, 0.3), # default 0.1 

#     "max_depth": randint(2, 6), # default 3

#     "n_estimators": randint(100, 150), # default 100

#     "subsample": uniform(0.3, 0.8)

# }
# search_no_outlierL = RandomizedSearchCV(xgb_modelL, param_distributions=params, random_state=42, n_iter=100, cv=3, verbose=1, n_jobs=1, return_train_score=True)

# search_no_outlierL.fit(X_trainno_outlierL, y_trainno_outlierL)

# y_cvgridno_outlierL=search_no_outlierL.predict(X_trainno_outlierL)

# y_cvgridtestL=search_no_outlierL.predict(X_testL)

# print(roc_auc_score(y_trainno_outlierL,y_cvgridno_outlierL),roc_auc_score(y_testL,y_cvgridtestL))
# xgb_modelL.feature_importances_
# l=list()

# l.append()

# modelLen = StackingClassifier(estimators=random_forest, final_estimator=LogisticRegression(), cv=5)

# # fit the model on all available data

# modelLen.fit(X_trainno_outlierL, y_trainno_outlierL)
# from sklearn.model_selection import GridSearchCV
# param_grid = { 

#     'n_estimators': [200, 500],

#     'max_features': ['auto', 'sqrt', 'log2'],

#     'max_depth' : [4,5,6,7,8],

#     'criterion' :['gini', 'entropy']

# }
# CV_rfc = GridSearchCV(estimator=random_forest, param_grid=param_grid, cv= 5)

# CV_rfc.fit(X_trainno_outlierL, y_trainno_outlierL)
X_trainL, X_testL, y_trainL, y_testL = train_test_split(X, y, test_size=0.05, random_state=0)

X_trainL = sc.fit_transform(X_trainL)

X_testL=sc.transform(X_testL)

dtrain = xgb.DMatrix(X_trainL, label=y_trainL)
dtest=xgb.DMatrix(X_testL, label=y_testL)
paramxb = {'eta': 0.1,'objective': 'binary:logistic','max_delta_step':30,'subsample':0.5,'lambda':2,

           'alpha':0,'tree_method':'gpu_hist','colsample_bylevel':1}

paramxb['gamma'] = 60

paramxb['eval_metric'] = 'auc'

paramxb['colsample_bytree'] = 0.5

paramxb['scale_pos_weight']=5

paramxb['min_child_weight']=180

paramxb['colsample_bynode']=1
evallist = [(dtest, 'eval'), (dtrain, 'train')]
num_round = 7437

bst = xgb.train(paramxb, dtrain, num_round, evallist)
# num_round5 = 5000

# bst = xgb.train(paramxb, dtrain, num_round5, evallist)
# search_no_outlierL = RandomizedSearchCV(xgb_modelL, param_distributions=params, random_state=42, n_iter=100, cv=3, verbose=1, n_jobs=1, return_train_score=True)

# search_no_outlierL.fit(X_trainno_outlierL, y_trainno_outlierL)
# bst.save_model('0001.model')
# bst = xgb.Booster({'nthread':4})

# bst.load_model('model.bin')
# xgb_grid = xgb.XGBClassifier(objective="binary:logistic",tree_method ='gpu_hist', random_state=42,eval_metric='auc',gamma=42.5,n_estimators=200,colsample_bytree=0.5,scale_pos_weight=5,min_child_weight=150,learning_rate=0.1,max_delta_step=30,subsample=0.5,reg_lambda=0.65)
# params = {

# #     "gamma": [60,70,80,100,200],

# #     "learning_rate": [0.03,0.07],

# #     "n_estimators": [200,500],

#     "subsample": [0,0.2,0.5,0.8,1],

#     "scale_pos_weight" : [5,20,100,200,385,500],

#     "colsample_bytree" : [0,0.2,0.5,0.8,1],

#     "colsample_bylevel" : [0,0.2,0.5,0.8,1],

#     "colsample_bynode": [0,0.2,0.5,0.8,1]

# }
# # cv = RepeatedStratifiedKFold(n_splits=6, n_repeats=2, random_state=1)

# grid =  RandomizedSearchCV(xgb_grid,n_iter=100, param_distributions=params, n_jobs=-1, cv=3, scoring='roc_auc',return_train_score=True)
# grid_result = grid.fit(X_trainL, y_trainL)
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# means = grid_result.cv_results_['mean_test_score']

# stds = grid_result.cv_results_['std_test_score']

# params = grid_result.cv_results_['params']

# y_pred_CV=grid.predict(X_testL)

# print(roc_auc_score(y_testL,y_pred_CV))
# paramxb2 = {'eta': 0.1, 'objective': 'binary:logistic','max_delta_step':30,'subsample':0.5,'lambda':0.65,'alpha':0,'tree_method':'gpu_hist'}

# paramxb2['gamma'] = 70

# paramxb2['eval_metric'] = 'auc'

# paramxb2['colsample_bytree'] = 0.5

# paramxb2['scale_pos_weight']=1

# paramxb2['min_child_weight']=10

# paramxb2['colsample_bylevel']=1
# evallist2 = [(dtest, 'eval'), (dtrain2, 'train')]
# num_round2 = 50

# bst2=xgb.train(paramxb2, dtrain2, num_round2, evallist2)
# bst2.save_model('0001.model2')
# smt3=SMOTE(random_state=14,sampling_strategy={1: 500000})

# # RUS3=RandomUnderSampler(sampling_strategy={0: 550000},random_state=9)

# pipe = make_pipeline(smt3)



# X_smt3, y_smt3 = pipe.fit_resample(X_trainL,y_trainL)
# dtrain3 = xgb.DMatrix(X_smt3, label=y_smt3)

# paramxb3 = {'eta': 0.05, 'updater':'grow_gpu_hist','objective': 'binary:logistic','max_delta_step':30,'subsample':0.5,'lambda':0.65,'alpha':0,'tree_method':'gpu_hist'}

# paramxb3['gamma'] = 200

# paramxb3['eval_metric'] = 'auc'

# paramxb3['colsample_bytree'] = 0.5

# paramxb3['scale_pos_weight']=1

# paramxb3['min_child_weight']=200

# paramxb3['colsample_bylevel']=1
# evallist3= [(dtest, 'eval'), (dtrain3, 'train')]
# num_round3 = 100

# bst3=xgb.train(paramxb3, dtrain3, num_round3, evallist3)
idd=testf['id']

testf2=testf.drop(['id'],axis=1)
testf2.values.shape

test2=sc.transform(testf2)

# type(test2)
y_pred1=bst.predict(dtestf2)
data={'id':idd,

      'target':y_pred1}

output2=pd.DataFrame(data)

out= output2.to_csv('out3.csv', index = False)

# paramxb3 = {'learning_rate': 1, 'updater':'grow_gpu_hist','objective': 'binary:logistic','n_estimators':5,

#             'max_delta_step':30,'subsample':0.5,'lambda':0.65,'alpha':0,'tree_method':'gpu_hist'}

# paramxb3['gamma'] = 200

# paramxb3['eval_metric'] = 'auc'

# paramxb3['colsample_bytree'] = 0.8

# paramxb3['scale_pos_weight']=1

# paramxb3['min_child_weight']=200

# paramxb3['colsample_bynode']=0.5
# xgb_forest = xgb.XGBRFClassifier(**paramxb3)
# xgb_forest.fit(X_trainL, y_trainL,

#         eval_set=[(X_trainL, y_trainL), (X_testL, y_testL)],

#         eval_metric='auc',

#         verbose=True)
# evals_result = xgb_forest.evals_result()

# print(evals_result)
# preds = xgb_forest.predict(X_testL, ntree_limit=50)

# print(roc_auc_score(y_testL,preds))