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



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df1 = pd.read_csv('/kaggle/input/final-project-sanbercode-0720/train.csv')

df2 = pd.read_csv('/kaggle/input/final-project-sanbercode-0720/test.csv')
df1.head()
## replace gaji

dict_gaji = {'<=7jt' : 0, '>7jt' : 1}

df1['Gaji'] = df1['Gaji'].replace(dict_gaji)
col_dummies = ['Kelas Pekerja','Status Perkawinan','Pekerjaan', 'Jenis Kelamin'] ## column yang diberi dummies

df1 = pd.get_dummies(df1, columns = col_dummies)

df2 = pd.get_dummies(df2, columns = col_dummies)
dict_pend = {

    '1st-4th' : 1, '5th-6th' : 2, 'SD' : 3, '7th-8th' : 4, '9th' : 5, '10th' : 6, '11th' : 7, '12th' : 8,

    'SMA' : 9, 'Sekolah Professional' : 10, 'Pendidikan Tinggi' : 11, 'D3' : 12, 'D4' : 13, 'Sarjana' : 14,

    'Master' : 15, 'Doktor' : 16

}



df1['Pendidikan'] = df1['Pendidikan'].replace(dict_pend)

df2['Pendidikan'] = df2['Pendidikan'].replace(dict_pend)
# pemisahan data feature dan target

from sklearn.preprocessing import scale



X = df1.drop(['id', 'Gaji'], axis = 1)

Xs = scale(X)

y = df1['Gaji']



X_test = scale(df2.drop(['id'], axis = 1))
from sklearn.model_selection import cross_val_score

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn import tree

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from xgboost import XGBClassifier
# importing GSCV dan RSCV

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV 
# fungsi untuk melihat performansi masing-masing model

def clf_performance(classifier, model_name):

    print(model_name)

    print('Best Score: ' + str(classifier.best_score_))

    print('Best Parameters: ' + str(classifier.best_params_))
# LogisticRegression model

lr = LogisticRegression()

param_grid = {'max_iter' : [2000],

              'penalty' : ['l1', 'l2'],

              'C' : np.logspace(-3,3,20),

              'solver' : ['liblinear']

             }



clf_lr = GridSearchCV(lr, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1, scoring = 'roc_auc')

best_clf_lr = clf_lr.fit(Xs,y)

clf_performance(best_clf_lr,'Logistic Regression')
# KNN model

knn = KNeighborsClassifier()

param_grid = {'n_neighbors' : np.arange(2,6,1),

              'weights' : ['uniform', 'distance'],

              'algorithm' : ['auto', 'ball_tree','kd_tree'],

              'p' : [1,2]}

clf_knn = GridSearchCV(knn, param_grid = param_grid, cv = 5, verbose = True,

                             n_jobs = -1, scoring = 'roc_auc')

best_clf_knn = clf_knn.fit(Xs,y)

clf_performance(best_clf_knn,'KNN')
#RF model

rf = RandomForestClassifier(random_state = 1)

param_grid =  {'n_estimators': [100,500,1000], 

                                  'bootstrap': [True,False],

                                  'max_depth': [3,5,10,20,50,75,100,None],

                                  'max_features': ['auto','sqrt'],

                                  'min_samples_leaf': [1,2,4,10],

                                  'min_samples_split': [2,5,10]}

                                  

clf_rf_rnd = RandomizedSearchCV(rf, param_distributions = param_grid, n_iter = 100, cv = 5,

                                verbose = True, n_jobs = -1, scoring = 'roc_auc')

best_clf_rf_rnd = clf_rf_rnd.fit(Xs,y)

clf_performance(best_clf_rf_rnd,'Random Forest')
## melihat signifikansi masing-masing feature pada RF model

best_rf = best_clf_rf.best_estimator_.fit(Xs,y)

feat_importances = pd.Series(best_rf.feature_importances_, index=Xs.columns)

feat_importances.nlargest(20).plot(kind='barh')
# XGB

xgb = XGBClassifier(random_state = 1)



param_grid = {

    'n_estimators': [20, 50, 100, 250, 500,1000],

    'colsample_bytree': [0.2, 0.5, 0.7, 0.8, 1],

    'max_depth': [2, 5, 10, 15, 20, 25, None],

    'reg_alpha': [0, 0.5, 1],

    'reg_lambda': [1, 1.5, 2],

    'subsample': [0.5,0.6,0.7, 0.8, 0.9],

    'learning_rate':[.01,0.1,0.2,0.3,0.5, 0.7, 0.9],

    'gamma':[0,.01,.1,1,10,100],

    'min_child_weight':[0,.01,0.1,1,10,100],

    'sampling_method': ['uniform', 'gradient_based']

}





clf_xgb_rnd = RandomizedSearchCV(xgb, param_distributions = param_grid, n_iter = 1000, cv = 5, verbose = True, n_jobs = -1, scoring = 'roc_auc')

best_clf_xgb_rnd = clf_xgb_rnd.fit(Xs,y)

clf_performance(best_clf_xgb_rnd,'XGB')
X_test = scale(df2.drop(['id'], axis = 1))
y_predict = best_clf_xgb_rnd.best_estimator_.predict(X_test).astype(int)

xgb_submission = {'id': df2['id'], 'Gaji': y_predict}

submission_xgb = pd.DataFrame(data=xgb_submission)

submission_xgb.to_csv('xgb_submission3.csv', index=False)
## voting 

best_lr = best_clf_lr.best_estimator_

best_knn = best_clf_knn.best_estimator_

best_svc = best_clf_svc.best_estimator_

best_rf = best_clf_rf_rnd.best_estimator_

best_xgb = best_clf_xgb_rnd.best_estimator_



voting_clf_hard = VotingClassifier(estimators = [('knn',best_knn),('rf',best_rf),('svc',best_svc)], voting = 'hard') 

voting_clf_soft = VotingClassifier(estimators = [('knn',best_knn),('rf',best_rf),('svc',best_svc)], voting = 'soft') 

voting_clf_all = VotingClassifier(estimators = [('knn',best_knn),('rf',best_rf),('svc',best_svc), ('lr', best_lr)], voting = 'soft') 

voting_clf_xgb = VotingClassifier(estimators = [('knn',best_knn),('rf',best_rf),('svc',best_svc), ('xgb', best_xgb),('lr', best_lr)], voting = 'soft')



print('voting_clf_hard :',cross_val_score(voting_clf_hard,X_train,y_train,cv=5))

print('voting_clf_hard mean :',cross_val_score(voting_clf_hard,X_train,y_train,cv=5).mean())



print('voting_clf_soft :',cross_val_score(voting_clf_soft,X_train,y_train,cv=5))

print('voting_clf_soft mean :',cross_val_score(voting_clf_soft,X_train,y_train,cv=5).mean())



print('voting_clf_all :',cross_val_score(voting_clf_all,X_train,y_train,cv=5))

print('voting_clf_all mean :',cross_val_score(voting_clf_all,X_train,y_train,cv=5).mean())



print('voting_clf_xgb :',cross_val_score(voting_clf_xgb,X_train,y_train,cv=5))

print('voting_clf_xgb mean :',cross_val_score(voting_clf_xgb,X_train,y_train,cv=5).mean())
params = {'weights' : [[1,1,1],[1,2,1],[1,1,2],[2,1,1],[2,2,1],[1,2,2],[2,1,2]]}



vote_weight = GridSearchCV(voting_clf_soft, param_grid = params, cv = 5, verbose = True, n_jobs = -1)

best_clf_weight = vote_weight.fit(Xs,y)

clf_performance(best_clf_weight,'VC Weights')

voting_clf_sub = best_clf_weight.best_estimator_.predict(Xs)
#Make Predictions 

voting_clf_hard.fit(Xs, y)

voting_clf_soft.fit(Xs, y)

voting_clf_all.fit(Xs, y)

voting_clf_xgb.fit(Xs, y)



best_rf.fit(Xs, y)

y_hat_vc_hard = voting_clf_hard.predict(X_test).astype(int)

y_hat_rf = best_rf.predict(X_test).astype(int)

y_hat_vc_soft =  voting_clf_soft.predict(X_test).astype(int)

y_hat_vc_all = voting_clf_all.predict(X_test).astype(int)

y_hat_vc_xgb = voting_clf_xgb.predict(X_test).astype(int)
final_data = {'id': df2['id'], 'Gaji': y_hat_rf}

submission = pd.DataFrame(data=final_data)



final_data_2 = {'id': df2['id'], 'Gaji': y_hat_vc_hard}

submission_2 = pd.DataFrame(data=final_data_2)



final_data_3 = {'id': df2['id'], 'Gaji': y_hat_vc_soft}

submission_3 = pd.DataFrame(data=final_data_3)



final_data_4 = {'id': df2['id'], 'Gaji': y_hat_vc_all}

submission_4 = pd.DataFrame(data=final_data_4)



final_data_5 = {'id': df2['id'], 'Gaji': y_hat_vc_xgb}

submission_5 = pd.DataFrame(data=final_data_5)



final_data_comp = {'id': df2['id'], 'Gaji_vc_hard': y_hat_vc_hard, 'Gaji_rf': y_hat_rf, 'Gaji_vc_soft' : y_hat_vc_soft, 'Gaji_vc_all' : y_hat_vc_all,  'Gaji_vc_xgb' : y_hat_vc_xgb}

comparison = pd.DataFrame(data=final_data_comp)
submission.to_csv('submission_rf.csv', index =False)

submission_2.to_csv('submission_vc_hard.csv',index=False)

submission_3.to_csv('submission_vc_soft.csv', index=False)

submission_4.to_csv('submission_vc_all.csv', index=False)

submission_5.to_csv('submission_vc_xgb2.csv', index=False)
print('DONE!!')