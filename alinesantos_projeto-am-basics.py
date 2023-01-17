# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import matplotlib

#matplotlib.use("Agg") #Needed to save figures

from sklearn import cross_validation

import xgboost as xgb

from sklearn.metrics import roc_auc_score

from sklearn.neural_network import MLPClassifier



training = pd.read_csv("../input/trainv2santander/train_v2.csv", index_col=0)

#training = pd.read_csv("../input/modifiedtrainset/train_v2.csv", index_col=0)

test = pd.read_csv("../input/santander-customer-satisfaction/test.csv", index_col=0)

#test = pd.read_csv("../input/modifiedtrainset/test_v2.csv", index_col=0)



print(training.shape)

print(test.shape)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input/onesideselec"]).decode("utf8"))

training.head()

# Any results you write to the current directory are saved as output.
X = training.iloc[:,:-1]

y = training.TARGET



#X['n0'] = (X == 0).sum(axis=1)

#X['logvar38'] = X['var38'].map(np.log1p)

#X['var38mc'] = np.isclose(X.var38, 117310.979016)

feature = ['var3','var15','imp_ent_var16_ult1','imp_op_var39_comer_ult1',

           'imp_op_var39_comer_ult3','imp_op_var40_comer_ult1','imp_op_var40_comer_ult3',

           'imp_op_var40_efect_ult1','imp_op_var40_efect_ult3','imp_op_var40_ult1',

           'imp_op_var41_comer_ult1','imp_op_var41_comer_ult3','imp_op_var41_efect_ult1',

           'imp_op_var41_efect_ult3','imp_op_var41_ult1','imp_op_var39_efect_ult1',

           'imp_op_var39_efect_ult3','imp_op_var39_ult1','imp_sal_var16_ult1',

           'ind_var1_0','ind_var5_0','ind_var5','ind_var8_0','ind_var12_0','ind_var13_0',

           'ind_var13','ind_var14_0','ind_var24','ind_var25_cte','ind_var26_cte',

           'ind_var30_0','ind_var30','ind_var37_cte','ind_var37_0','ind_var39_0','ind_var40_0',

           'ind_var41_0','num_var4','num_var8_0','num_op_var41_hace2','num_op_var41_hace3',

           'num_op_var41_ult1','num_op_var41_ult3','num_op_var39_hace2','num_op_var39_ult1',

           'num_op_var39_ult3','num_var30_0','num_var30','num_var35','num_var37_med_ult2',

           'num_var37_0','num_var39_0','num_var41_0','num_var42_0','num_var42',

           'saldo_var1','saldo_var5','saldo_var8','saldo_var12','saldo_var13_corto',

           'saldo_var13','saldo_var26','saldo_var25','saldo_var30','saldo_var37',

           'saldo_var42','saldo_var46','var36','delta_imp_amort_var34_1y3',

           'delta_imp_reemb_var33_1y3','delta_imp_trasp_var33_in_1y3','delta_num_aport_var13_1y3',

           'imp_aport_var13_hace3','imp_compra_var44_hace3','imp_var43_emit_ult1',

           'imp_trans_var37_ult1','ind_var10_ult1','ind_var10cte_ult1',

           'ind_var9_cte_ult1','ind_var9_ult1','ind_var43_emit_ult1','num_ent_var16_ult1',

           'num_var22_hace2','num_var22_hace3','num_var22_ult1','num_var22_ult3',

           'num_med_var22_ult3','num_med_var45_ult3','num_meses_var5_ult3',

           'num_meses_var8_ult3','num_meses_var39_vig_ult3','num_op_var39_comer_ult1',

           'num_op_var39_comer_ult3','num_op_var41_comer_ult1','num_op_var41_comer_ult3',

           'num_op_var41_efect_ult1','num_op_var41_efect_ult3','num_op_var39_efect_ult1',

           'num_op_var39_efect_ult3','num_reemb_var13_ult1','num_var43_emit_ult1',

           'num_var43_recib_ult1','num_var45_hace2','num_var45_hace3','num_var45_ult1',

           'num_var45_ult3','saldo_medio_var5_hace2','saldo_medio_var5_hace3','saldo_medio_var5_ult1',

           'saldo_medio_var5_ult3','saldo_medio_var8_hace2','saldo_medio_var8_hace3',

           'saldo_medio_var8_ult1','saldo_medio_var8_ult3','saldo_medio_var12_hace2',

           'saldo_medio_var12_hace3','saldo_medio_var12_ult1','saldo_medio_var12_ult3',

           'saldo_medio_var13_corto_hace2','saldo_medio_var13_corto_ult3','saldo_medio_var44_ult3'

           ,'var38','n0','var38mc','logvar38']

X_sel = X[feature]

X_sel.head()
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold



X = X_sel

skf = StratifiedKFold(n_splits=5, random_state=1301)

auc = np.zeros(5)

i=0



for train_index, test_index in skf.split(X, y):



    X_train = X.iloc[train_index,:]

    y_train = y.iloc[train_index]

    X_test = X.iloc[test_index,:]

    y_test = y.iloc[test_index]



    ratio = float(np.sum(y == 1)) / np.sum(y==0)



    # Initial parameters for the parameter exploration



    # clf = xgb.XGBClassifier(missing=9999999999,

    #                         max_depth = 10,

    #                         n_estimators=1000,

    #                         learning_rate=0.1,

    #                         nthread=4,

    #                         subsample=1.0,

    #                         colsample_bytree=0.5,

    #                         min_child_weight = 5,

    #                         scale_pos_weight = ratio,

    #                         seed=4242)

    # parameters = {'max_depth':[5,10,20,40,80,120,160,200,240,280,320],

    #               'subsample':[0.8, 0.9, 1.0],

    #               'min_child_weight':[8,3,1,4,2],

    #               'colsample_bytree':[0.3,0.7,0.6,0.5],

    #               'learning_rate':[0.07, 0.2, 0.15, 0.09, 0.1, 0.05],}



    print(X_train.shape[1])

    clf = xgb.XGBClassifier(missing=9999999999,

                    max_depth = X_train.shape[1],

                    subsample = 1.0,

                    min_child_weight = 8,

                    colsample_bytree = 1.0,

                    learning_rate = 0.1,

                    n_estimators=300,

                    nthread=4,

                    scale_pos_weight = ratio,

                    reg_alpha=0.03,

                    seed=1301)



    print('begin')



    clf.fit(X_train, y_train, early_stopping_rounds=50, eval_metric="auc", eval_set=[(X_test, y_test)])

    print('Overall AUC:', roc_auc_score(y_test, clf.predict_proba(X_test, ntree_limit=clf.best_iteration)[:,1]))

    auc[i] = roc_auc_score(y_test, clf.predict_proba(X_test, ntree_limit=clf.best_iteration)[:,1])

    i += 1

    print('end')



print(auc)

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold



parameters = {'learning_rate_init':[0.1, 0.01, 0.001],'hidden_layer_sizes':[5, 10, 15, 50, 100]}



svc = MLPClassifier(max_iter=300)

clf = GridSearchCV(svc, parameters, cv = 5, scoring = 'roc_auc', n_jobs=4)



print('begin')

clf = clf.fit(X_sel, y)

print('end')



print(clf.cv_results_)

print(clf.best_estimator_)

print(clf.best_score_)

print(clf.best_params_)

print(clf.scorer_)
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold

from sklearn.neighbors import KNeighborsClassifier



parameters = {'n_neighbors':[3, 5, 7, 9, 11, 13, 15]}

X_sel = X

svc = KNeighborsClassifier()



clf = GridSearchCV(svc, parameters, cv = 5, scoring = 'roc_auc')



print('begin')

clf = clf.fit(X_sel, y)

print('end')





print(clf.cv_results_)

print(clf.best_estimator_)

print(clf.best_score_)

print(clf.best_params_)

print(clf.scorer_)



X_test = test



X_test = X_test.replace(-999999,2)

X_test['n0'] = (X_test == 0).sum(axis=1)

X_test['logvar38'] = X_test['var38'].map(np.log1p)

X_test['var38mc'] = np.isclose(X_test.var38, 117310.979016)



X_test = X_test[feature]



print(X_test.shape, X_sel.shape)



#prev = clf.predict_proba(X_test, ntree_limit=clf.best_iteration)[:,1]

prev = clf.predict_proba(X_test)[:,1]



testid = pd.read_csv("../input/santander-customer-satisfaction/test.csv", usecols=[0])

testid = testid['ID']





submission = pd.DataFrame({"ID":testid, "TARGET":prev})

submission.to_csv("submission.csv", index=False)



submission.head()
