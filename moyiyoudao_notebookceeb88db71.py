# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt

%matplotlib inline
train_df = pd.read_csv('../input/mushrooms.csv')

train_df.head()
train_df.info()
#

var = ['cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing',

       'gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring',

      'stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type',

      'spore-print-color','population','habitat','class']

for v in var:

    print('\nFrequency count for variable %s'%v)

    print(train_df[v].value_counts())
train_df['class'] = train_df['class'].map({'e':1,'p':0})

y = train_df.pop('class')

train_df = pd.get_dummies(train_df)

train_df.info()
y.dtypes
import xgboost as xgb

from sklearn import cross_validation, metrics   #Additional     scklearn functions

def Xgmodelfit(alg, dtrain, predictors,y,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

    if useTrainCV:

        xgb_param = alg.get_xgb_params()

        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=y.values)

        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'],

            nfold=cv_folds,metrics='auc', early_stopping_rounds=early_stopping_rounds)

        alg.set_params(n_estimators=cvresult.shape[0])

    

    #Fit the algorithm on the data

    alg.fit(dtrain[predictors], y,eval_metric='auc')

    

    #Predict training set:

    dtrain_predictions = alg.predict(dtrain[predictors])

    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]

    

    #Print model report:

    print ("\nModel Report")

    print ("Accuracy : %.4g" % metrics.accuracy_score(y.values, dtrain_predictions))

    print ("AUC Score (Train): %f" % metrics.roc_auc_score(y, dtrain_predprob))

    

    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)

    feat_imp.plot(kind='bar', title='Feature Importances')

    plt.ylabel('Feature Importance Score')

predictors = train_df.columns
#para init

from xgboost.sklearn import XGBClassifier

xgb1 = XGBClassifier(

 learning_rate =0.1,

 n_estimators=1000,

 max_depth=5,

 min_child_weight=1,

 gamma=0,

 subsample=0.8,

 colsample_bytree=0.8,

 objective= 'binary:logistic',

 n_jobs=4,

 scale_pos_weight=1,

 random_state=27)

Xgmodelfit(xgb1,train_df,predictors,y)
#'max_depth''min_child_weight调试

from sklearn.model_selection import GridSearchCV

param_test1 = {

 'max_depth':range(3,10,2),

 'min_child_weight':range(1,6,2)

}

gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140,gamma=0, subsample=0.8,

        colsample_bytree=0.8,objective= 'binary:logistic', n_jobs=4,scale_pos_weight=1, random_state=27), 

    param_grid = param_test1,     scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch1.fit(train_df[predictors],y)

from sklearn.model_selection import GridSearchCV

param_test1 = {

 'max_depth':range(2,5,1),

 'min_child_weight':range(1,3,1)

}

gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140,gamma=0, subsample=0.8,

        colsample_bytree=0.8,objective= 'binary:logistic', n_jobs=4,scale_pos_weight=1, random_state=27), 

    param_grid = param_test1,     scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch1.fit(train_df[predictors],y)

gsearch1.grid_scores_, gsearch1.best_params_,gsearch1.best_score_
#调试gamma

param_test3 = {

 'gamma':[i/10.0 for i in range(0,5)]

}

gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=3, 

    min_child_weight=1,  subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', n_jobs=4, 

    scale_pos_weight=1,random_state=27), param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)



gsearch3.fit(train_df[predictors],y)

gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_
xgb2 = XGBClassifier(

 learning_rate =0.1,

 n_estimators=1000,

 max_depth=3,

 min_child_weight=1,

 gamma=0.4,

 subsample=0.8,

 colsample_bytree=0.8,

 objective= 'binary:logistic',

 n_jobs=4,

scale_pos_weight=1,

random_state=27)

Xgmodelfit(xgb2, train_df, predictors,y)
#subsample 和 colsample_bytree 参数

param_test4 = {

 'subsample':[i/10.0 for i in range(6,10)],

 'colsample_bytree':[i/10.0 for i in range(6,10)]

}



gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=3, 

    min_child_weight=1, gamma=0.4,objective= 'binary:logistic', n_jobs=4, scale_pos_weight=1,random_state=27), 

        param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)



gsearch4.fit(train_df[predictors],y)
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_
#subsample 和 colsample_bytree 参数

param_test5 = {

 'subsample':[i/100.0 for i in range(75,83)],

 'colsample_bytree':[i/100.0 for i in range(75,83)]

}



gsearch5 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=3, 

    min_child_weight=1, gamma=0.4,objective= 'binary:logistic', n_jobs=4, scale_pos_weight=1,random_state=27), 

        param_grid = param_test5, scoring='roc_auc',n_jobs=4,iid=False, cv=5)



gsearch5.fit(train_df[predictors],y)

gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_