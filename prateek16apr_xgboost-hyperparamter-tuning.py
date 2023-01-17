!pip install xgboost
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
training=pd.read_csv("/kaggle/input/titanic/train.csv")
training['Age']=training.Age.fillna(training.groupby("Sex")["Age"].transform("mean"))
training['Sex']=np.where(training.Sex=='male',1,0)
features=['Survived','Pclass','Sex','Fare','SibSp','Parch', 'Age']
training_processed=training[features]
y = training_processed.Survived
X = training_processed.drop('Survived', axis=1)
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
model=xgb.XGBClassifier(random_state=1,learning_rate=0.01)
model.fit(train_X, train_y)
val_predictions = model.predict(val_X)
accuracy_score(val_y, val_predictions)
def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    print("Number of Estimators %f " %alg.n_estimators )
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Survived'],eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain['Survived'].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Survived'], dtrain_predprob))
                    
    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn import model_selection  #Additional scklearn functions
from sklearn.model_selection import GridSearchCV  #Perforing grid search

target='Survived'

predictors = [x for x in training_processed.columns if x not in [target]]
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb1, training_processed, predictors)
param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=56, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch1.fit(training_processed[predictors],training_processed[target])
gsearch1.best_params_, gsearch1.best_score_
param_test2 = {
 'max_depth':[6,7,8],
 'min_child_weight':[4,5,6]
}
gsearch2 = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=56, max_depth=5,
 min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test2, scoring='roc_auc',n_jobs=4, cv=5)
gsearch2.fit(training_processed[predictors],training_processed[target])
gsearch2.best_params_, gsearch2.best_score_
param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=56, max_depth=7,
 min_child_weight=5, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test3, scoring='roc_auc',n_jobs=4, cv=5)
gsearch3.fit(training_processed[predictors],training_processed[target])
gsearch3.best_params_, gsearch3.best_score_
xgb2 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=7,
 min_child_weight=5,
 gamma=0.3,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb2, training_processed, predictors)
param_test4 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}
gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=167, max_depth=7,
 min_child_weight=5, gamma=0.3, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test4, scoring='roc_auc',n_jobs=4, cv=5)
gsearch4.fit(training_processed[predictors],training_processed[target])
gsearch4.best_params_, gsearch4.best_score_
param_test6 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}

gsearch6 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=167, max_depth=7,
 min_child_weight=5, gamma=0.3, subsample=0.8, colsample_bytree=0.7,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test6, scoring='roc_auc',n_jobs=4, cv=5)
gsearch6.fit(training_processed[predictors],training_processed[target])
gsearch6.best_params_, gsearch4.best_score_
param_test7 = {
 'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]
}

gsearch7 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=167, max_depth=7,
 min_child_weight=5, gamma=0.3, subsample=0.8, colsample_bytree=0.7,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test7, scoring='roc_auc',n_jobs=4,cv=5)
gsearch7.fit(training_processed[predictors],training_processed[target])
gsearch7.best_params_, gsearch7.best_score_
xgb3 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=7,
 min_child_weight=5,
 gamma=0.3,
 subsample=0.8,
 colsample_bytree=0.7,
 reg_alpha=0.001,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb3, training_processed, predictors)
xgb4 = XGBClassifier(
 learning_rate =0.01,
 n_estimators=5000,
 max_depth=7,
 min_child_weight=5,
 gamma=0.3,
 subsample=0.8,
 colsample_bytree=0.7,
 reg_alpha=0.001,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb4, training_processed, predictors)
xgb4 = XGBClassifier(
 learning_rate =0.01,
 n_estimators=5000,
 max_depth=7,
 min_child_weight=5,
 gamma=0.3,
 subsample=0.8,
 colsample_bytree=0.7,
 reg_alpha=0.001,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit2(xgb4, training_processed, predictors,early_stopping_rounds=200)
test=pd.read_csv("/kaggle/input/titanic/test.csv")
test['Age']=test.Age.fillna(test.groupby("Sex")["Age"].transform("mean"))
test['Sex']=np.where(test.Sex=='male',1,0)
test_processed=test[['Pclass','Sex','Fare','SibSp','Parch', 'Age']]
test_predictions = xgb3.predict(test_processed)
dataset = pd.DataFrame({'Survived': test_predictions[:,]})
test[['PassengerId']].join(dataset).to_csv("results.csv",index=False)