import os
print(os.listdir("../input"))
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import time
import catboost as cb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
traindata = pd.read_csv('../input/train.csv')
testdata = pd.read_csv('../input/test.csv')
submission =  pd.read_csv('../input/sample_submission.csv')
X_train = traindata.iloc[:, :-1].values
y_train = traindata.iloc[:, 14].values
X_test = testdata.iloc[:, :].values
start_time = time.time()
warnings.filterwarnings('ignore')
classifier = cb.CatBoostClassifier( task_type = 'GPU',silent=True , cat_features=[2,6,7,9,11,12,13 ], one_hot_max_size=2,loss_function='Logloss',eval_metric='AUC',boosting_type='Ordered', random_seed=25)
classifier.fit(X_train , y_train)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print ( accuracies.mean() *100)
print ( accuracies*100 )
print ( 'Variance:' , accuracies.std()*100 )
end_time = time.time()
print("The Execution Time  is %s seconds" % (end_time - start_time))
y_pred = classifier.predict(X_test).astype(int)
submission['class']=y_pred
submission.to_csv('Sub_catg_5.csv' , index = False)
start_time = time.time()
warnings.filterwarnings('ignore')
classifier = lgb.LGBMClassifier(silent=False,min_data_in_leaf=2000, subsample_for_bin=400000,n_estimators=300, categorical_feature = [2,6,7,9,11,12,13 ] ,random_state=None)
classifier.fit(X_train , y_train)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10 )
print ( accuracies.mean()*100 )
print ( accuracies*100 )
print ( 'Variance:' , accuracies.std()*100 )
end_time = time.time()
print("The Execution Time  is %s seconds" % (end_time - start_time))