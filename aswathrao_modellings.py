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
train = pd.read_csv('/kaggle/input/cleaned/trains.csv')
test = pd.read_csv('/kaggle/input/cleaned/tests.csv')
sample = pd.read_csv('/kaggle/input/hr-analysis/sample_submission.csv')
train
import h2o
h2o.init()
train1 = h2o.H2OFrame(train)
test1 = h2o.H2OFrame(test)
train1.columns
y = 'target'
x = train1.col_names
x.remove(y)
train1['target'] = train1['target'].asfactor()
train1['target'].levels()
from h2o.automl import H2OAutoML
aml = H2OAutoML(max_models = 20,max_runtime_secs=2000, seed = 42)
aml.train(x = x, y = y, training_frame = train1)
preds = aml.predict(test1)
ans=h2o.as_list(preds) 
sample['target'] = ans['predict']
sample.to_csv('Solution_of_H20_EDA.csv',index=False)


def extra_tree(Xtrain,Ytrain,Xtest):
    extra = ExtraTreesClassifier()
    extra.fit(Xtrain, Ytrain) 
    extra_prediction = extra.predict(Xtest)
    return extra_prediction

def Xg_boost(Xtrain,Ytrain,Xtest):
    xg = XGBClassifier(loss='exponential', learning_rate=0.05, n_estimators=1000, subsample=1.0, criterion='friedman_mse', 
                                  min_samples_split=2, 
                                  min_samples_leaf=5, min_weight_fraction_leaf=0.0, max_depth=10, min_impurity_decrease=0.0, 
                                  min_impurity_split=None, 
                                  init=None, random_state=None, max_features=None, verbose=1, max_leaf_nodes=None, warm_start=False, 
                                  presort='deprecated', 
                                  validation_fraction=0.1, n_iter_no_change=None, tol=0.0001)
    xg.fit(Xtrain, Ytrain) 
    xg_prediction = xg.predict(Xtest)
    return xg_prediction

def LGBM(Xtrain,Ytrain,Xtest):
    lgbm = LGBMClassifier(boosting_type='gbdt', num_leaves=40,
                            max_depth=5, learning_rate=0.05, n_estimators=1000, subsample_for_bin=200, objective='binary', 
                            min_split_gain=0.0, min_child_weight=0.001, min_child_samples=10,
                            subsample=1.0, subsample_freq=0, colsample_bytree=1.0, reg_alpha=0.0,
                            reg_lambda=0.0, random_state=None, n_jobs=1, silent=True, importance_type='split')
    #lgbm = LGBMClassifier(n_estimators= 500)
    lgbm.fit(X_train, Y_train)
    lgbm_preds = lgbm.predict(X_test)
    return lgbm_preds
cols = train.columns
cols
X_train = train[cols[:len(cols)-1]]
Y_train = train['target']
X_test = test[cols[:len(cols)-1]]
target = 'target'
scoring_parameter = 'balanced-accuracy'
!pip install autoviml
from autoviml.Auto_ViML import Auto_ViML
m, feats, trainm, testm = Auto_ViML(train, target, test,
                                    scoring_parameter=scoring_parameter,
                                    hyper_param='GS',feature_reduction=True,
                                     Boosting_Flag='Boosting_Flag',Binning_Flag=False)
from catboost import Pool, CatBoostClassifier, cv, CatBoostRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
cate_features_index = np.where(X_train.dtypes != float)[0]
model = CatBoostClassifier(iterations=7000, learning_rate=0.001, l2_leaf_reg=3.5, depth=5, 
                           rsm=0.99, loss_function= 'Logloss', eval_metric='AUC',use_best_model=True,random_seed=50)
xtrain,xtest,ytrain,ytest = train_test_split(X_train,Y_train,train_size=0.99,random_state=1236)
model.fit(xtrain,ytrain,cat_features=cate_features_index,eval_set=(xtest,ytest))
predss = model.predict(X_test)
sample['target'] = predss
sample.to_csv('Submission_suing_cat.csv',index = False)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.feature_selection import SelectFwe, f_regression
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import PolynomialFeatures
from tpot.builtins import OneHotEncoder, StackingEstimator
pred_xg = Xg_boost(X_train,Y_train,X_test)
pred_et = extra_tree(X_train,Y_train,X_test)
pred_l = LGBM(X_train,Y_train,X_test)
sample['target'] = pred_xg
print(sample['target'].unique())
sample.to_csv('XG.csv',index = False)
dict(sample['target'])
sample['target'] = pred_et
print(sample['target'].unique())
sample.to_csv('ET.csv',index = False)
sample['target'] = pred_l
print(sample['target'].unique())
sample.to_csv('LG.csv',index = False)
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0).fit(X_train, Y_train)
ans = clf.predict(X_test)
ans
sample['target'] = ans
print(sample['target'].unique())
ans
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=10).fit(X_train, Y_train)
prediction_of_rf = rf.predict(X_test)
sample['target'] = prediction_of_rf
print(sample['target'].unique())
sample.to_csv('RF.csv',index = False)