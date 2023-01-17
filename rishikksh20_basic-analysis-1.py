import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pandas import Series,DataFrame
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
import xgboost as xgb
%matplotlib inline
from sklearn.cross_validation import train_test_split
train_df   = pd.read_csv('../input/train.csv')
test_df  = pd.read_csv('../input/test.csv')
sns.countplot(x="TARGET", data=train_df)



for feat in train_df.columns:
    if train_df[feat].dtype == 'float64':
        train_df[feat][np.isnan(train_df[feat])] = train_df[feat].mean()
        test_df[feat][np.isnan(test_df[feat])] = test_df[feat].mean()
      
    elif train_df[feat].dtype == 'object':
        train_df[feat][train_df[feat] != train_df[feat]] = train_df[feat].value_counts().index[0]
        test_df[feat][test_df[feat] != test_df[feat]] = test_df[feat].value_counts().index[0]
for feat in train_df.columns:
    if train_df[feat].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(np.unique(list(train_df[feat].values) + list(test_df[feat].values)))
        train_df[feat]   = lbl.transform(list(train_df[feat].values))
        test_df[feat]  = lbl.transform(list(test_df[feat].values))
X_train = train_df.drop(["ID","TARGET"],axis=1)
Y_train = train_df["TARGET"]
X_test  = test_df.drop("ID",axis=1).copy()
xgtrain = xgb.DMatrix(X_train, Y_train)
xgtest = xgb.DMatrix(X_test)
# Create watchlist folds
x_train, x_eval, y_train, y_eval = train_test_split(X_train, Y_train, test_size=0.33, stratify=Y_train, random_state=1)
dtrain = xgb.DMatrix(x_train, y_train)
deval = xgb.DMatrix(x_eval, y_eval)
watchlist = [(dtrain, 'train'), (deval, 'eval')]
# XGBoost params
xgboost_params = {
    'objective': 'multi:softprob',
    'booster': 'gbtree',
    'eval_metric': 'auc',
    'eta': 0.3,
    'subsample': 0.75,
    'colsample_bytree': 0.75,
}

# Grasp on parameters
num_rounds = 1000
cv = xgb.cv(xgboost_params, xgtrain, metrics='auc', num_boost_round=num_rounds, nfold=5, stratified=True, seed=1, early_stopping_rounds=25, verbose_eval=True,show_stdv=False)
# Modeling
bst = xgb.train(
    xgboost_params,
    dtrain,
    num_boost_round=num_rounds,
    evals=watchlist,
    early_stopping_rounds=25,
    verbose_eval=True)
# Make predictions
preds = bst.predict(xgtest, ntree_limit=bst.best_iteration)
# Create submission

submission = pd.DataFrame()
submission["ID"] = test_df["ID"]
submission["TARGET"] = preds[:,1]

submission.to_csv('santander.csv', index=False)

