import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import cross_validation
import xgboost as xgb
from sklearn.metrics import roc_auc_score

training = pd.read_csv("../input/train.csv", index_col=0)
test = pd.read_csv("../input/test.csv", index_col=0)

print(training.shape)
print(test.shape)

# Replace -999999 in var3 column with most common value 2 
# See https://www.kaggle.com/cast42/santander-customer-satisfaction/debugging-var3-999999
# for details
training = training.replace(-999999,np.nan)
training=training.replace(9999999999,np.nan)

X = training.iloc[:,:-1]
y = training.TARGET


X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, stratify=y, test_size=0.3)

# xgboost parameter tuning with p = 75
# recipe: https://www.kaggle.com/c/bnp-paribas-cardif-claims-management/forums/t/19083/best-practices-for-parameter-tuning-on-models/108783#post108783

ratio = float(np.sum(y == 1)) / np.sum(y==0)
# Initial parameters for the parameter exploration
# clf = xgb.XGBClassifier(missing=9999999999,
#                 max_depth = 10,
#                 n_estimators=1000,
#                 learning_rate=0.1, 
#                 nthread=4,
#                 subsample=1.0,
#                 colsample_bytree=0.5,
#                 min_child_weight = 5,
#                 scale_pos_weight = ratio,
#                 seed=4242)

# gives : validation_1-auc:0.845644
# max_depth=8 -> validation_1-auc:0.846341
# max_depth=6 -> validation_1-auc:0.845738
# max_depth=7 -> validation_1-auc:0.846504
# subsample=0.8 -> validation_1-auc:0.844440
# subsample=0.9 -> validation_1-auc:0.844746
# subsample=1.0,  min_child_weight=8 -> validation_1-auc:0.843393
# min_child_weight=3 -> validation_1-auc:0.848534
# min_child_weight=1 -> validation_1-auc:0.846311
# min_child_weight=4 -> validation_1-auc:0.847994
# min_child_weight=2 -> validation_1-auc:0.847934
# min_child_weight=3, colsample_bytree=0.3 -> validation_1-auc:0.847498
# colsample_bytree=0.7 -> validation_1-auc:0.846984
# colsample_bytree=0.6 -> validation_1-auc:0.847856
# colsample_bytree=0.5, learning_rate=0.05 -> validation_1-auc:0.847347
# max_depth=8 -> validation_1-auc:0.847352
# learning_rate = 0.07 -> validation_1-auc:0.847432
# learning_rate = 0.2 -> validation_1-auc:0.846444
# learning_rate = 0.15 -> validation_1-auc:0.846889
# learning_rate = 0.09 -> validation_1-auc:0.846680
# learning_rate = 0.1 -> validation_1-auc:0.847432
# max_depth=7 -> validation_1-auc:0.848534
# learning_rate = 0.05 -> validation_1-auc:0.847347
# 


X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, stratify=y, test_size=0.3)

xgbooster = xgb.XGBClassifier(missing=np.nan,
                max_depth = 5,
                n_estimators=1000,
                learning_rate=0.1, 
                nthread=4,
                subsample=1.0,
                colsample_bytree=0.5,
                min_child_weight = 3,
                scale_pos_weight = ratio,
                reg_alpha=0.03,
                seed=234)
scorestrain=[]
scorescv=[]
iterations=[]


for i in range(1000,len(X_train),3000) :                
    print(i)
    xgbooster.fit(X_train[:i], y_train[:i], early_stopping_rounds=50, eval_metric="auc",
    eval_set=[(X_test, y_test)])
    scorestrain.append(roc_auc_score(y_train[:i], xgbooster.predict_proba(X_train[:i], ntree_limit=xgbooster.best_iteration)[:,1]))
    scorescv.append(roc_auc_score(y_test,xgbooster.predict_proba(X_test,ntree_limit=xgbooster.best_iteration)[:,1]))
    iterations.append(i)
  
"""

    print(pd.Series(X_testpred[:,1]).describe())


submission = pd.DataFrame({"ID":test.index, "TARGET":y_pred[:,1]})
submission.to_csv("submission.csv", index=False)


"""

import matplotlib.pyplot as plt

plt.figure(figsize=(15,10))
plt.plot(iterations,scorestrain,'r')
plt.plot(iterations,scorescv, 'b')
plt.ylim(0.8,0.9)
plt.xlim(1000,55000)
plt.xlabel('# training examples')
plt.ylabel('AUC')
plt.legend(['Training Set','CV set'],loc='lower right')



