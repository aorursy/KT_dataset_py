import csv as csv 
import numpy as np
import pandas as pd
import pylab as P
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import xgboost as xgb
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn import svm, datasets, metrics

plt.style.use('ggplot')
%matplotlib inline
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# remove constant columns
remove = []
for col in train.columns:
    if train[col].std() == 0:
        remove.append(col)

train.drop(remove, axis=1, inplace=True)
test.drop(remove, axis=1, inplace=True)

# remove duplicated columns
remove = []
c = train.columns
for i in range(len(c)-1):
    v = train[c[i]].values
    for j in range(i+1,len(c)):
        if np.array_equal(v,train[c[j]].values):
            remove.append(c[j])

train.drop(remove, axis=1, inplace=True)
test.drop(remove, axis=1, inplace=True)
y_train = train['TARGET'].values
X_train = train.drop(['ID','TARGET'], axis=1).values

id_test = test['ID']
X_test = test.drop(['ID'], axis=1).values

# length of dataset
len_train = len(X_train)
len_test  = len(X_test)

X_fit, X_eval, y_fit, y_eval=train_test_split(X_train, y_train, random_state=0, test_size=0.3)
xgtrain = xgb.DMatrix(X_fit, label=y_fit)
xgval = xgb.DMatrix(X_eval, label=y_eval)
xgtest = xgb.DMatrix(X_test)
# Number of boosting iterations.
num_round = 10000
params   = {'eta':0.03, 
            'objective':'binary:logistic',
            'booster':'gbtree',
            'eval_metric':'auc',
            #'min_child_weight':50,
            #'scale_pos_weight':50,      #For unbalanced classes
            'subsample':0.95,
            'n_estimators':350,
            'colsample_bytree':0.85,
            'max_depth':5,
            #'nfold':5,
            'nthread':8,
            'seed':4242,
            'missing':np.nan,
            'show_stdv':False
         }

evallist = [(xgtrain, 'train'), (xgval, 'val')]

model = xgb.train(dtrain=xgtrain, evals=evallist, params=params, num_boost_round=num_round, 
                  early_stopping_rounds=50)
#ROC Plot
preds_train = model.predict(xgtrain, ntree_limit=model.best_iteration)
preds_eval = model.predict(xgval, ntree_limit=model.best_iteration)


false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_fit, preds_train)
false_positive_rate2, true_positive_rate2, thresholds2 = metrics.roc_curve(y_eval, preds_eval)

roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc2 = auc(false_positive_rate2, true_positive_rate2)
#print roc_auc
#print roc_auc2

#Plotting
plt.rcParams['figure.figsize'] = (8.0, 8.0)
plt.title("Classifier's ROC")
plt.plot(false_positive_rate, true_positive_rate, 'g',label='Train = %0.4f'% roc_auc)
plt.plot(false_positive_rate2, true_positive_rate2, 'y',label='Eval = %0.4f'% roc_auc2)
plt.plot([0,1],[0,1],'r--', label='Random choice')

plt.legend(loc='lower right')
plt.xlim([0,1])
plt.ylim([0,1.05])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
preds = model.predict(xgtest, ntree_limit=model.best_iteration)

test_aux = pd.read_csv('test.csv')
result = pd.DataFrame({"Id": id_test, 'TARGET': preds})

#result.to_csv("submission.csv", index=False)

#print "Training parameters"
#print "xgtrain : ", xgtrain.num_row(), xgtrain.num_col()
#print "xgval : ", xgval.num_row(), xgval.num_col()
#print "xgtest : ", xgtest.num_row(), xgtest.num_col()

#print "preds.shape ", preds.shape
#print "np.max preds = ", np.max(preds)
#print "np.min preds = ", np.min(preds)
#print "np.avg preds = ", np.average(preds)