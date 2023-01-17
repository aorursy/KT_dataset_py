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
import time

import matplotlib.pyplot as plt
train_file = "../input/minor-project-2020/train.csv"

test_file = "../input/minor-project-2020/test.csv"



df = pd.read_csv(train_file,header=0, delimiter=",")



df.head()
df_test = pd.read_csv(test_file,header=0, delimiter=",")



df_test.head()
df.info()
df.describe()
df.isna().sum()
df['target'].value_counts()
# from scipy import stats

# z = np.abs(stats.zscore(df))

# print(z)
# threshold = 5

# print(np.where(z > threshold))



# r,c = np.where(z > threshold)
# print(len(r))

# print(len(np.unique(r)))
#ADD OUTLIER REMOVAL CODE

# print(len(df))

# df = df.drop(np.unique(r))

# print(len(df))
#ADD FEATURE SELECTION CODE

X = df.drop(['id','target'], axis=1)

y = df['target']
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

X_scaler = scaler.fit(X)

# from sklearn import preprocessing



# min_max_scaler = preprocessing.MinMaxScaler()

# X_minmax = min_max_scaler.fit_transform(X)

# X_minmax = X
# from sklearn.model_selection import train_test_split



# X_train,X_val,y_train,y_val = train_test_split(X_scaler, y, test_size=0.2, random_state = 1212)
# (y_train).value_counts()
# (y_val).value_counts()
from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import RandomUnderSampler

from imblearn.pipeline import Pipeline



over = SMOTE()

under = RandomUnderSampler()



# steps = [('o', over), ('u', under)]

# pipeline = Pipeline(steps=steps)



print(y.value_counts())



# X, y = pipeline.fit_resample(X, y)

X, y = under.fit_resample(X, y)



print(y.value_counts())
# MAX_TREE_DEPTH = 8

# TREE_METHOD = 'gpu_hist'

# ITERATIONS = 1000

# SUBSAMPLE = 0.6

# REGULARIZATION = 0.1

# GAMMA = 0.3

# POS_WEIGHT = 1

# EARLY_STOP = 10



# #params = {'tree_method': TREE_METHOD, 'max_depth': MAX_TREE_DEPTH, 'alpha': REGULARIZATION, 'gamma': GAMMA, 'subsample': SUBSAMPLE, 'scale_pos_weight': POS_WEIGHT, 'learning_rate': 0.05, 'silent': 1, 'objective':'binary:logistic', 'eval_metric': 'auc', 'n_gpus': 1}

# params = {'tree_method': TREE_METHOD}
# from xgboost import XGBClassifier

# import xgboost as xgb
# dtrain = xgb.DMatrix(X_train, y_train)



# dval = xgb.DMatrix(X_val, y_val)
# print("Training with Single GPU ...")

# num_round = 1000

# param = {}

# param['objective'] = 'binary:hinge'

# param['eval_metric'] = 'auc'

# param['silent'] = 1

# param['tree_method'] = 'gpu_hist'

# # param['num_class'] = 2

# tmp = time.time()

# gpu_res = {}

# model = xgb.train(param, dtrain, num_round, evals=[(dval, "test")], evals_result=gpu_res)

# gpu_time = time.time() - tmp

# print("GPU Training Time: %s seconds" % (str(gpu_time)))
# y_pred = model.predict(dval)
# from sklearn.ensemble import RandomForestClassifier
# rf = RandomForestClassifier(n_estimators=200)



# tmp = time.time()

# rf.fit(X_train, y_train)

# gpu_time = time.time() - tmp

# print("Training Time: %s seconds" % (str(gpu_time)))
# print(rf.score(X_val, y_val))
# y_pred = rf.predict(X_val)
# from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text

# from sklearn.model_selection import GridSearchCV
# parameters = {'criterion': ("gini", "entropy")}



# dt_cv = DecisionTreeClassifier()



# clf = GridSearchCV(dt_cv, parameters, verbose=1)
# %%time

# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_val)
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.model_selection import StratifiedKFold

from xgboost import XGBClassifier
params_classifier = { 

    "learning_rate":0.02, 

    "n_estimators": 400, 

    "objective":'binary:logistic', 

    'tree_method':'gpu_hist', 

    'predictor':'gpu_predictor'

}



xgb = XGBClassifier(**params_classifier)
params = {

        'min_child_weight': [1, 5, 10],

        'gamma': [0.5, 1, 1.5, 2, 5],

        'subsample': [0.6, 0.8, 1.0],

        'colsample_bytree': [0.6, 0.8, 1.0]

        }
folds = 5

param_comb = 4



skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)



random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=skf.split(X,y), verbose=3, random_state=1001 )

# gsearch = GridSearchCV(estimator = xgb, param_grid = params, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
tmp = time.time()



random_search.fit(X, y)

# gsearch.fit(X,y)



gpu_time = time.time() - tmp

print("Training Time: %s seconds" % (str(gpu_time)))
# gsearch.grid_scores_, gsearch.best_params_, gsearch.best_score_
print('\n All results:')

print(random_search.cv_results_)

print('\n Best estimator:')

print(random_search.best_estimator_)

print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))

print(random_search.best_score_ * 2 - 1)

print('\n Best hyperparameters:')

print(random_search.best_params_)
# for i in range(len(y_pred)):

#     if y_pred[i] >= 0.5:

#         y_pred[i] = 0

#     else:

#         y_pred[i] = 1



# y_pred

model = random_search
# from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix



# print("Confusion Matrix: ")



# print(confusion_matrix(y_val, y_pred))
# plot_confusion_matrix(model, X_val, y_val, cmap = plt.cm.Blues)
# print("Classification Report: ")

# print(classification_report(y_val, y_pred))
# from sklearn.metrics import roc_curve, auc

# plt.style.use('seaborn-pastel')



# FPR, TPR, _ = roc_curve(y_val, y_pred)

# ROC_AUC = auc(FPR, TPR)

# print (ROC_AUC)



# plt.figure(figsize =[11,9])

# plt.plot(FPR, TPR, label= 'ROC curve(area = %0.2f)'%ROC_AUC, linewidth= 4)

# plt.plot([0,1],[0,1], 'k--', linewidth = 4)

# plt.xlim([0.0,1.0])

# plt.ylim([0.0,1.05])

# plt.xlabel('False Positive Rate', fontsize = 18)

# plt.ylabel('True Positive Rate', fontsize = 18)

# plt.title('ROC for Titanic survivors', fontsize= 18)

# plt.show()
# parameters = {'criterion': ("gini", "entropy")}



# dt_cv = DecisionTreeClassifier()



# clf = GridSearchCV(dt_cv, parameters, verbose=1)
# clf.fit(X_minmax, y)
test = df_test.drop('id',axis=1)



test.head()
# Use with XGB

# Y = model.predict(xgb.DMatrix(test))
test = scaler.fit_transform(test)



# Y = model.predict(test)

Y = model.predict_proba(test)
Y
pd.DataFrame(Y).value_counts()
out = df_test['id']

out = pd.DataFrame(out, columns=['id'])

out['target'] = Y[:,1]



# out.target = out.target.astype('int64')



out.head()
out.to_csv("./submit.csv",index=False)