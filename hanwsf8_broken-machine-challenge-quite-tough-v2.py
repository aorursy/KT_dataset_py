import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import gc

import time

import warnings

warnings.simplefilter(action = 'ignore', category = FutureWarning)
from sklearn.metrics import roc_auc_score, precision_score, recall_score, roc_curve, f1_score, confusion_matrix, accuracy_score

from sklearn.model_selection import KFold, StratifiedKFold

import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

from joblib import dump, load

file_path = '../input/the-broken-machine/'

model_path = '../input/broke-machine-model/'

# file_path = './the-broken-machine/'
xtrain = pd.read_csv(file_path + 'xtrain.csv')

ytrain = pd.read_csv(file_path + 'ytrain.csv')

print(xtrain.shape)

print(ytrain.shape)

xtrain.head()
print("1 ratio is：",ytrain[ytrain==1].count()/len(ytrain))

#那么accuracy小于70%是没有意义的
#check data

pd.set_option('display.max_columns', None)

xtrain.describe()
#Check missing data

all_data_na = (xtrain.isnull().sum() / len(xtrain)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

f, ax = plt.subplots(figsize=(8, 6))

plt.xticks(rotation='90')

sns.barplot(x=all_data_na.index, y=all_data_na)

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data by feature', fontsize=15)
#EDA NA processing,lgb doesn't need na processing

for col in xtrain.columns:

    xtrain[col] = xtrain[col].fillna(xtrain[col].mode()[0])#用众数

xtrain.describe()
xtrain.head()
# EDA skew

xtrain.skew(axis=0).sort_values(ascending=False)

#发现37数值异常
xtrain['37'].hist()
xtrain['37']=xtrain['37'].apply(lambda x:200 if x>100 else x) #处理37号
#EDA No. 37

from scipy import stats

from scipy.stats import norm, skew #for some statistics

def check_skewness(col):

    sns.distplot(xtrain[col] , fit=norm);

    fig = plt.figure()

#     res = stats.probplot(xtrain[col], plot=plt) #probplot不能显示，如果是index整数就可显示

    # Get the fitted parameters used by the function

    (mu, sigma) = norm.fit(xtrain[col])

    print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

    

check_skewness(['37']) 
# check unique value

for i in xtrain.columns:

    print(i,": ",len(xtrain[i].unique()))
#Feature distribution

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

h = .2  # step size in the mesh



x_min, x_max = xtrain.iloc[0:1000, 33].min() - .5, xtrain.iloc[0:1000, 33].max() + .5

y_min, y_max = xtrain.iloc[0:1000, 36].min() - .5, xtrain.iloc[0:1000, 36].max() + .5

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),

                     np.arange(y_min, y_max, h))

# just plot the dataset first

cm = plt.cm.RdBu

cm_bright = ListedColormap(['#FF0000', '#0000FF'])

ax = plt.subplot()

ax.scatter(xtrain.iloc[0:1000, 33], xtrain.iloc[0:1000, 36], c=list(ytrain.iloc[0:1000,0]),cmap=cm_bright,

           edgecolors='k')

ax.set_xlim(xx.min(), xx.max())

ax.set_ylim(yy.min(), yy.max())

ax.set_xticks(())

ax.set_yticks(())
xtrain['1'].hist()
#corelation

corrmat = xtrain.corr()

corrmat
corrmat[corrmat>0.01].count()

#No clear corelation
# plt.figure(figsize=(10,10))

# g = sns.heatmap(train_data[top_corr_features].corr(),annot=True,cmap="RdYlGn")



# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.check_cv.html#sklearn.model_selection.check_cv

# from sklearn.model_selection import check_cv

# cv = check_cv(3, xtrain, ytrain, classifier=True)
from sklearn.model_selection import KFold, cross_val_score, train_test_split

X_train, X_test, y_train, y_test=train_test_split(xtrain[0:-10000],ytrain[0:-10000], test_size=0.2, random_state=3)

# gc.collect()  

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)
X_train.head()
#Basic classification model 123

'''from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB 

from sklearn.ensemble import RandomForestClassifier

import numpy as np



np.random.seed(123)



clf1 = LogisticRegression(n_jobs=-1)

clf2 = RandomForestClassifier(n_jobs=-1)

clf3 = GaussianNB()



print('5-fold cross validation:\n')



for clf, label in zip([clf1, clf2, clf3], ['Logistic Regression', 'Random Forest', 'naive Bayes']):



    scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')

    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))'''



# Accuracy: 0.69 (+/- 0.00) [Logistic Regression]

# Accuracy: 0.67 (+/- 0.00) [Random Forest]

# Accuracy: 0.69 (+/- 0.00) [naive Bayes]
'''clf1.fit(X_train, y_train)

preds2 = clf1.predict(X_test)

print('scores {:.5f} accuracy_score on the test set.'.format(accuracy_score(y_test, preds2)))

print('scores {:.5f} AUC ROC on the test set.'.format(roc_auc_score(y_test, preds2)))

print('scores {:.5f} precision_score on the test set.'.format(precision_score(y_test, preds2)))

print('scores {:.5f} recall_score on the test set.'.format(recall_score(y_test, preds2)))

print('scores {:.5f} f1_score on the test set.'.format(f1_score(y_test, preds2,'binary')))'''



'''scores 0.69342 accuracy_score on the test set.

scores 0.50000 AUC ROC on the test set.

scores 0.00000 precision_score on the test set.

scores 0.00000 recall_score on the test set.

scores 0.00000 f1_score on the test set.'''
'''clf3.fit(X_train, y_train)

preds2 = clf3.predict(X_test)

print('scores {:.5f} accuracy_score on the test set.'.format(accuracy_score(y_test, preds2)))

print('scores {:.5f} AUC ROC on the test set.'.format(roc_auc_score(y_test, preds2)))

print('scores {:.5f} precision_score on the test set.'.format(precision_score(y_test, preds2)))

print('scores {:.5f} recall_score on the test set.'.format(recall_score(y_test, preds2)))

print('scores {:.5f} f1_score on the test set.'.format(f1_score(y_test, preds2,'binary')))

confusion_matrix(y_test, preds2, labels=None, sample_weight=None)'''

'''scores 0.69342 accuracy_score on the test set.

scores 0.50000 AUC ROC on the test set.

scores 0.00000 precision_score on the test set.

scores 0.00000 recall_score on the test set.

scores 0.00000 f1_score on the test set.'''
'''clf2.fit(X_train, y_train)

preds2 = clf2.predict(X_test)

print('scores {:.5f} accuracy_score on the test set.'.format(accuracy_score(y_test, preds2)))

print('scores {:.5f} AUC ROC on the test set.'.format(roc_auc_score(y_test, preds2)))

print('scores {:.5f} precision_score on the test set.'.format(precision_score(y_test, preds2)))

print('scores {:.5f} recall_score on the test set.'.format(recall_score(y_test, preds2)))

print('scores {:.5f} f1_score on the test set.'.format(f1_score(y_test, preds2,'binary')))

confusion_matrix(y_test, preds2, labels=None, sample_weight=None)'''

'''scores 0.67264 accuracy_score on the test set.

scores 0.50607 AUC ROC on the test set.

scores 0.34509 precision_score on the test set.

scores 0.07550 recall_score on the test set.

scores 0.12389 f1_score on the test set.'''
# Plot feature importance

'''feature_importance = clf2.feature_importances_

# make importances relative to max importance

feature_importance = 100.0 * (feature_importance / feature_importance.max())

sorted_idx = np.argsort(feature_importance)

pos = np.arange(sorted_idx.shape[0]) + .5

plt.figure(figsize=(8, 16))

plt.subplot(1, 2, 2)

plt.barh(pos, feature_importance[sorted_idx], align='center')

plt.yticks(pos, xtrain.columns[sorted_idx])

plt.xlabel('Relative Importance')

plt.title('Variable Importance')

plt.show()'''
'''dump(clf1, '1LogisticRegression.joblib') 

dump(clf2, '2RandomForestClassifier.joblib') 

dump(clf3, '3GaussianNB.joblib') '''
#ensemble 4,GBM needs too much time

'''from sklearn.ensemble import GradientBoostingClassifier

clf4 = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,

     max_depth=5, random_state=0,validation_fraction=0.1,n_iter_no_change=30).fit(X_train, y_train)

clf4.score(X_test, y_test) 



preds2 = clf4.predict(X_test)

print('scores {:.5f} accuracy_score on the test set.'.format(accuracy_score(y_test, preds2)))

print('scores {:.5f} AUC ROC on the test set.'.format(roc_auc_score(y_test, preds2)))

print('scores {:.5f} precision_score on the test set.'.format(precision_score(y_test, preds2)))

print('scores {:.5f} recall_score on the test set.'.format(recall_score(y_test, preds2)))

print('scores {:.5f} f1_score on the test set.'.format(f1_score(y_test, preds2,'binary')))

confusion_matrix(y_test, preds2, labels=None, sample_weight=None)'''

# 众数、validation有所提升

'''scores 0.74246 accuracy_score on the test set.

scores 0.60915 AUC ROC on the test set.

scores 0.71664 precision_score on the test set.

scores 0.26454 recall_score on the test set.

scores 0.38643 f1_score on the test set.'''
'''confusion_matrix(y_test, preds2, labels=None, sample_weight=None)'''
'''dump(clf4, '4GradientBoostingClassifier.joblib') '''
#5

# https://blog.csdn.net/linxid/article/details/80785131?utm_source=blogxgwz7

# http://lightgbm.apachecn.org/#/docs/8

# clf5 = lgb.LGBMClassifier(learning_rate=0.05,n_estimators=10000,num_leaves=100,objective='binary', metrics='auc',random_state=50，n_jobs=-1)

# clf5.fit(X_train, y_train, eval_set=(xtrain[-10000:-1], ytrain[-10000:-1]),

# #         eval_metric='auc',#缺省用logloss

#          n_jobs=-1，

#         early_stopping_rounds=50)

# clf5.score(X_test, y_test)

# preds2 = clf5.predict(X_test)

# print('scores {:.5f} accuracy_score on the test set.'.format(accuracy_score(y_test, preds2)))

# print('scores {:.5f} AUC ROC on the test set.'.format(roc_auc_score(y_test, preds2)))

# print('scores {:.5f} precision_score on the test set.'.format(precision_score(y_test, preds2)))

# print('scores {:.5f} recall_score on the test set.'.format(recall_score(y_test, preds2)))

# print('scores {:.5f} f1_score on the test set.'.format(f1_score(y_test, preds2,'binary')))
# https://scikit-learn.org/stable/modules/model_evaluation.html#model-evaluation

#learning_rate=0.01,n_estimators=10000,num_leaves=100后效果明显

'''scores 0.75272 accuracy_score on the test set.

scores 0.62243 AUC ROC on the test set.

scores 0.75593 precision_score on the test set.

scores 0.28565 recall_score on the test set.

scores 0.41462 f1_score on the test set.'''
# from sklearn.metrics import roc_curve

# #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve

# fpr,tpr,thresholds = roc_curve(y_test,clf5.predict_proba(X_test)[:,1],pos_label=1)

# plt.plot(fpr,tpr,linewidth=2,label="ROC")#,marker = 'o')

# plt.xlabel("false presitive rate")

# plt.ylabel("true presitive rate")

# plt.ylim(0,1.05)

# plt.xlim(0,1.05)

# plt.legend(loc=4)#图例的位置

# plt.show()

# from sklearn.metrics import precision_recall_curve

# from sklearn.utils.fixes import signature

# from sklearn.metrics import average_precision_score

# average_precision = average_precision_score(y_test, preds2)

# precision, recall, _ = precision_recall_curve(y_test,preds2)

# step_kwargs = ({'step': 'post'}

#                if 'step' in signature(plt.fill_between).parameters

#                else {})

# plt.step(recall, precision, color='b', alpha=0.2,

#          where='post')

# plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)



# plt.xlabel('Recall')

# plt.ylabel('Precision')

# plt.ylim([0.0, 1.05])

# plt.xlim([0.0, 1.0])

# plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(

#           average_precision))
'''confusion_matrix(y_test, preds2, labels=None, sample_weight=None)'''

# array([[120512,   4477],

#        [ 41963,  13048]])
# 保存模型

# clf5.booster_.save_model('lgbmodel.txt')

# 载入模型

# clf5 = lgb.Booster(model_file='5lgbmodel.txt')

# pred2 = lgbmodel.predict(X_test) #Booster出来的是prob
#6

'''from catboost import CatBoostClassifier, Pool, cv

clf6 = CatBoostClassifier(

    custom_loss=['Accuracy'],#这里是提供个列表

    random_seed=42,

    logging_level='Silent')

clf6.fit(

    X_train, y_train,

    #cat_features=list(categorical_features_indices),

    eval_set=(xtrain[-10000:-1], ytrain[-10000:-1]),

    logging_level='Verbose',  # you can uncomment this for text output

    plot=True,

     early_stopping_rounds=50

)

preds2 = clf6.predict(X_test)

print('scores {:.5f} accuracy_score on the test set.'.format(accuracy_score(y_test, preds2)))

print('scores {:.5f} AUC ROC on the test set.'.format(roc_auc_score(y_test, preds2)))

print('scores {:.5f} precision_score on the test set.'.format(precision_score(y_test, preds2)))

print('scores {:.5f} recall_score on the test set.'.format(recall_score(y_test, preds2)))

print('scores {:.5f} f1_score on the test set.'.format(f1_score(y_test, preds2,'binary')))

confusion_matrix(y_test, preds2, labels=None, sample_weight=None)

dump(clf6, '6catboost.joblib') '''
# modelc.score(X_test, y_test)

#没有eval_set

'''scores 0.74396 f1_score on the test set.

scores 0.60772 AUC ROC on the test set.

scores 0.73018 precision_score on the test set.

scores 0.25728 recall_score on the test set.

scores 0.38049 f1_score on the test set.'''

#7  xgboost is not good

# from xgboost import XGBClassifier

# xgbc = XGBClassifier()

# xgbc.fit(X_train, y_train)

# xgbc.score(X_test, y_test)

# #0.742跟GBM差不多

# preds2 = xgbc.predict(X_test)

'''print('scores {:.5f} f1_score on the test set.'.format(accuracy_score(y_test, preds2)))

print('scores {:.5f} AUC ROC on the test set.'.format(roc_auc_score(y_test, preds2)))

print('scores {:.5f} precision_score on the test set.'.format(precision_score(y_test, preds2)))

print('scores {:.5f} recall_score on the test set.'.format(recall_score(y_test, preds2)))

print('scores {:.5f} f1_score on the test set.'.format(f1_score(y_test, preds2,'binary')))  '''



'''scores 0.69726 f1_score on the test set.

scores 0.50586 AUC ROC on the test set.

scores 0.76701 precision_score on the test set.

scores 0.01352 recall_score on the test set.

scores 0.02658 f1_score on the test set.'''
# #准备lgb数据集，这个是不同的

# train_set=lgb.Dataset(X_train, label=y_train)

# valid_set=lgb.Dataset(xtrain[-10000:-1], ytrain[-10000:-1], reference=train_set)
# import csv

# from hyperopt import STATUS_OK

# from timeit import default_timer as timer

# import time

# MAX_EVALS = 10000 #这里定义了2轮，后面可以在验证了后可以改大

# N_FOLDS = 3#这里定义了2轮，后面可以在验证了后可以改大

# def objective(params, n_folds = N_FOLDS):

#     """Objective function for Gradient Boosting Machine Hyperparameter Optimization"""

    

#     # Keep track of evals

#     global ITERATION

    

#     ITERATION += 1

#     print('现在ITERATION+1为: ',ITERATION)

#     print (time.asctime( time.localtime(time.time()) ))

#     #encoding_type = params['encoding']

    

#     '''# Handle the encoding

#     if encoding_type == 'one_hot':

#         train_set = oh_train_set

#     elif encoding_type == 'label':

#         train_set = le_train_set'''

    

#     #del params['encoding']

    

#     # Retrieve the subsample

#     subsample = params['boosting_type'].get('subsample', 1.0)

    

#     # Extract the boosting type and subsample to top level keys

#     params['boosting_type'] = params['boosting_type']['boosting_type']

#     params['subsample'] = subsample

    

#     # Make sure parameters that need to be integers are integers

#     for parameter_name in ['num_leaves', 'subsample_for_bin', 'min_child_samples']:

#         params[parameter_name] = int(params[parameter_name])

    

#     start = timer()

    

#     # Perform n_folds cross validation

#     cv_results = lgb.cv(params, train_set, nfold = n_folds, 

#                         early_stopping_rounds = 50, metrics = 'auc', seed = 50,verbose_eval=1,show_stdv=True)#,num_boost_round = 1000, ) ##gpu

    

#     run_time = timer() - start

    

#     # Extract the best score

#     best_score = np.max(cv_results['auc-mean'])

    

#     # Loss must be minimized

#     loss = 1 - best_score

#     print('loss is:', loss)

#     # Boosting rounds that returned the highest cv score

#     n_estimators = int(np.argmax(cv_results['auc-mean']) + 1)

#     print("n_estimators: ",n_estimators)

#     #params['encoding'] = encoding_type

    

#     if ITERATION % 100 == 0: #取余数，每100返回一个显示

#         # Display the information

#         display('Iteration {}: {} Fold CV AUC ROC {:.5f}'.format(ITERATION, N_FOLDS, best_score))



#     # Write to the csv file ('a' means append)

#     of_connection = open(out_file, 'a')

#     writer = csv.writer(of_connection)

#     writer.writerow([loss, params, ITERATION, n_estimators, run_time, best_score])

#     of_connection.close()

    

#     # Dictionary with information for evaluation

#     return {'loss': loss, 'params': params, 'iteration': ITERATION,

#             'estimators': n_estimators, 

#             'train_time': run_time, 'status': STATUS_OK}
# from hyperopt import hp

# from hyperopt.pyll.stochastic import sample
# .60772 

# space = {

#    # 'encoding': hp.choice('encoding', ['one_hot', 'label']),

#     'class_weight': hp.choice('class_weight', [None, 'balanced']),

#     'boosting_type': hp.choice('boosting_type', 

#                                             [{'boosting_type': 'gbdt', 'subsample': hp.uniform('gdbt_subsample', 0.5, 1)}, 

#                                              {'boosting_type': 'dart', 'subsample': hp.uniform('dart_subsample', 0.5, 1)},

#                                              {'boosting_type': 'goss', 'subsample': 1.0}]),

#     'num_leaves': hp.quniform('num_leaves', 30, 200, 2),

#     'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.2)),

#     'subsample_for_bin': hp.quniform('subsample_for_bin', 100000, 800000, 100000),

#     'min_child_samples': hp.quniform('min_child_samples', 10, 60, 2),

#     'reg_alpha': hp.uniform('reg_alpha', 0.0, 0.8),

#     'reg_lambda': hp.uniform('reg_lambda', 0.0, 0.8),

#     'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0)

# }
# from hyperopt import tpe

# tpe_algorithm = tpe.suggest

# from hyperopt import Trials

# trials = Trials()



# # File to save first results

# out_file = 'results/gbm_results_kaggle.csv'

# of_connection = open(out_file, 'a')#这里写错了，应该是a，w就直接覆盖了

# writer = csv.writer(of_connection)

# # Write the headers to the file

# writer.writerow(['loss', 'params', 'iteration', 'estimators', 'time', 'ROC AUC'])

# of_connection.close()
# from hyperopt import fmin

# # https://hyperopt.github.io/hyperopt/

# # https://github.com/hyperopt/hyperopt/wiki/FMin

# # %%capture

# # Global variable

# global  ITERATION

# #ITERATION=MAX_EVALS+1

# #print(ITERATION)

# ITERATION = 0

# # Run optimization 运行优化##################这里需要从84次开始，comment

# best = fmin(fn = objective, space = space, algo = tpe.suggest, 

#             max_evals = MAX_EVALS, trials = trials)#, verbose = 1)
# best
# 一种方法Sort the trials with lowest loss (highest AUC) first 

# trials_results = sorted(trials.results, key = lambda x: x['loss'])

# trials_results[:10]



'''# 另一种OPTION 从文件中等到

results = pd.read_csv('results/gbm_results_kaggle.csv')



# Sort with best scores on top and reset index for slicing

results.sort_values('loss', ascending = True, inplace = True)

results.reset_index(inplace = True, drop = True)

results.head()'''
'''# 只有读文件的时候采用导 OPTIN

import ast

# Convert from a string to a dictionary

ast.literal_eval(results.loc[0, 'params'])'''
# trials_results
# Extract the ideal number of estimators and hyperparameters这是从csv trail文件中读取参数

'''best_bayes_estimators = int(results.loc[0, 'estimators'])

best_bayes_params = ast.literal_eval(results.loc[0, 'params']).copy()'''

#这是直接从前面的内存中读取最好参数

'''best_bayes_estimators = int(trials_results[0][ 'estimators'])

best_bayes_params = trials_results[0]['params']



# Re-create the best model and train on the training data

best_bayes_model = lgb.LGBMClassifier(n_estimators=best_bayes_estimators, 

                                      n_jobs = -1, 

                                      objective = 'binary', 

                                      random_state = 50, 

                                      verbose_eval =1,

                                      **best_bayes_params)#用**来取得超参

best_bayes_model.fit(X_train, y_train)

best_bayes_model.score(X_test, y_test)'''

#这20个回合应该是比较少，出来的结果比传统的或缺省的还要差

#0.6394277777777778
# # Continue training继续训练，会把之前的trail文件覆盖掉，因为这次是从linux是直接开始的？下次停下来可以先复制一份

# ITERATION = MAX_EVALS + 1#上个训练周期MAX_EVALS 结束

# #ITERATION = 84 #上次到这里

# # Set more evaluations

# MAX_EVALS = 1000  #10个半小时，880个iteration



# #from IPython.display import display

# #%%capture --no-display #不能用



# # Use the same trials object to keep training

# best = fmin(fn = objective, space = space, algo = tpe.suggest, 

#             max_evals = MAX_EVALS, trials = trials, verbose = 0, #trials = bayes_trials

#             rstate = np.random.RandomState(50))
# import ast

# # Sort the trials with lowest loss (highest AUC) first

# bayes_trials_results = sorted(trials.results, key = lambda x: x['loss'])#bayes_trials.result

# bayes_trials_results[:2]



'''results = pd.read_csv('results/gbm_results_kaggle.csv')



# Sort values with best on top and reset index for slicing

results.sort_values('loss', ascending = True, inplace = True)

results.reset_index(inplace = True, drop = True)

results.head()



# Extract the ideal number of estimators and hyperparameters

best_bayes_estimators = int(results.loc[0, 'estimators'])

best_bayes_params = ast.literal_eval(results.loc[0, 'params']).copy()



# Re-create the best model and train on the training data

best_bayes_model = lgb.LGBMClassifier(n_estimators=best_bayes_estimators, n_jobs = -1, 

                                       objective = 'binary', random_state = 50, **best_bayes_params)

best_bayes_model.fit(X_train, y_train)

best_bayes_model.score(X_test, y_test)

# 0.7552333333333333 #900次iteration

# Evaluate on the testing data 

preds = best_bayes_model.predict_proba(X_test)

print('The best model from Bayes optimization scores {:.5f} AUC ROC on the test set.'.format(roc_auc_score(y_test, preds)))

print('This was achieved after {} search iterations'.format(results.loc[0, 'iteration']))'''
# import ast



# results = pd.read_csv('results/gbm_results_kaggle.csv')



# # Sort values with best on top and reset index for slicing

# results.sort_values('loss', ascending = True, inplace = True)

# results.reset_index(inplace = True, drop = True)

# results.head()



# # Extract the ideal number of estimators and hyperparameters

# best_bayes_estimators = int(results.loc[0, 'estimators'])

# best_bayes_params = ast.literal_eval(results.loc[0, 'params']).copy()



# # Re-create the best model and train on the training data

# best_bayes_model = lgb.LGBMClassifier(n_estimators=best_bayes_estimators, n_jobs = -1, 

#                                        objective = 'binary', random_state = 50, **best_bayes_params)

# best_bayes_model.fit(X_train, y_train)

# best_bayes_model.score(X_test, y_test)
# Evaluate on the testing data 

# preds = best_bayes_model.predict_proba(X_test)

# preds[0:5]
clf4 = load(model_path+'4GradientBoostingClassifier.joblib') 

clf5 = lgb.Booster(model_file=model_path+'5lgbmodel.txt')

clf6 = load(model_path+'6catboost.joblib') 
# https://www.itcodemonkey.com/article/10354.html

# clf4.fit(X_train, y_train)

# val_pred4=clf4.predict(xtrain[-10000:-1])

# test_pred4=clf4.predict(X_test)

# val_pred4=pd.DataFrame(val_pred4)

# test_pred4=pd.DataFrame(test_pred4)

# clf4.fit(X_train, y_train)

test_pred4=clf4.predict_proba(X_test)

test_pred4
print(test_pred4[1].min())

print(test_pred4[1].max())

print(np.median(test_pred4[1]))
# clf5.fit(X_train, y_train) #Booster has not fit

# val_pred5=clf5.predict(xtrain[-10000:-1])

# test_pred5=clf5.predict(X_test)

# val_pred5=pd.DataFrame(val_pred5)

# test_pred5=pd.DataFrame(test_pred5)

test_pred5=clf5.predict(X_test)

test_pred5
print(test_pred5.min())

print(test_pred5.max())

print(np.median(test_pred5))
# clf6.fit(X_train, y_train) #catboost doesn't need to fit

# val_pred6=clf6.predict(xtrain[-10000:-1])

# test_pred6=clf6.predict(X_test)

# val_pred6=pd.DataFrame(val_pred6)

# test_pred6=pd.DataFrame(test_pred6)

test_pred6=clf6.predict_proba(X_test)

test_pred6
print(test_pred6[1].min())

print(test_pred6[1].max())

print(np.median(test_pred6[1]))
# pred2=test_pred4[:,1]*0.25 + test_pred5*0.5 +test_pred6[:,1]*0.25

# pred2=test_pred4[:,1]*0.15 + test_pred5*0.35 +test_pred6[:,1]*0.35

pred2=test_pred4[:,1]*0.6 + test_pred5*0.2 +test_pred6[:,1]*0.2

pred2
pred2=pd.DataFrame(pred2)[0].apply(lambda x:1 if x>0.35 else 0)

print('scores {:.5f} f1_score on the test set.'.format(accuracy_score(y_test, pred2)))

print('scores {:.5f} AUC ROC on the test set.'.format(roc_auc_score(y_test, pred2)))

print('scores {:.5f} precision_score on the test set.'.format(precision_score(y_test, pred2)))

print('scores {:.5f} recall_score on the test set.'.format(recall_score(y_test, pred2)))

print('scores {:.5f} f1_score on the test set.'.format(f1_score(y_test, pred2,'binary')))

#0.35   ######################这个还可以###################pred2=test_pred4[:,1]*0.25 + test_pred5*0.5 +test_pred6[:,1]*0.25#

# scores 0.72465 accuracy_score on the test set.

# scores 0.64574 AUC ROC on the test set.

# scores 0.56516 precision_score on the test set.

# scores 0.44175 recall_score on the test set.

# scores 0.49590 f1_score on the test set.

# array([[104881,  18548],

#        [ 30464,  24107]], dtype=int64)