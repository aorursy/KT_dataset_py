# Data manipulation
import pandas as pd
import numpy as np

# Modeling
import lightgbm as lgb

# Evaluation of the model
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.size'] = 18
%matplotlib inline

# Governing choices for search
N_FOLDS = 5
MAX_EVALS = 5
features = pd.read_csv('../input/home-credit-default-risk/application_train.csv')

# Sample 16000 rows (10000 for training, 6000 for testing)
features = features.sample(n = 16000, random_state = 42)

# Only numeric features
features = features.select_dtypes('number')

# Extract the labels
labels = np.array(features['TARGET'].astype(np.int32)).reshape((-1, ))
features = features.drop(columns = ['TARGET', 'SK_ID_CURR'])

# Split into training and testing data
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 6000, random_state = 42)

print('Train shape: ', train_features.shape)
print('Test shape: ', test_features.shape)

train_features.head()
model = lgb.LGBMClassifier(random_state=50)

# Training set
train_set = lgb.Dataset(train_features, label = train_labels)
test_set = lgb.Dataset(test_features, label = test_labels)
# Default hyperparamters
hyperparameters = model.get_params()

# Using early stopping to determine number of estimators.
del hyperparameters['n_estimators']

# Perform cross validation with early stopping
cv_results = lgb.cv(hyperparameters, train_set, num_boost_round = 10000, nfold = N_FOLDS, metrics = 'auc', 
           early_stopping_rounds = 100, verbose_eval = False, seed = 42)

# Highest score
best = cv_results['auc-mean'][-1]

# Standard deviation of best score
best_std = cv_results['auc-stdv'][-1]

print('The maximium ROC AUC in cross validation was {:.5f} with std of {:.5f}.'.format(best, best_std))
print('The ideal number of iterations was {}.'.format(len(cv_results['auc-mean'])))
model.n_estimators = len(cv_results['auc-mean'])

model.fit(train_features, train_labels)
preds = model.predict_proba(test_features)[:, 1]
baseline_auc = roc_auc_score(test_labels, preds)

print('The baseline model scores {:.5f} ROC AUC on the test set.'.format(baseline_auc))
import csv
from hyperopt import STATUS_OK
from timeit import default_timer as timer

def objective(hyperparameters):
    """Objective function for Gradient Boosting Machine Hyperparameter Optimization.
       Writes a new line to `outfile` on every iteration"""
    
    # 确定是哪一组验证集
    global ITERATION
    
    ITERATION += 1
    
    # 使用 early stopping 确定弱学习器的数量
    if 'n_estimators' in hyperparameters:
        del hyperparameters['n_estimators']
    
    # 因为设定参数 不同的boosting type需要不同的subsample参数，所以重新构造调参范围
    subsample = hyperparameters['boosting_type'].get('subsample', 1.0)
    hyperparameters['boosting_type'] = hyperparameters['boosting_type']['boosting_type']
    hyperparameters['subsample'] = subsample
    
    # 确认整数参数的情况
    for parameter_name in ['num_leaves', 'subsample_for_bin', 'min_child_samples']:
        hyperparameters[parameter_name] = int(hyperparameters[parameter_name])
        
    start = timer()
    
    cv_results = lgb.cv(hyperparameters, train_set, num_boost_round = 10000, nfold = N_FOLDS, 
                        early_stopping_rounds = 100, metrics = 'auc', seed = 50)

    run_time = timer() - start
    
    best_score = cv_results['auc-mean'][-1]
    
    # 损失函数 越小越好
    loss = 1 - best_score
    
    # 返回最优结果的 Boosting 轮数
    n_estimators = len(cv_results['auc-mean'])
    
    hyperparameters['n_estimators'] = n_estimators

    # Write to the csv file ('a' means append)
    of_connection = open(OUT_FILE, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss, hyperparameters, ITERATION, run_time, best_score])
    of_connection.close()
    
    # Dictionary with information for evaluation
    return {'loss': loss, 'hyperparameters': hyperparameters, 'iteration': ITERATION,
            'train_time': run_time, 'status': STATUS_OK}
from hyperopt import hp
from hyperopt.pyll.stochastic import sample
learning_rate = {'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.2))}
learning_rate_dist = []

# Draw 10000 samples from the learning rate domain
for _ in range(10000):
    learning_rate_dist.append(sample(learning_rate)['learning_rate'])
    
plt.figure(figsize = (8, 6))
sns.kdeplot(learning_rate_dist, color = 'red', linewidth = 2, shade = True);
plt.title('Learning Rate Distribution', size = 18); plt.xlabel('Learning Rate', size = 16); plt.ylabel('Density', size = 16);
num_leaves = {'num_leaves': hp.quniform('num_leaves', 30, 150, 1)}
num_leaves_dist = []

for _ in range(10000):
    num_leaves_dist.append(sample(num_leaves)['num_leaves'])
    
plt.figure(figsize = (8, 6))
sns.kdeplot(num_leaves_dist, linewidth = 2, shade = True);
plt.title('Number of Leaves Distribution', size = 18); plt.xlabel('Number of Leaves', size = 16); plt.ylabel('Density', size = 16);
boosting_type = {'boosting_type': hp.choice('boosting_type', 
                                            [{'boosting_type': 'gbdt', 'subsample': hp.uniform('subsample', 0.5, 1)}, 
                                             {'boosting_type': 'dart', 'subsample': hp.uniform('subsample', 0.5, 1)},
                                             {'boosting_type': 'goss', 'subsample': 1.0}])}

hyperparams = sample(boosting_type)
hyperparams
space = {
    'boosting_type': hp.choice('boosting_type', 
                                            [{'boosting_type': 'gbdt', 'subsample': hp.uniform('gdbt_subsample', 0.5, 1)}, 
                                             {'boosting_type': 'dart', 'subsample': hp.uniform('dart_subsample', 0.5, 1)},
                                             {'boosting_type': 'goss', 'subsample': 1.0}]),
    'num_leaves': hp.quniform('num_leaves', 20, 150, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.5)),
    'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
    'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0),
    'is_unbalance': hp.choice('is_unbalance', [True, False]),
}
x = sample(space)

# 因为之前对 相互影响的参数进行了特殊设置，现在也需要特殊处理一下，让整个参数处于一个层级
subsample = x['boosting_type'].get('subsample', 1.0)
x['boosting_type'] = x['boosting_type']['boosting_type']
x['subsample'] = subsample

x
x = sample(space)
subsample = x['boosting_type'].get('subsample', 1.0)
x['boosting_type'] = x['boosting_type']['boosting_type']
x['subsample'] = subsample
x
# 创建一个文件，用来存调参结果
OUT_FILE = 'bayes_test.csv'
of_connection = open(OUT_FILE, 'w')
writer = csv.writer(of_connection)

ITERATION = 0

# 写入列名
headers = ['loss', 'hyperparameters', 'iteration', 'runtime', 'score']
writer.writerow(headers)
of_connection.close()

# 测试目标函数
results = objective(sample(space))
print('The cross validation loss = {:.5f}.'.format(results['loss']))
print('The optimal number of estimators was {}.'.format(results['hyperparameters']['n_estimators']))
from hyperopt import tpe

# Create the algorithm
tpe_algorithm = tpe.suggest
from hyperopt import Trials

# Record results
trials = Trials()
OUT_FILE = 'bayes_test.csv'
of_connection = open(OUT_FILE, 'w')
writer = csv.writer(of_connection)

ITERATION = 0

headers = ['loss', 'hyperparameters', 'iteration', 'runtime', 'score']
writer.writerow(headers)
of_connection.close()
from hyperopt import fmin
global  ITERATION

ITERATION = 0

# 这一步就是调参的运行过程
best = fmin(fn = objective, space = space, algo = tpe.suggest, trials = trials,
            max_evals = MAX_EVALS)

best
results = pd.read_csv(OUT_FILE)
import ast

def evaluate(results, name):
    """evaluate 函数用来评估最佳参数的表现
    返回的结果是 将之前用csv记录的结果，结构化返回，方便后续统计分析参数的分布"""
    
    new_results = results.copy()
    # 使用ast.literal_eval str -> dic
    new_results['hyperparameters'] = new_results['hyperparameters'].map(ast.literal_eval)
    
    # Sort with best values on top
    new_results = new_results.sort_values('score', ascending = False).reset_index(drop = True)
    
    # 打印最高分数
    print('The highest cross validation score from {} was {:.5f} found on iteration {}.'.format(name, new_results.loc[0, 'score'], new_results.loc[0, 'iteration']))
    
    # 使用最佳参数建模训练，返回分数
    hyperparameters = new_results.loc[0, 'hyperparameters']
    model = lgb.LGBMClassifier(**hyperparameters)
    model.fit(train_features, train_labels)
    preds = model.predict_proba(test_features)[:, 1]
    
    print('ROC AUC from {} on test data = {:.5f}.'.format(name, roc_auc_score(test_labels, preds)))
    
    # 将dict存储的参数转化为 结构化的数据 df
    hyp_df = pd.DataFrame(columns = list(new_results.loc[0, 'hyperparameters'].keys()))

    for i, hyp in enumerate(new_results['hyperparameters']):
        hyp_df = hyp_df.append(pd.DataFrame(hyp, index = [0]), 
                               ignore_index = True)
        
    # 增加 iteration score 两列 
    hyp_df['iteration'] = new_results['iteration']
    hyp_df['score'] = new_results['score']
    
    return hyp_df
bayes_results = evaluate(results, name = 'Bayesian')
bayes_results
bayes_results = pd.read_csv('../input/home-credit-model-tuning/bayesian_trials_1000.csv').sort_values('score', ascending = False).reset_index()
random_results = pd.read_csv('../input/home-credit-model-tuning/random_search_trials_1000.csv').sort_values('score', ascending = False).reset_index()
random_results['loss'] = 1 - random_results['score']

bayes_params = evaluate(bayes_results, name = 'Bayesian')
random_params = evaluate(random_results, name = 'random')
# Dataframe of just scores
scores = pd.DataFrame({'ROC AUC': random_params['score'], 'iteration': random_params['iteration'], 'search': 'Random'})
scores = scores.append(pd.DataFrame({'ROC AUC': bayes_params['score'], 'iteration': bayes_params['iteration'], 'search': 'Bayesian'}))

scores['ROC AUC'] = scores['ROC AUC'].astype(np.float32)
scores['iteration'] = scores['iteration'].astype(np.int32)

scores.head()# Dataframe of just scores
scores = pd.DataFrame({'ROC AUC': random_params['score'], 'iteration': random_params['iteration'], 'search': 'Random'})
scores = scores.append(pd.DataFrame({'ROC AUC': bayes_params['score'], 'iteration': bayes_params['iteration'], 'search': 'Bayesian'}))

scores['ROC AUC'] = scores['ROC AUC'].astype(np.float32)
scores['iteration'] = scores['iteration'].astype(np.int32)

scores.head()
best_random_params = random_params.iloc[random_params['score'].idxmax(), :].copy()
best_bayes_params = bayes_params.iloc[bayes_params['score'].idxmax(), :].copy()
best_random_params
# Plot of scores over the course of searching
sns.lmplot('iteration', 'ROC AUC', hue = 'search', data = scores, size = 8);
plt.scatter(best_bayes_params['iteration'], best_bayes_params['score'], marker = '*', s = 400, c = 'orange', edgecolor = 'k')
plt.scatter(best_random_params['iteration'], best_random_params['score'], marker = '*', s = 400, c = 'blue', edgecolor = 'k')
plt.xlabel('Iteration'); plt.ylabel('ROC AUC'); plt.title("Validation ROC AUC versus Iteration");
fig, axs = plt.subplots(3, 1, figsize = (24, 22))

# 第一张参数的分布
hyper = 'learning_rate'
# sns.regplot('iteration', 'learning_rate', data = bayes_params, ax = axs[0])
# axs[i].scatter(best_bayes_params['iteration'], best_bayes_params[hyper], marker = '*', s = 200, c = 'k')
# axs[i].set(xlabel = 'Iteration', ylabel = '{}'.format(hyper), title = '{} over Search'.format(hyper));

sns.kdeplot(learning_rate_dist, label='Sampling Distribution', linewidth=4, ax=axs[0])
sns.kdeplot(random_params['learning_rate'], label='Random Search', linewidth=4, ax=axs[0])
sns.kdeplot(bayes_params['learning_rate'], label='Bayes Optimization', linewidth=4, ax=axs[0])
axs[0].vlines(x=best_random_params['learning_rate'], ymin=0.0, 
              ymax=20.0, linestyles='--', linewidth=4, colors=['orange', 'green'])
axs[0].set(xlabel='Learning Rate', ylabel='Learning Rate', title = 'Bayes Optimization Search');

# 第二章 贝叶斯优化的时序分布
sns.regplot('iteration', hyper, data = bayes_params, ax = axs[1])
axs[1].scatter(best_bayes_params['iteration'], best_bayes_params[hyper], marker = '*', s = 200, c = 'k')
axs[1].set(xlabel = 'Iteration', ylabel = '{}'.format(hyper), title = 'Bayes Optimization Search vs Random Search');

# 第三章 随机搜索的时序分布
sns.regplot('iteration', hyper, data=random_params, ax=axs[2])
axs[2].scatter(best_random_params['iteration'], best_random_params[hyper], marker='*', s=300, c='k')
axs[2].set(xlabel='Iteration', ylabel='{}'.format(hyper), title='Random Search')

plt.show()
hyper_list = ['colsample_bytree', 
              'min_child_samples', 
              'num_leaves',
              'reg_alpha',
              'reg_lambda',
              'subsample_for_bin']

vline_heights = [10, 0.01, 0.012, 2.6, 1.7, 0.000007]

for hyper, vheight in zip(hyper_list, vline_heights):
        
    fig, axs = plt.subplots(3, 1, figsize = (24, 22))

    # 第一张参数的分布
    sns.kdeplot([sample(space[hyper]) for _ in range(1000)], label = 'Sampling Distribution', linewidth = 4, ax=axs[0])
    sns.kdeplot(random_params[hyper], label='Random Search', linewidth=4, ax=axs[0])
    sns.kdeplot(bayes_params[hyper], label='Bayes Optimization', linewidth=4, ax=axs[0])
    axs[0].vlines(x=[best_bayes_params[hyper],best_random_params[hyper]], ymin=0.0, 
                  ymax=vheight, linestyles='--', linewidth=4, colors=['orange', 'green'])
    axs[0].set(xlabel=hyper, ylabel='density', title = 'Bayes Optimization Search vs Random Search');

    # 第二章 贝叶斯优化的时序分布
    sns.regplot('iteration', hyper, data = bayes_params, ax = axs[1])
    axs[1].scatter(best_bayes_params['iteration'], best_bayes_params[hyper], marker = '*', s = 200, c = 'k')
    axs[1].set(xlabel = 'Iteration', ylabel = '{}'.format(hyper), title = 'Bayes Optimization');

    # 第三章 随机搜索的时序分布
    sns.regplot('iteration', hyper, data=random_params, ax=axs[2])
    axs[2].scatter(best_random_params['iteration'], best_random_params[hyper], marker='*', s=300, c='k')
    axs[2].set(xlabel='Iteration', ylabel='{}'.format(hyper), title='Random Search')

    plt.show()

