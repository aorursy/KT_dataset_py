import gc

import os

import logging

import datetime

import warnings

import numpy as np

import pandas as pd

import seaborn as sns

import lightgbm as lgb

from tqdm import tqdm_notebook

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')
IS_LOCAL = False

if(IS_LOCAL):

    PATH="../input/Santander/"

else:

    PATH="../input/"

os.listdir(PATH)
%%time

train_df = pd.read_csv(PATH+"train.csv")

test_df = pd.read_csv(PATH+"test.csv")
train_df.shape, test_df.shape
train_df.head()
test_df.head()
def missing_data(data):

    total = data.isnull().sum()

    percent = (data.isnull().sum()/data.isnull().count()*100)

    tt = pd.concat([total,percent],axis=1,keys=['Total','Percent'])

    types = []

    for col in data.columns:

        dtype = str(data[col].dtype)

        types.append(dtype)

    tt['Types'] = types

    return (np.transpose(tt))
%%time

missing_data(train_df)
%%time

missing_data(test_df)
%%time

train_df.describe()
%%time

test_df.describe()
def plot_feature_scatter(df1,df2,features):

    i = 0

    sns.set_style('whitegrid')

    plt.figure()

    fig,ax=plt.subplots(4,4,figsize=[14,14])

    

    for feature in features:

        i+=1

        plt.subplot(4,4,i)

        plt.scatter(df1[feature],df2[feature],marker='+')

        plt.xlabel(feature,fontsize=9)

    plt.show()
features = ['var_0', 'var_1','var_2','var_3', 'var_4', 'var_5', 'var_6', 'var_7', 

           'var_8', 'var_9', 'var_10','var_11','var_12', 'var_13', 'var_14', 'var_15', 

           ]

plot_feature_scatter(train_df[::20],test_df[::20], features)
sns.countplot(train_df['target'])
print("There are {}% target values with 1".format(100 * train_df["target"].value_counts()[1]/train_df.shape[0]))
def plot_feature_distribution(df1,df2,label1,label2,features):

    i = 0

    sns.set_style('whitegrid')

    plt.figure()

    fig,ax = plt.subplots(10,10,figsize=[18,22])

    

    for feature in features:

        i += 1

        plt.subplot(10,10,i)

        sns.kdeplot(df1[feature],bw=0.5,label=label1)

        sns.kdeplot(df2[feature],bw=0.5,label=label2)

        plt.xlabel(feature,fontsize=9)

        locs, labels = plt.xticks()

        plt.tick_params(axis='x',which='major',labelsize=6,pad=-6)

        plt.tick_params(axis='y',which='major',labelsize=6)

    plt.show()
t0 = train_df.loc[train_df['target']==0]

t1 = train_df.loc[train_df['target']==1]

features = train_df.columns.values[2:102]

plot_feature_distribution(t0,t1,'0','1',features)
features = train_df.columns.values[102:202]

plot_feature_distribution(t0, t1, '0', '1', features)
features = train_df.columns.values[2:102]

plot_feature_distribution(train_df, test_df, 'train', 'test', features)
features = train_df.columns.values[102:202]

plot_feature_distribution(train_df, test_df, 'train', 'test', features)
plt.figure(figsize=[6,6])

features = train_df.columns.values[2:202]

plt.title("Distribution of mean values per row in the train and test set")

sns.distplot(train_df[features].mean(axis=1),color="green",kde=True,bins=120,label='train')

sns.distplot(test_df[features].mean(axis=1),color="blue",kde=True,bins=120,label='test')

plt.legend()

plt.show()
plt.figure(figsize=(16,6))

plt.title("Distribution of mean values per column in the train and test set")

sns.distplot(train_df[features].mean(axis=0),color="magenta",kde=True,bins=120, label='train')

sns.distplot(test_df[features].mean(axis=0),color="darkblue", kde=True,bins=120, label='test')

plt.legend()

plt.show()
plt.figure(figsize=(16,6))

plt.title("Distribution of std values per row in the train and test set")

sns.distplot(train_df[features].std(axis=1),color="black", kde=True,bins=120, label='train')

sns.distplot(test_df[features].std(axis=1),color="red", kde=True,bins=120, label='test')

plt.legend();plt.show()
plt.figure(figsize=(16,6))

plt.title("Distribution of std values per column in the train and test set")

sns.distplot(train_df[features].std(axis=0),color="blue",kde=True,bins=120, label='train')

sns.distplot(test_df[features].std(axis=0),color="green", kde=True,bins=120, label='test')

plt.legend(); plt.show()
t0 = train_df.loc[train_df['target'] == 0]

t1 = train_df.loc[train_df['target'] == 1]

plt.figure(figsize=(16,6))

plt.title("Distribution of mean values per row in the train set")

sns.distplot(t0[features].mean(axis=1),color="red", kde=True,bins=120, label='target = 0')

sns.distplot(t1[features].mean(axis=1),color="blue", kde=True,bins=120, label='target = 1')

plt.legend(); plt.show()
plt.figure(figsize=(16,6))

plt.title("Distribution of mean values per column in the train set")

sns.distplot(t0[features].mean(axis=0),color="green", kde=True,bins=120, label='target = 0')

sns.distplot(t1[features].mean(axis=0),color="darkblue", kde=True,bins=120, label='target = 1')

plt.legend(); plt.show()
#I think this code is better than orgin code



correlations = train_df[features].corr().where(np.triu(np.ones(train_df[features].corr().shape),k=1).astype(np.bool))

correlations_df = correlations.abs().unstack().dropna().sort_values().reset_index()

correlations_df.shape
correlations_df.head(5)
correlations_df.tail(5)
[col for col in correlations.columns if any(abs(correlations[col])>0.95)]
%%time

features = train_df.columns.values[2:202]

unique_max_train = []

unique_max_test = []

for feature in features:

    values = train_df[feature].value_counts()

    unique_max_train.append([feature,values.max(),values.idxmax()])



    values = test_df[feature].value_counts()

    unique_max_test.append([feature,values.max(),values.idxmax()])
np.transpose(pd.DataFrame(unique_max_train,columns=['Feature','Max duplicates','Values']).sort_values(by='Max duplicates',ascending=False).head(15))
np.transpose(pd.DataFrame(unique_max_test,columns=['Feature','Max duplicates','Values']).sort_values(by='Max duplicates',ascending=False).head(15))
%%time



i = 1

for df in [test_df, train_df]:

    idx = df.columns.values[i:i+200]

    df['sum'] = df[idx].sum(axis=1)  

    df['min'] = df[idx].min(axis=1)

    df['max'] = df[idx].max(axis=1)

    df['mean'] = df[idx].mean(axis=1)

    df['std'] = df[idx].std(axis=1)

    df['skew'] = df[idx].skew(axis=1)

    df['kurt'] = df[idx].kurtosis(axis=1)

    df['med'] = df[idx].median(axis=1)

    df['range'] = df['max']-df['min']

    i = i + 1
train_df[train_df.columns[202:]].head()
test_df[test_df.columns[201:]].head()
features = train_df.columns.values[2:]

correlations = train_df[features].corr().where(np.triu(np.ones(train_df[features].corr().shape),k=1).astype(np.bool))

correlations_df = correlations.abs().stack().reset_index().rename(columns={0:'corr'}).sort_values(by='corr')

correlations_df.shape
correlations_df.head()
correlations_df.tail()
drop_cols = [col for col in correlations.columns if any(abs(correlations[col])>0.95)]
print("Shape of train_df: {}, test_df: {}".format(train_df.shape,test_df.shape))
train_df = train_df.drop(columns=drop_cols)

test_df = test_df.drop(columns=drop_cols)
print("Shape of train_df: {}, test_df: {}".format(train_df.shape,test_df.shape))
def plot_new_feature_distribution(df1,df2,label1,label2,features):

    i = 0

    sns.set_style('whitegrid')

    fig, ax = plt.subplots(2,4,figsize=[18,8])

    

    for feature in features:

        i += 1

        plt.subplot(2,4,i)

        sns.kdeplot(df1[feature],bw=0.5,label=label1)

        sns.kdeplot(df2[feature],bw=0.5,label=label2)

        plt.xlabel(feature,fontsize=11)

        locs, lables = plt.xticks()

        plt.tick_params(axis="x",which="major",labelsize=8)

        plt.tick_params(axis="y",which="major",labelsize=8)

    plt.show()
t0 = train_df.loc[train_df['target'] == 0]

t1 = train_df.loc[train_df['target'] == 1]

features = train_df.columns.values[202:]

plot_new_feature_distribution(t0, t1, 'target: 0', 'target: 1', features)
features = train_df.columns.values[202:]

plot_new_feature_distribution(train_df, test_df, 'train', 'test', features)
print('Train and test columns: {} {}'.format(len(train_df.columns), len(test_df.columns)))
train = train_df.drop(columns=['ID_code','target'])

train_label = train_df['target']

test = test_df.drop(columns='ID_code')

test_ids = test_df['ID_code']
# from sklearn.ensemble import RandomForestClassifier

# from sklearn.metrics import make_scorer,roc_auc_score

# from sklearn.model_selection import cross_val_score



# scorer = make_scorer(roc_auc_score,greater_is_better=True)



# rf = RandomForestClassifier(random_state=12,n_estimators=15000,n_jobs=-1)



# rf.fit(train,train_label)



# # cv_score = cross_val_score(rf,train,train_label,cv=5,scoring=scorer)

# # print(f"5 fold cv score is {cv_score.mean()}")



# indices = np.argsort(rf.feature_importances_)[::-1]

# feature_names = train.columns

# importances = rf.feature_importances_



# df = pd.DataFrame(columns=['feature','importance'])

# df['feature'] = feature_names

# df['importance'] = importances





# df.sort_values(by='importance',ascending=False).tail()
# from sklearn.feature_selection import RFECV



# estimator = RandomForestClassifier(random_state=12,n_estimators=15000,n_jobs=-1)



# selector = RFECV(estimator,step=1,cv=5,scoring=scorer,n_jobs=-1)



# selector.fit(train,train_label)
import re

import string



def del_punct(one_list):

    

    return_list = []    

    regex = re.compile('['+re.escape("'")+']')

    

    for element in one_list:

        return_list.append(regex.sub(" ",element).strip())

    

    return return_list
def distinguish_str(value_list):

    

    output = []

    

    regex = re.compile('[0-9]')

    

    for i,element in enumerate(value_list):

        if regex.search(element):

            output.append(float(element))

        else:

            output.append(element)

    

    return output
def model_gbm(train,train_label,test,test_ids,nfolds=5,hyperparameters=None):

    

    feature_names = list(train.columns)

    

    valid_scores = np.zeros(len(train))

    predictions = np.zeros(len(test))

    

    feature_importance_df = pd.DataFrame()

    

    max_iters_df = pd.DataFrame(columns=["folds","iters"])

    

    iters = []

    folds = []

    

    if hyperparameters:

        params = hyperparameters

        

#         If you guys get hyperparams below dataframe by hyperopt, the dictionary will be string type!! 

#         So You should change to dict following commented area. 

#         But, As I mentioned below, I'll put my hyperparams what tested at colab environment already!!

        

#         keys = []

#         values = []

        

#         integer_elements = ['subsample_freq','max_depth','num_leaves','subsample_for_bin','min_child_samples','n_estimators']

        

#         for element in params[1:-1].split(","):

#             keys.append(element.split(":")[0])

#             values.append(element.split(":")[1])

            

#         keys = del_punct(keys)

#         values = distinguish_str(del_punct(values)) 

        

#         params = dict(zip(keys,values))



#         for element in integer_elements:

#             params[element] = int(params[element])



        del(params['n_estimators'])

        

        params['boost_from_average'] = True

        params['seed'] = 31452

        params['feature_fraction_seed'] = 31415

        params['bagging_seed'] = 31415

        params['drop_seed'] = 31415

        params['data_random_seed'] =31415

        params['metric'] = 'auc'

    

    #The hyperparams where I got from Gabriel's code

    else:

        params = {

        'num_leaves': 6,

        'max_bin': 63,

        'min_data_in_leaf': 45,

        'learning_rate': 0.01,

        'min_sum_hessian_in_leaf': 0.000446,#min_child_weight

        'bagging_fraction': 0.55, 

        'bagging_freq': 5, 

        'max_depth': 14,

        'save_binary': True,

        'seed': 31452,

        'feature_fraction_seed': 31415, 

        'feature_fraction': 0.51, #colsample_by_tree => 매 트리 생성시 가져오는 피쳐의 개수

        'bagging_seed': 31415, #배깅을 사용한다면 쓰는 시드

        'drop_seed': 31415,

        'data_random_seed': 31415,

        'objective': 'binary',

        'boosting_type': 'gbdt',

        'verbose': 1,

        'metric': 'auc',

        'is_unbalance': True,

        'boost_from_average': False

    }

    

    strfkold = StratifiedKFold(n_splits=nfolds,shuffle=True,random_state=12)

    

    for i,(train_indices,valid_indices) in enumerate(strfkold.split(train.values,train_label.values)):

        

        print("{} fold processing".format(i+1)+"#"*20)

        

        d_train = lgb.Dataset(train.values[train_indices,:],label = train_label[train_indices])

        d_valid = lgb.Dataset(train.values[valid_indices,:],label = train_label[valid_indices])

        

        n_rounds = 15000

        

        lgb_model = lgb.train(params,d_train,num_boost_round=n_rounds,valid_sets=[d_train,d_valid],valid_names=['train','valid'],verbose_eval=1000,early_stopping_rounds=250)

        

        valid_scores[valid_indices] = lgb_model.predict(train.values[valid_indices,:],num_iteration=lgb_model.best_iteration)

        

        fold_importance_df = pd.DataFrame(columns=["Feature","importance","fold"])

        fold_importance_df["Feature"] = feature_names

        fold_importance_df["importance"] = lgb_model.feature_importance()

        fold_importance_df["fold"] = i + 1

        

        feature_importance_df = pd.concat([feature_importance_df,fold_importance_df],axis=0)

        

        folds.append(i+1)

        iters.append(lgb_model.best_iteration)

        

        predictions += lgb_model.predict(test.values,num_iteration=lgb_model.best_iteration)/nfolds    

        

        display("valid_set score is %f and best_iteration is %d of %d fold"%(roc_auc_score(train_label[valid_indices],valid_scores[valid_indices]),lgb_model.best_iteration,i+1))

        

    max_iters_df["folds"] = folds

    max_iters_df["iters"] = iters

    

    display("CV score of valid_set for %d fold is %f and maximum of best_iteration is %d of %d fold"%(nfolds,roc_auc_score(train_label,valid_scores),max_iters_df['iters'].max(),max_iters_df['iters'].idxmax()+1))

    

    return valid_scores,predictions,feature_importance_df
from hyperopt import hp,tpe,Trials,fmin, STATUS_OK

from hyperopt.pyll.stochastic import sample
import csv

import ast

from timeit import default_timer as timer
def lgb_roc_auc(labels,predictions):

#     print(predictions)

#     predictions = predictions.reshape(len(np.unique(labels)),-1).argmax(axis=0)

    

    metric_value = roc_auc_score(labels,predictions)

    

    return 'roc_auc',metric_value,True
def objective(hyperparameters, nfold=5):

    

    global ITERATION

    ITERATION += 1

    

    for parameter_name in ['max_depth','num_leaves','subsample_for_bin','min_child_samples','subsample_freq']:

        hyperparameters[parameter_name] = int(hyperparameters[parameter_name])

        

    strkfold = StratifiedKFold(n_splits=nfold,shuffle=True)

        

    features = np.array(train)

    labels = np.array(train_label).reshape((-1))

        

    valid_scores = []

    best_estimators = []

    run_times = []

    

    model = lgb.LGBMClassifier(**hyperparameters,n_jobs=-1,metric='None',n_estimators=1000)

        

    for i, (train_indices,valid_indices) in enumerate(strkfold.split(features,labels)):



        print("#"*20,"%d fold of %d itertaion"%(i+1,ITERATION))

        

        X_train,X_valid = features[train_indices],features[valid_indices]

        y_train,y_valid = labels[train_indices], labels[valid_indices]

            

        start = timer()

        #250 / 1000    

        model.fit(X_train,y_train,early_stopping_rounds=50,

                eval_metric=lgb_roc_auc,eval_set=[(X_train,y_train),(X_valid,y_valid)],

                eval_names=['train','valid'],verbose=200)

            

        end = timer()

            

        valid_scores.append(model.best_score_['valid']['roc_auc'])

            

        best_estimators.append(model.best_iteration_)

            

        run_times.append(end-start)

            

    score = np.mean(valid_scores)

    score_std = np.std(valid_scores)

    loss = 1-score

        

    run_time = np.mean(run_times)

    run_time_std = np.std(run_times)

        

    estimators = int(np.mean(best_estimators))

    hyperparameters['n_estimators'] = estimators

        

    of_connection = open(OUT_FILE,'a')

    writer = csv.writer(of_connection)

    writer.writerow([loss,hyperparameters,ITERATION,run_time,score,score_std])

    of_connection.close()

    

    display(f'Iteration: {ITERATION}, Score: {round(score, 4)}.')

    

    if ITERATION % PROGRESS == 0:

        display(f'Iteration: {ITERATION}, Current Score: {round(score, 4)}.')

    

    return {'loss': loss, 'hyperparameters': hyperparameters, 'iteration': ITERATION,

            'time': run_time, 'time_std': run_time_std, 'status': STATUS_OK, 

            'score': score, 'score_std': score_std}
space = {

    'boosting_type':'gbdt',

    'objective':'binary',

    'is_unbalance':True,

    'subsample': hp.uniform('gbdt_subsample',0.5,1),

    'subsample_freq':hp.quniform('gbdt_subsample_freq',1,10,1),

    'max_depth': hp.quniform('max_depth',5,20,3),

    'num_leaves': hp.quniform('num_leaves',20,60,10),

    'learning_rate':hp.loguniform('learning_rate',np.log(0.025),np.log(0.25)),

    'subsample_for_bin':hp.quniform('subsample_for_bin',2000,100000,2000),

    'min_child_samples': hp.quniform('min_child_samples',5,80,5),

    'colsample_bytree': hp.uniform('colsample_by_tree', 0.5, 1.0),

    'min_child_weight':hp.uniform('min_child_weight',0.01,0.000001)

}
sample(space)
algo = tpe.suggest
# Record results

trials = Trials()



# Create a file and open a connection

OUT_FILE = 'optimization.csv'

of_connection = open(OUT_FILE, 'w')

writer = csv.writer(of_connection)



MAX_EVALS = 10

PROGRESS = 10

ITERATION = 0



# Write column names

headers = ['loss', 'hyperparameters', 'iteration', 'runtime', 'score', 'std']

writer.writerow(headers)

of_connection.close()
import datetime



print("beginning time is {}".format(datetime.datetime.now()))

display("Running Optimization for {} Trials.".format(MAX_EVALS))



# Run optimization

best = fmin(fn = objective, space = space, algo = tpe.suggest, trials = trials,max_evals = MAX_EVALS)



print("end time is {}".format(datetime.datetime.now()))
import json



with open('trials.json','w') as f:

    f.write(json.dumps(str(trials)))
results = pd.read_csv(OUT_FILE).sort_values('loss', ascending = True).reset_index()

results.head()
plt.figure(figsize=[8,6])

sns.regplot('iteration','score',data=results);

plt.title('OPT Scores')

plt.xticks(list(range(1,results.iteration.max()+1,3)))
hyperparameters = results.hyperparameters[0]
hyperparameters = {

    'boosting_type': 'gbdt',

    'colsample_bytree': 0.7812943473676428,

    'is_unbalance': True,

    'learning_rate': 0.012732207618246335,

    'max_bin': 200,

    'max_depth': 14,

    'min_child_samples': 70,

    'min_child_weight': 0.0010242091278688855,

    'num_leaves': 10,

    'objective': 'binary',

    'subsample': 0.8026192939361728,

    'subsample_for_bin': 72000,

    'subsample_freq': 7,

    'n_estimators': 6589}
val_scores, predictions, gbm_fi= model_gbm(train,train_label,test,test_ids,hyperparameters=hyperparameters)
submission = pd.read_csv('../input/sample_submission.csv')

submission['target'] = predictions

submission.to_csv("submission.csv",index=False)