#heavily borrwed from avaiable code
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/whats-cooking-kernels-only"))
print(os.listdir("../input/svm-12jul18"))
print(os.listdir("../input/let-s-cook-model"))
# Any results you write to the current directory are saved as output.
# Import the required libraries 
random_state = None
import time
starttime = time.monotonic()
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.sparse import csr_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedShuffleSplit
from scipy.sparse import hstack, csr_matrix
import numpy as np
from sklearn.preprocessing import StandardScaler
import gc
import pandas as pd
import json
import pdb
from sklearn.model_selection import KFold, StratifiedKFold

#get train
train_df = pd.read_json('../input/whats-cooking-kernels-only/train.json')
train_df.columns = ['target', 'id', 'ingredients']
train_df["num_ingredients"] = train_df['ingredients'].apply(lambda x: len(x))
train_df = train_df[ (train_df['num_ingredients'] > 1) ]
print(train_df.shape)
train_df_start = train_df.shape[0]
train_df.head()
#make beck up of target
y_bk = list(train_df['target'].copy().values)
#get test
test_df = pd.read_json('../input/whats-cooking-kernels-only/test.json')
test_df['cousine']=np.nan
test_df=test_df[['cousine','id','ingredients']]
test_df.columns = ['target', 'id', 'ingredients']
test_df["num_ingredients"] = test_df['ingredients'].apply(lambda x: len(x))
print (test_df.shape)
test_df.head()
#make beck up of test ids
test_ids_for_sub = test_df['id'].values
test_ids_for_sub
print("Combine Train and Submission")
df = pd.concat([train_df, test_df],axis=0,ignore_index=True)
del train_df, test_df
gc.collect()
print (df.shape)
df.head()
df.tail()
print (df.shape)
print(df.head())
print(df.tail())
#brutal copy paste from public kernel
from nltk.stem import WordNetLemmatizer
import re
lemmatizer = WordNetLemmatizer()
def preprocess(ingredients):
    ingredients_text = ' '.join(ingredients)
    ingredients_text = ingredients_text.lower()
    ingredients_text = ingredients_text.replace('-', ' ')
    words = []
    for word in ingredients_text.split():
        if re.findall('[0-9]', word): continue
        if len(word) <= 2: continue
        if '’' in word: continue
        word = lemmatizer.lemmatize(word)
        if len(word) > 0: words.append(word)
    return ' '.join(words)

for ingredient, expected in [
    ('Eggs', 'egg'),
    ('all-purpose flour', 'all purpose flour'),
    ('purée', 'purée'),
    ('1% low-fat milk', 'low fat milk'),
    ('half & half', 'half half'),
    ('safetida (powder)', 'safetida (powder)')
]:
    actual = preprocess([ingredient])
    assert actual == expected, f'"{expected}" is excpected but got "{actual}"'
df['ingredients'] = df['ingredients'].apply(lambda ingredients: preprocess(ingredients))
df.head()
df.tail()
# Feature Engineering 
print ("TF-IDF on text data ... ")
tfidf = TfidfVectorizer(binary=True)
print("combine other features with text vectors")
text_features = tfidf.fit_transform(df['ingredients'])
print (text_features.shape)
df_1 = pd.DataFrame(text_features.toarray())
del text_features
gc.collect()
print (df_1.shape)
print (df.shape)
df_1.head()
#less brutal copy paste from public kernel, we try to add a different ngram range
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
vectorizer = make_pipeline(
    TfidfVectorizer(sublinear_tf=True, ngram_range =(2,4), max_features=2000),
    FunctionTransformer(lambda x: x.astype('float32'), validate=False)
)
text_features_2 = vectorizer.fit_transform(df['ingredients'].values)
text_features_2.shape

df_2 = pd.DataFrame(text_features_2.todense())
del text_features_2
gc.collect()
print (df_2.shape)
print (df.shape)
df_2.head()
#combine all toghete... possible because the dataset is relatievely small
df = pd.concat([df, df_1, df_2],axis=1)
del df_1
del df_2
gc.collect()
print (df.shape)
df.head()
df.tail()
#rename columns, just to know where they come from
df.columns =['target', 'id', 'ingredients', 'num_ingredients']+['TfidfV_'+str(n) for n in range(2867)]+['Ngram_'+str(n) for n in range(2000)]
df.drop('ingredients',inplace=True,axis=True)
df.head()
df.tail()
#now this is wrong!
#i didnt manage to make the multiclass to work 
#so i'm using a regression to find the best parameters

from hyperopt import hp
from hyperopt import tpe
from hyperopt import STATUS_OK
from hyperopt import Trials
from hyperopt import fmin
import hyperopt.pyll.stochastic as st
import  lightgbm as lgb
import csv
from sklearn.preprocessing import OneHotEncoder
train_df = df[df['target'].notnull()]

target = train_df['target']
print ("Label Encode the Target Variable ... ")
lb = LabelEncoder()
y = lb.fit_transform(target)
len_y = len(set(y))

'''
#good when i will mange to make the multiclass working
#https://stackoverflow.com/questions/51139150/how-to-write-custom-f1-score-metric-in-light-gbm-python-in-multiclass-classifica
def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    print (len(labels))
    print(len(preds))
    #preds = preds.reshape(-1, len_y)
    #preds = preds.argmax(axis = 1)
    f_score = f1_score(labels , preds,  average = 'weighted')
    return 'f1_score', f_score, True
'''


bayes_trials = Trials()
tpe_algorithm = tpe.suggest

final_selection = [n for n in train_df.columns if n not in ['target', 'id']]
train_set = lgb.Dataset(train_df[final_selection], y)
print (train_set)
min_value = 100000
iteration = 0
def objective(params, n_folds = 5):
    global iteration
    iteration+=1
    """Objective function for Gradient Boosting Machine Hyperparameter Tuning"""
    #print (iteration)
    # Perform n_fold cross validation with hyperparameters
    # Use early stopping and evalute based on ROC AUC
    params['num_leaves']=int(params['num_leaves'])
    params['min_data_in_leaf']=int(params['min_data_in_leaf'])
    params['max_depth']=int(params['max_depth'])
    params['min_child_weight']=int(params['min_child_weight'])
    params['colsample_bytree']=round(params['colsample_bytree'],5)
    params['reg_alpha']=round(params['reg_alpha'],5)
    params['reg_lambda']=round(params['reg_lambda'],5)
    params['subsample']=round(params['subsample'],5)  
    params['learning_rate']=round(params['learning_rate'],5)
    params['learning_rate']=round(params['learning_rate'],5)
    #params['objective']= 'multiclass'
    #params['num_class']=len(train_df['target'].unique())
    #params['subsample_for_bin']=int(params['subsample_for_bin'])
    #params['is_unbalance']=True
    
    cv_results = lgb.cv(params, 
                        train_set, 
                        nfold = n_folds, 
                        num_boost_round = 10000, 
                        early_stopping_rounds = 200, 
                        metrics='multi_logloss',
                        #feval=evalerror,
                        seed = 50,
                        stratified=False)
    
    # Extract the best score
    #print(cv_results)
    best_score = min(cv_results['multi_logloss-mean'])
    optimal_rounds =len(cv_results['multi_logloss-mean']) 
    
    
    global min_value
    if  str(best_score) == '-inf':
        best_score = min_value+1
        
    if  best_score < 0:
        best_score = min_value+1
        
    if best_score < min_value:
        min_value = best_score
    print (f'Round: {iteration}\n the score is: {best_score:.3f}\n the best score so far {min_value:.3f}')
    
    # Loss must be minimized
    loss= best_score
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    
    writer.writerow([loss, params, best_score, optimal_rounds])
    of_connection.close()
    # Dictionary with information for evaluation
    
    return {'loss': loss, 'params': params, 'status': STATUS_OK}


space = {
    'subsample': hp.uniform('subsample', 0.2, 1),
    'num_leaves': hp.quniform('num_leaves', 5, 150, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.9)),
    'min_child_samples': hp.quniform('min_child_samples', 1, 100, 1),
    'min_data_in_leaf': hp.quniform('min_data_in_leaf', 1, 100, 1),
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.2, 1.0),
    'max_depth':hp.uniform('max_depth', 2, 15),
    'min_child_weight':hp.quniform('min_child_weight', 1, 100, 1),
    #'subsample_for_bin': hp.loguniform('subsample_for_bin', np.log(10), np.log(3000)),
}

# File to save first results
out_file = 'gbm_trials_essential.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['loss', 'params', 'best_score', 'optimal_rounds'])
of_connection.close() 
      
MAX_EVALS = 40
# Optimize
best = fmin(fn = objective,
            space = space, 
            algo = tpe.suggest, 
            max_evals = MAX_EVALS, 
            trials = bayes_trials)  
#print (best)
best
params = {}
params['num_leaves']=int(best['num_leaves'])
params['min_data_in_leaf']=int(best['min_data_in_leaf'])
params['max_depth']=int(best['max_depth'])
params['min_child_weight']=int(best['min_child_weight'])
params['colsample_bytree']=round(best['colsample_bytree'],5)
params['reg_alpha']=round(best['reg_alpha'],5)
params['reg_lambda']=round(best['reg_lambda'],5)
params['subsample']=round(best['subsample'],5)  
params['learning_rate']=round(best['learning_rate'],5)

len(target.unique())

#function to facilitate cv prediction of lightgbm model
def kfold_lightgbm(train_df, test_df, 
                   num_folds, lr = 0.02, 
                   stratified = True,  params={},
                   n_estimators=5000000, early_stopping_rounds= 200):    
    # Divide in training/validation and test data
    #print (test_df.head())
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    #del df
    gc.collect()
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=1001)
    # Create arrays and dataframes to store results
    nclasses = len(train_df['target'].unique())
    oof_preds = np.zeros( (train_df.shape[0],nclasses))
    sub_preds = np.zeros( (test_df.shape[0],nclasses))
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['target', 'id']]
    
    print ('len_feat', len(feats))
    feature_importance_df['f']=feats
    
    
    y = train_df['target']

    train_df = train_df[feats]
    
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, y)):
        train_x, train_y = train_df.iloc[train_idx], y.iloc[train_idx]
        valid_x, valid_y = train_df.iloc[valid_idx], y.iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(
            nthread=10,
            n_estimators=n_estimators,
            silent=-1,
            objective='multiclass',
            num_leaves=params['num_leaves'],
            min_data_in_leaf=params['min_data_in_leaf'],
            max_depth=params['max_depth'],
            min_child_weight=params['min_child_weight'],
            colsample_bytree=params['colsample_bytree'],
            reg_alpha=params['reg_alpha'],
            reg_lambda=params['reg_lambda'],
            subsample=params['subsample'],  
            learning_rate=params['learning_rate']     
        
        )

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
             verbose= 100, early_stopping_rounds= early_stopping_rounds)

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)
        preds = clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)
        #print (preds)
        sub_preds += preds
        #[:, 1] / float(folds.n_splits)
        #print('inside', len(feats))
        #print ('len clf.feature_importances_', len(clf.feature_importances_))
        feature_importance_df["fold"+str(n_fold)] = clf.feature_importances_

        del train_x, train_y, valid_x, valid_y
        gc.collect()
    
    sub_preds = sub_preds/float(folds.n_splits)
    return feature_importance_df, clf, sub_preds, oof_preds
test_df = df[df['target'].isnull()]
del df
gc.collect()
test_df.head()
test_df.tail()
#### Pass the best parameters to a lightgbm model
#### Get cv prediction and feature importance

feat_importance, clf, sub_preds, oof_preds = kfold_lightgbm(train_df, test_df, num_folds= 5, lr=0.01, 
                                 stratified= True, params=params,
                                 #n_estimators=200, early_stopping_rounds= 2
                                                                  )
feat_importance.to_csv('fimp_all.csv') 
#### Again, Pass the best parameters to a lightgbm model after scramble of target
#### Get cv prediction and feature importance
train_df['target'] = train_df['target'].sample(frac=1,random_state=1976).values
feat_importance_random, clf, sub_preds, oof_preds = kfold_lightgbm(train_df, test_df, num_folds= 5, lr=0.01, 
                                 stratified= True, params=params,
                                 #n_estimators=200, early_stopping_rounds= 2
                                                                  )
feat_importance_random.to_csv('fimp_all_random.csv') 
#### Select the best feature comparing the normal scores vs the scrambled score 
feat_importance.set_index('f',inplace=True)
feat_importance_random.set_index('f',inplace=True)
from scipy import stats
from statsmodels.sandbox.stats.multicomp import multipletests
from tqdm import tqdm

importance = pd.DataFrame()
#selection_df['f']=feat_importance.index.values
statistics = []
pvalues = []
for f in tqdm(feat_importance.index.values):
    statistic, pvalue = stats.ks_2samp(feat_importance.loc[f], feat_importance_random.loc[f])
    statistics.append(statistic)
    pvalues.append(pvalue)
importance['sum']=feat_importance.sum(axis=1).values
importance['random_sum']=feat_importance_random.sum(axis=1).values
importance['f']=feat_importance.index.values
importance.set_index('f',inplace=True)
#just for some plotting, we add to the sum the minimum value of the random dataset
importance['sum']=importance['sum']+importance['random_sum'].min()
importance['random_sum']=importance['random_sum']+importance['random_sum'].min()
importance['fc']=(importance['sum']+1)/(importance['random_sum']+1)
importance['pval']=pvalues
importance['ks_stat']=statistics
padj = multipletests(importance['pval'], method='bonferroni')
importance['padj']=padj[1]
importance.sort_values('pval').head()    

importance.sort_values('fc').tail(20)  
importance.loc['num_ingredients']
np.log2(importance['fc']).plot(kind='hist')
print(importance[importance['padj']<0.05].shape)
print(importance[np.log2(importance['fc'])>1].shape)
feature_identified = list(importance[importance['fc']>1].index.values)
print (len(feature_identified))
train_set = lgb.Dataset(train_df[feature_identified], y)
print (train_set)
min_value = 100000
iteration = 0
def objective(params, n_folds = 5):
    global iteration
    iteration+=1
    """Objective function for Gradient Boosting Machine Hyperparameter Tuning"""
    #print (iteration)
    # Perform n_fold cross validation with hyperparameters
    # Use early stopping and evalute based on ROC AUC
    params['num_leaves']=int(params['num_leaves'])
    params['min_data_in_leaf']=int(params['min_data_in_leaf'])
    params['max_depth']=int(params['max_depth'])
    params['min_child_weight']=int(params['min_child_weight'])
    params['colsample_bytree']=round(params['colsample_bytree'],5)
    params['reg_alpha']=round(params['reg_alpha'],5)
    params['reg_lambda']=round(params['reg_lambda'],5)
    params['subsample']=round(params['subsample'],5)  
    params['learning_rate']=round(params['learning_rate'],5)
    params['learning_rate']=round(params['learning_rate'],5)
    #params['objective']= 'multiclass'
    #params['num_class']=len(train_df['target'].unique())
    #params['subsample_for_bin']=int(params['subsample_for_bin'])
    #params['is_unbalance']=True
    
    cv_results = lgb.cv(params, 
                        train_set, 
                        nfold = n_folds, 
                        num_boost_round = 1000000, 
                        early_stopping_rounds = 200, 
                        metrics='multi_logloss',
                        #feval=evalerror,
                        seed = 50,
                        stratified=False)
    
    # Extract the best score
    #print(cv_results)
    best_score = min(cv_results['multi_logloss-mean'])
    optimal_rounds =len(cv_results['multi_logloss-mean']) 
    
    
    global min_value
    if  str(best_score) == '-inf':
        best_score = min_value+1 
        
    if  best_score < 0:
        best_score = min_value+1        
        
    if best_score < min_value:
        min_value = best_score
    print (f'Round: {iteration}\n the score is: {best_score:.3f}\n the best score so far {min_value:.3f}')
    
    # Loss must be minimized
    loss= best_score
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    
    writer.writerow([loss, params, best_score, optimal_rounds])
    of_connection.close()
    # Dictionary with information for evaluation
    
    return {'loss': loss, 'params': params, 'status': STATUS_OK}


space = {
    'subsample': hp.uniform('subsample', 0.2, 1),
    'num_leaves': hp.quniform('num_leaves', 5, 150, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.9)),
    'min_child_samples': hp.quniform('min_child_samples', 1, 100, 1),
    'min_data_in_leaf': hp.quniform('min_data_in_leaf', 1, 100, 1),
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.2, 1.0),
    'max_depth':hp.uniform('max_depth', 2, 15),
    'min_child_weight':hp.quniform('min_child_weight', 1, 100, 1),
    #'subsample_for_bin': hp.loguniform('subsample_for_bin', np.log(10), np.log(3000)),
}

# File to save first results
out_file = 'gbm_trials_essential.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['loss', 'params', 'best_score', 'optimal_rounds'])
of_connection.close() 
      
MAX_EVALS = 150
# Optimize
best = fmin(fn = objective,
            space = space, 
            algo = tpe.suggest, 
            max_evals = MAX_EVALS, 
            trials = bayes_trials)  
print (best)
params = {}
params['num_leaves']=int(best['num_leaves'])
params['min_data_in_leaf']=int(best['min_data_in_leaf'])
params['max_depth']=int(best['max_depth'])
params['min_child_weight']=int(best['min_child_weight'])
params['colsample_bytree']=round(best['colsample_bytree'],5)
params['reg_alpha']=round(best['reg_alpha'],5)
params['reg_lambda']=round(best['reg_lambda'],5)
params['subsample']=round(best['subsample'],5)  
params['learning_rate']=round(best['learning_rate'],5)
### run the final model, add back the non scarmbled target
train_df['target']=y_bk
feat_importance, clf, sub_preds, oof_preds = kfold_lightgbm(train_df[feature_identified+['target']], 
                                                            test_df[feature_identified+['target']],
                                                            num_folds= 5, lr=0.01, stratified= True,params=params,
                                                           #n_estimators=200, early_stopping_rounds= 2
                                                           )
                                 

feat_importance.to_csv('fimp_final.csv') 
print(clf.classes_)
sub = pd.DataFrame()
for index,c in enumerate(clf.classes_):
    sub[c]=sub_preds[:,index]
cuisine = []
for item in sub.index.values:
    cuisine.append(np.argmax(sub.loc[item]))
sub['cuisine1']=cuisine
sub['id']=test_ids_for_sub
### Blend with the top score models
temp_1 = pd.read_csv('../input/svm-12jul18/svm_output_None.csv')
temp_2 = pd.read_csv('../input/let-s-cook-model/submission.csv')
sub['cuisine2']=temp_1['cuisine']
sub['cuisine3']=temp_2['cuisine']
sub['diff'] = [1 if a != b else 0 for a,b in zip(sub['cuisine2'], sub['cuisine3'])]
sub['diff'].value_counts()
sub[sub['diff']==1][['cuisine1','cuisine2','cuisine3']].head(50)
cuisine = []
for index in sub.index.values:
    temp = sub[['cuisine1', 'cuisine2', 'cuisine3']].iloc[index].value_counts()
    temp_score = temp[0]
    temp_res = temp.index.values[0]
    if temp_score ==1:
        temp_res = sub[['cuisine3']].iloc[index].values[0]
    cuisine.append(temp_res)
sub['cuisine']=cuisine
sub[['id','cuisine']].head()
### et voila
sub[['id','cuisine']].to_csv('finalsub.csv',index=False)