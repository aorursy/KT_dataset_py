import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,KFold,train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from category_encoders import CatBoostEncoder, TargetEncoder

from sklearn.linear_model import SGDClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from time import time

from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,roc_curve,classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')

train.head()
test.head()
train.info()
%matplotlib notebook
%matplotlib inline
plt.figure(figsize=(18,9))
sns.heatmap(train.isnull(), cbar= False)
print(100*train['Age'].isnull().sum()/train.shape[0])
print(100*train['Cabin'].isnull().sum()/train.shape[0])
print(100*train['Embarked'].isnull().sum()/train.shape[0])
%matplotlib notebook
%matplotlib inline
plt.figure(figsize=(18,9))
sns.heatmap(test.isnull(), cbar= False)
print(100*test['Age'].isnull().sum()/train.shape[0])
print(100*test['Cabin'].isnull().sum()/train.shape[0])
print(100*test['Fare'].isnull().sum()/train.shape[0])
%matplotlib notebook
%matplotlib inline
def cor_heat(df):
    cor=df.corr()
    plt.figure(figsize=(10,5),dpi=100)
    sns.heatmap(data=cor,annot=True,square=True,linewidths=0.1,cmap='YlGnBu')
    plt.title("Pearson Co-relation: Heat Map")
cor_heat(train)
cat_columns= [col for col in train.columns if train[col].dtype==object]
cat_columns
def id_relation(col):
    print(train.groupby(col)['Survived'].value_counts(normalize=True).unstack().dropna()[1].sort_values(ascending=False))
for col in cat_columns:
    id_relation(col)

train_titanic = train.drop(['PassengerId','Age','Cabin'],axis = 1)
test_titanic = test.drop(['PassengerId','Age','Cabin'],axis = 1)
train_titanic.Embarked.mode()
train_titanic.Embarked.fillna('S',inplace=True)
train_titanic.isnull().sum().sum()
test_titanic.Fare.median()
test_titanic.Fare.fillna(14.4542,inplace=True)
test_titanic.isnull().sum().sum()
train_titanic.columns
test_titanic.columns
cat_cols = [col for col in train_titanic.columns if train_titanic[col].dtype=='object']
cat_cols
train_titanic.drop('Name',inplace=True,axis=1)
test_titanic.drop('Name',inplace=True,axis=1)
train_titanic.columns
test_titanic.columns
cat_cols = [col for col in train_titanic.columns if train_titanic[col].dtype=='object']
cat_cols
for cols in cat_cols:
    print(train_titanic[cols].nunique())
#balanced/imbalanced?
%matplotlib notebook
%matplotlib inline
plt.figure(figsize=(10,5))
sns.countplot(x='Survived',data=train_titanic)
cbe=CatBoostEncoder(cols=cat_cols)
    # X= df_tran_tr.drop(['isFraud'],axis=1)
    # y= df_tran_tr[['isFraud']]
cbe.fit(train_titanic[cat_cols],train_titanic[['Survived']])

    # #Train & Test Set transforming
train_titanic=train_titanic.join(cbe.transform(train_titanic[cat_cols]).add_suffix('_target'))
train_titanic.drop(['Sex', 'Ticket', 'Embarked'],axis=1,inplace=True)

test_titanic=test_titanic.join(cbe.transform(test_titanic[cat_cols]).add_suffix('_target'))
test_titanic.drop(['Sex', 'Ticket', 'Embarked'],axis=1,inplace=True)

train_titanic.head()
test_titanic


#don't do this before category encoding
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df
#Reducing memory without any data loss
train_titanic=reduce_mem_usage(train_titanic)
test_titanic=reduce_mem_usage(test_titanic)
train_set = lgb.Dataset(train_titanic.drop("Survived",axis=1), label = train_titanic["Survived"])

import csv
from hyperopt import STATUS_OK
from timeit import default_timer as timer
N_FOLDS=5

def objective(params, n_folds = N_FOLDS):
    """Objective function for Gradient Boosting Machine Hyperparameter Optimization"""
    
    # Keep track of evals
    global ITERATION
    
    ITERATION += 1
    
    # Retrieve the subsample if present otherwise set to 1.0
    subsample = params['boosting_type'].get('subsample', 1.0)
    
    # Extract the boosting type
    params['boosting_type'] = params['boosting_type']['boosting_type']
    params['subsample'] = subsample
    
    # Make sure parameters that need to be integers are integers
    for parameter_name in ['num_leaves', 'subsample_for_bin', 'min_child_samples']:
        params[parameter_name] = int(params[parameter_name])
    
    start = timer()
    
    # Perform n_folds cross validation
    cv_results = lgb.cv(params, train_set, num_boost_round = 1000, nfold = n_folds, 
                        early_stopping_rounds = 100, metrics = 'auc', seed = 50)
    
    run_time = timer() - start
    
    # Extract the best score
    best_score = np.max(cv_results['auc-mean'])
    
    # Loss must be minimized
    loss = 1 - best_score
    
    # Boosting rounds that returned the highest cv score
    n_estimators = int(np.argmax(cv_results['auc-mean']) + 1)

    # Write to the csv file ('a' means append)
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss, params, ITERATION, n_estimators, run_time])
    
    # Dictionary with information for evaluation
    return {'loss': loss, 'params': params, 'iteration': ITERATION,
            'estimators': n_estimators, 
            'train_time': run_time, 'status': STATUS_OK}
from hyperopt import hp
from hyperopt.pyll.stochastic import sample
# Create the learning rate
learning_rate = {'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.2))}
learning_rate_dist = []

# Draw 10000 samples from the learning rate domain
for _ in range(10000):
    learning_rate_dist.append(sample(learning_rate)['learning_rate'])
    
plt.figure(figsize = (8, 6))
sns.kdeplot(learning_rate_dist, color = 'red', linewidth = 2, shade = True);
plt.title('Learning Rate Distribution', size = 18); 
plt.xlabel('Learning Rate', size = 16); plt.ylabel('Density', size = 16);
# Discrete uniform distribution
num_leaves = {'num_leaves': hp.quniform('num_leaves', 30, 150, 1)}
num_leaves_dist = []

# Sample 10000 times from the number of leaves distribution
for _ in range(10000):
    num_leaves_dist.append(sample(num_leaves)['num_leaves'])
    
# kdeplot
plt.figure(figsize = (8, 6))
sns.kdeplot(num_leaves_dist, linewidth = 2, shade = True);
plt.title('Number of Leaves Distribution', size = 18); plt.xlabel('Number of Leaves', size = 16); plt.ylabel('Density', size = 16);
# boosting type domain 
boosting_type = {'boosting_type': hp.choice('boosting_type', 
                                            [{'boosting_type': 'gbdt', 'subsample': hp.uniform('subsample', 0.5, 1)}, 
                                             {'boosting_type': 'dart', 'subsample': hp.uniform('subsample', 0.5, 1)},
                                             {'boosting_type': 'goss', 'subsample': 1.0}])}

# Draw a sample
params = sample(boosting_type)
params
# Retrieve the subsample if present otherwise set to 1.0
subsample = params['boosting_type'].get('subsample', 1.0)

# Extract the boosting type
params['boosting_type'] = params['boosting_type']['boosting_type']
params['subsample'] = subsample

params
# Define the search space
space = {
   'class_weight': hp.choice('class_weight', [None, 'balanced']),
    'boosting_type': hp.choice('boosting_type', [{'boosting_type': 'gbdt', 'subsample': hp.uniform('gdbt_subsample', 0.5, 1)}, 
                                                 {'boosting_type': 'dart', 'subsample': hp.uniform('dart_subsample', 0.5, 1)},
                                                 {'boosting_type': 'goss', 'subsample': 1.0}]),
    'num_leaves': hp.quniform('num_leaves', 30, 150, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.05), np.log(0.2)),
    'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
    'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0)
}
# Sample from the full space
x = sample(space)

# Conditional logic to assign top-level keys
subsample = x['boosting_type'].get('subsample', 1.0)
x['boosting_type'] = x['boosting_type']['boosting_type']
x['subsample'] = subsample

x
x = sample(space)
subsample = x['boosting_type'].get('subsample', 1.0)
x['boosting_type'] = x['boosting_type']['boosting_type']
x['subsample'] = subsample
x
from hyperopt import tpe

# optimization algorithm
tpe_algorithm = tpe.suggest
from hyperopt import Trials

# Keep track of results
bayes_trials = Trials()
# File to save first results
import pandas as pd
df = pd.DataFrame(list())
df.to_csv('gbm_trials_titanic.csv')

out_file = 'gbm_trials_titanic.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['loss', 'params', 'iteration', 'estimators', 'train_time'])
of_connection.close()
from hyperopt import fmin
%%capture

# Global variable
global  ITERATION

ITERATION = 0

# Run optimization
best = fmin(fn = objective, space = space, algo = tpe.suggest, 
            max_evals = 100, trials = bayes_trials, rstate = np.random.RandomState(50))

results = pd.read_csv('gbm_trials_titanic.csv')

# Sort with best scores on top and reset index for slicing
results.sort_values('loss', ascending = True, inplace = True)
results.reset_index(inplace = True, drop = True)
print(results.params[0])
print(results.head())
def model_output(your_model):
    #training the model with inp and out df
    your_model.fit(X_train,y_train)
    your_model_pred= your_model.predict(test_titanic)
    your_model_df= pd.DataFrame({'PassengerId':test['PassengerId'],'Survived': your_model_pred})
    your_model_df.to_csv('submission_titanic_lgbmc_bayesian_prev_cat_21num_leaves.csv',index=False)

rdf_model=RandomForestClassifier(warm_start=True)
#xgb_model=XGBClassifier()
nbg_model=GaussianNB()
mplc_model=MLPClassifier()
adb_model= AdaBoostClassifier()
gbb_model=GradientBoostingClassifier()
svc_model= SVC()
knn_model=KNeighborsClassifier()
sgd_model=SGDClassifier()
lgbmc_model= LGBMClassifier(boosting_type=  'gbdt',objective='binary',random_state=42, max_depth=4,class_weight= None, colsample_bytree= 0.8175679964858448,
                            learning_rate= 0.05088656537291398, min_child_samples= 25, num_leaves= 21, 
                            reg_alpha= 0.5804082563063744, reg_lambda=  0.192747845878722, 
                            subsample_for_bin= 40000, subsample= 0.8293119621855407)


model_output(lgbmc_model)