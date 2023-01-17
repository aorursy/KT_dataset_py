
%matplotlib inline

import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style='darkgrid')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy.random as nr
import math
from sklearn import preprocessing
import sklearn.model_selection as ms
from sklearn import linear_model
import sklearn.metrics as sklm
import scipy.stats as ss
data=pd.read_csv("../input/train_values.csv")
accept=pd.read_csv('../input/train_labels.csv')
data['accepted']=accept['accepted']

data2=pd.read_csv('../input/test_values.csv')
data['co_applicant'] = data['co_applicant'].map({False:0,True:1}).astype(np.int)
data['loan_ratio']=data['applicant_income']/data['loan_amount']
data['applicant_ratio']=(data['ffiecmedian_family_income'])/data['applicant_income']
data.minority_population_pct=data.minority_population_pct.fillna(data.minority_population_pct.median())
data['number_of_owner-occupied_units']=data['number_of_owner-occupied_units'].fillna(data['number_of_owner-occupied_units'].median())
data.ffiecmedian_family_income=data.ffiecmedian_family_income.fillna(data.ffiecmedian_family_income.median())
data.population=data.population.fillna(data.population.median())
data.number_of_1_to_4_family_units=data.number_of_1_to_4_family_units.fillna(data.number_of_1_to_4_family_units.median())
data.tract_to_msa_md_income_pct=data.tract_to_msa_md_income_pct.fillna(data.tract_to_msa_md_income_pct.median())
data['applicant_income']=data['applicant_income'].fillna(data['applicant_income'].median())
data['log_loan']=np.log(data['loan_amount'])
data['log_income']=np.log(data['applicant_income'])

y=data.groupby(['msa_md','lender']).agg({'accepted': 'mean'})
data = pd.merge(data,y, on=['msa_md','lender'], how='left')
p=data.groupby(['msa_md','loan_type','loan_purpose','applicant_race']).agg({'accepted_x': 'mean'})
data = pd.merge(data,p, on=['msa_md','loan_type','loan_purpose','applicant_race'], how='left')
v=data.groupby(['msa_md','applicant_race','applicant_ethnicity']).agg({'accepted_x_x': 'mean'})
data = pd.merge(data,v, on=['msa_md','applicant_race','applicant_ethnicity'], how='left')
k=data.groupby(['state_code','loan_type']).agg({'accepted_x_x_x': 'mean'})
data = pd.merge(data,k, on=['state_code','loan_type'], how='left')
q=data.groupby(['state_code','county_code']).agg({'accepted_x_x_x_x': 'mean'})
data = pd.merge(data,q, on=['state_code','county_code'], how='left')
f=data.groupby(['msa_md','loan_type','preapproval','applicant_race']).agg({'accepted_x_x_x_x_x': 'mean'})
data = pd.merge(data,f, on=['msa_md','loan_type','preapproval','applicant_race'], how='left')
data=data.fillna(-996)
data2['co_applicant'] = data2['co_applicant'].map({False:0,True:1}).astype(np.int)
data2['loan_ratio']=data2['applicant_income']/data2['loan_amount']
data2['applicant_ratio']=(data2['ffiecmedian_family_income'])/data2['applicant_income']
data2.minority_population_pct=data2.minority_population_pct.fillna(data2.minority_population_pct.median())
data2['number_of_owner-occupied_units']=data2['number_of_owner-occupied_units'].fillna(data2['number_of_owner-occupied_units'].median())
data2.ffiecmedian_family_income=data2.ffiecmedian_family_income.fillna(data2.ffiecmedian_family_income.median())
data2.population=data2.population.fillna(data2.population.median())
data2.number_of_1_to_4_family_units=data2.number_of_1_to_4_family_units.fillna(data2.number_of_1_to_4_family_units.median())
data2.tract_to_msa_md_income_pct=data2.tract_to_msa_md_income_pct.fillna(data2.tract_to_msa_md_income_pct.median())
data2['applicant_income']=data2['applicant_income'].fillna(data2['applicant_income'].median())
data2['log_loan']=np.log(data2['loan_amount'])
data2['log_income']=np.log(data2['applicant_income'])

data2 = pd.merge(data2,y, on=['msa_md','lender'], how='left')
data2.rename(columns={'accepted':'accepted_y'}, inplace=True)
data2['accepted_y']=data2['accepted_y'].fillna(data2['accepted_y'].mean())
data2 = pd.merge(data2,p, on=['msa_md','loan_type','loan_purpose','applicant_race'], how='left')
data2.rename(columns={'accepted_x':'accepted_x_y'}, inplace=True)
data2['accepted_x_y']=data2['accepted_x_y'].fillna(data2['accepted_x_y'].mean())
data2 = pd.merge(data2,v, on=['msa_md','applicant_race','applicant_ethnicity'], how='left')
data2.rename(columns={'accepted_x_x':'accepted_x_x_y'}, inplace=True)
data2 = pd.merge(data2,k, on=['state_code','loan_type'], how='left')
data2.rename(columns={'accepted_x_x_x':'accepted_x_x_x_y'}, inplace=True)
data2 = pd.merge(data2,q, on=['state_code','county_code'], how='left')
data2.rename(columns={'accepted_x_x_x_x':'accepted_x_x_x_x_y'}, inplace=True)
data2 = pd.merge(data2,f, on=['msa_md','loan_type','preapproval','applicant_race'], how='left')
data2.rename(columns={'accepted_x_x_x_x_x':'accepted_x_x_x_x_x_y'}, inplace=True)

data2=data2.fillna(-996)
data2.describe()
data=data.fillna(-996)

!pip install -q numpy pandas catboost hyperopt scikit-learn matplotlib
import catboost

from __future__ import absolute_import, division, print_function, unicode_literals

import catboost as cb
import catboost.datasets as cbd
import catboost.utils as cbu
import hyperopt
import sys


from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=0.3)
train.head()

def get_fixed_adult(): 
    X_train, y_train = train.drop('accepted_x_x_x_x_x_x', axis=1), train.accepted_x_x_x_x_x_x
    X_test, y_test = test.drop('accepted_x_x_x_x_x_x', axis=1), test.accepted_x_x_x_x_x_x
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = get_fixed_adult()
X_train.head(30)

print('train: {}\ntest: {}'.format(y_train.value_counts(), y_test.value_counts()))

class UciAdultClassifierObjective(object):
    def __init__(self, dataset, const_params, fold_count):
        self._dataset = dataset
        self._const_params = const_params.copy()
        self._fold_count = fold_count
        self._evaluated_count = 0
        
    def _to_catboost_params(self, hyper_params):
        return {
            'learning_rate': hyper_params['learning_rate'],
            'depth': hyper_params['depth'],
            'l2_leaf_reg': hyper_params['l2_leaf_reg']}
    
    # hyperopt optimizes an objective using `__call__` method (e.g. by doing 
    # `foo(hyper_params)`), so we provide one
    def __call__(self, hyper_params):
        # join hyper-parameters provided by hyperopt with hyper-parameters 
        # provided by the user
        params = self._to_catboost_params(hyper_params)
        params.update(self._const_params)
        
        print('evaluating params={}'.format(params), file=sys.stdout)
        sys.stdout.flush()
        
        # we use cross-validation for objective evaluation, to avoid overfitting
        scores = cb.cv(
            pool=self._dataset,
            params=params,
            fold_count=self._fold_count,
            partition_random_seed=20181224,
            verbose=False)
        
        # scores returns a dictionary with mean and std (per-fold) of metric 
        # value for each cv iteration, we choose minimal value of objective 
                # mean (though it will be better to choose minimal value among all folds)
        # because noise is additive
        max_mean_auc = np.max(scores['test-AUC-mean'])
        print('evaluated score={}'.format(max_mean_auc), file=sys.stdout)
        
        self._evaluated_count += 1
        print('evaluated {} times'.format(self._evaluated_count), file=sys.stdout)
        
        # negate because hyperopt minimizes the objective
        return {'loss': -max_mean_auc, 'status': hyperopt.STATUS_OK}
def find_best_hyper_params(dataset, const_params, max_evals=100):    
    # we are going to optimize these three parameters, though there are a lot more of them (see CatBoost docs)
    parameter_space = {
        'learning_rate': hyperopt.hp.uniform('learning_rate', 0.2, 1.0),
        'depth': hyperopt.hp.randint('depth', 7),
        'l2_leaf_reg': hyperopt.hp.uniform('l2_leaf_reg', 1, 10)}
    objective = UciAdultClassifierObjective(dataset=dataset, const_params=const_params, fold_count=6)
    trials = hyperopt.Trials()
    best = hyperopt.fmin(
        fn=objective,
        space=parameter_space,
        algo=hyperopt.rand.suggest,
        max_evals=max_evals,
        rstate=np.random.RandomState(seed=20181224))
    return best

def train_best_model(X, y, const_params, max_evals=100, use_default=False):
    # convert pandas.DataFrame to catboost.Pool to avoid converting it on each 
    # iteration of hyper-parameters optimization
    dataset = cb.Pool(X, y, cat_features=np.where(X.dtypes != np.float)[0])
    
    if use_default:
        # pretrained optimal parameters
        best = {
            'learning_rate': 0.11234185321620083, 
            'depth': 8, 
            'l2_leaf_reg': 9.464266235679002}
    else:
        best = find_best_hyper_params(dataset, const_params, max_evals=max_evals)
    
    # merge subset of hyper-parameters provided by hyperopt with hyper-parameters 
    # provided by the user
    hyper_params = best.copy()
    hyper_params.update(const_params)
    
    # drop `use_best_model` because we are going to use entire dataset for 
    # training of the final model
    hyper_params.pop('use_best_model', None)
    
    model = cb.CatBoostClassifier(**hyper_params)
    model.fit(dataset, verbose=False)
    
    return model, hyper_params

# make it True if your want to use GPU for training
have_gpu = False
# skip hyper-parameter optimization and just use provided optimal parameters
use_optimal_pretrained_params = True
# number of iterations of hyper-parameter search
hyperopt_iterations = 50

const_params = dict({
    'task_type': 'GPU' if have_gpu else 'CPU',
    'loss_function': 'Logloss',
    'eval_metric': 'AUC', 
    'custom_metric': ['AUC'],
    'iterations': 1000,
    'random_seed': 20181224})

model, params = train_best_model(
    X_train, y_train, 
    const_params, 
    max_evals=hyperopt_iterations, 
    use_default=use_optimal_pretrained_params)
print('best params are {}'.format(params), file=sys.stdout)
def calculate_score_on_dataset_and_show_graph(X, y, model):
    import sklearn.metrics
    import matplotlib.pylab as pl
    pl.style.use('ggplot')
    
    dataset = cb.Pool(X, y, cat_features=np.where(X.dtypes != np.float)[0])
    fpr, tpr, _ = cbu.get_roc_curve(model, dataset, plot=True)
    auc = sklearn.metrics.auc(fpr, tpr)
    return auc
calculate_score_on_dataset_and_show_graph(X_test, y_test, model)
data2['accepted']=model.predict(data2)
data2['accepted']=data2['accepted'].astype(int)
data3=data2[['row_id','accepted']]
data3.to_csv('predictionff.csv', encoding='utf-8', index=False)
data3.accepted.value_counts()