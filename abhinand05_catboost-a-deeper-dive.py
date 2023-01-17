# Main Libs

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# Utility libs

from tqdm import tqdm

import time

import datetime

from skopt import gp_minimize

from skopt.space import Real, Integer

from skopt.utils import use_named_args

from skopt.plots import plot_convergence

from copy import deepcopy

import pprint

import shap

import os



# You might have to do !pip install catboost

# If you don't have it on your local machine

# nevertheless Kaggle runtimes come preinstalled with CatBoost

import catboost



from pathlib import Path

data_dir = Path('../input/data-science-bowl-2019')

os.listdir(data_dir)
%%time

train = pd.read_csv('../input/data-science-bowl-2019/train.csv')

labels = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')

test = pd.read_csv('../input/data-science-bowl-2019/test.csv')

specs = pd.read_csv('../input/data-science-bowl-2019/specs.csv')

sample_submission = pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')
train.head()
labels.head()
sample_submission.head()
list_of_user_activities = list(set(train['title'].value_counts().index).union(set(test['title'].value_counts().index)))

activities_map = dict(zip(list_of_user_activities, np.arange(len(list_of_user_activities))))



train['title'] = train['title'].map(activities_map)

test['title'] = test['title'].map(activities_map)

labels['title'] = labels['title'].map(activities_map)
win_code = dict(zip(activities_map.values(), (4100*np.ones(len(activities_map))).astype('int')))

win_code[activities_map['Bird Measurer (Assessment)']] = 4110



train['timestamp'] = pd.to_datetime(train['timestamp'])

test['timestamp'] = pd.to_datetime(test['timestamp'])
# Thanks for this beautiful function https://www.kaggle.com/mhviraf/a-new-baseline-for-dsb-2019-catboost-model 

def get_data(user_sample, test_set=False):

    last_activity = 0

    user_activities_count = {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}

    accuracy_groups = {0:0, 1:0, 2:0, 3:0}

    all_assessments = []

    accumulated_accuracy_group = 0

    accumulated_accuracy=0

    accumulated_correct_attempts = 0 

    accumulated_uncorrect_attempts = 0 

    accumulated_actions = 0

    counter = 0

    durations = []

    for i, session in user_sample.groupby('game_session', sort=False):

        session_type = session['type'].iloc[0]

        session_title = session['title'].iloc[0]

        if test_set == True:

            second_condition = True

        else:

            if len(session)>1:

                second_condition = True

            else:

                second_condition= False

            

        if (session_type == 'Assessment') & (second_condition):

            all_attempts = session.query(f'event_code == {win_code[session_title]}')

            true_attempts = all_attempts['event_data'].str.contains('true').sum()

            false_attempts = all_attempts['event_data'].str.contains('false').sum()

            features = user_activities_count.copy()

            features['session_title'] = session['title'].iloc[0] 

            features['accumulated_correct_attempts'] = accumulated_correct_attempts

            features['accumulated_uncorrect_attempts'] = accumulated_uncorrect_attempts

            accumulated_correct_attempts += true_attempts 

            accumulated_uncorrect_attempts += false_attempts

            if durations == []:

                features['duration_mean'] = 0

            else:

                features['duration_mean'] = np.mean(durations)

            durations.append((session.iloc[-1, 2] - session.iloc[0, 2] ).seconds)

            features['accumulated_accuracy'] = accumulated_accuracy/counter if counter > 0 else 0

            accuracy = true_attempts/(true_attempts+false_attempts) if (true_attempts+false_attempts) != 0 else 0

            accumulated_accuracy += accuracy

            if accuracy == 0:

                features['accuracy_group'] = 0

            elif accuracy == 1:

                features['accuracy_group'] = 3

            elif accuracy == 0.5:

                features['accuracy_group'] = 2

            else:

                features['accuracy_group'] = 1



            features.update(accuracy_groups)

            features['accumulated_accuracy_group'] = accumulated_accuracy_group/counter if counter > 0 else 0

            features['accumulated_actions'] = accumulated_actions

            accumulated_accuracy_group += features['accuracy_group']

            accuracy_groups[features['accuracy_group']] += 1

            if test_set == True:

                all_assessments.append(features)

            else:

                if true_attempts+false_attempts > 0:

                    all_assessments.append(features)

                

            counter += 1

        accumulated_actions += len(session)

        if last_activity != session_type:

            user_activities_count[session_type] += 1

            last_activitiy = session_type

    if test_set:

        return all_assessments[-1] 

    return all_assessments
compiled_data = []

for i, (ins_id, user_sample) in tqdm(enumerate(train.groupby('installation_id', sort=False)), total=17000):

    compiled_data += get_data(user_sample)
new_train = pd.DataFrame(compiled_data)

del compiled_data

print("Train Data Shape:")

new_train.shape
import gc

gc.collect()
all_features = [x for x in new_train.columns if x not in ['accuracy_group']]

cat_features = ['session_title']

X, y = new_train[all_features], new_train['accuracy_group']

del train
class ModelOptimizer:

    best_score = None

    opt = None

    

    def __init__(self, model, X_train, y_train, categorical_columns_indices=None, n_fold=3, seed=2405, early_stopping_rounds=30, is_stratified=True, is_shuffle=True):

        self.model = model

        self.X_train = X_train

        self.y_train = y_train

        self.categorical_columns_indices = categorical_columns_indices

        self.n_fold = n_fold

        self.seed = seed

        self.early_stopping_rounds = early_stopping_rounds

        self.is_stratified = is_stratified

        self.is_shuffle = is_shuffle

        

        

    def update_model(self, **kwargs):

        for k, v in kwargs.items():

            setattr(self.model, k, v)

            

    def evaluate_model(self):

        pass

    

    def optimize(self, param_space, max_evals=10, n_random_starts=2):

        start_time = time.time()

        

        @use_named_args(param_space)

        def _minimize(**params):

            self.model.set_params(**params)

            return self.evaluate_model()

        

        opt = gp_minimize(_minimize, param_space, n_calls=max_evals, n_random_starts=n_random_starts, random_state=2405, n_jobs=-1)

        best_values = opt.x

        optimal_values = dict(zip([param.name for param in param_space], best_values))

        best_score = opt.fun

        self.best_score = best_score

        self.opt = opt

        

        print('optimal_parameters: {}\noptimal score: {}\noptimization time: {}'.format(optimal_values, best_score, time.time() - start_time))

        print('updating model with optimal values')

        self.update_model(**optimal_values)

        plot_convergence(opt)

        return optimal_values

class CatboostOptimizer(ModelOptimizer):

    def evaluate_model(self):

        validation_scores = catboost.cv(

        catboost.Pool(self.X_train, 

                      self.y_train, 

                      cat_features=self.categorical_columns_indices),

        self.model.get_params(), 

        nfold=self.n_fold,

        stratified=self.is_stratified,

        seed=self.seed,

        early_stopping_rounds=self.early_stopping_rounds,

        shuffle=self.is_shuffle,

#         metrics='auc',

        plot=False)

        self.scores = validation_scores

        test_scores = validation_scores.iloc[:, 2]

        best_metric = test_scores.max()

        return 1 - best_metric
default_cb = catboost.CatBoostClassifier(loss_function='MultiClass',

                                         task_type='CPU',

                                         random_seed=12,

                                         silent=True

                                        )

default_cb_optimizer = CatboostOptimizer(default_cb, X, y)

default_cb_optimizer.evaluate_model()
greedy_cb = catboost.CatBoostClassifier(

    loss_function='MultiClass',

    task_type="CPU",

    learning_rate=0.01,

    iterations=2000,

    od_type="Iter",

    early_stopping_rounds=500,

    random_seed=24,

    silent=True

)
from sklearn.metrics import confusion_matrix

def qwk(act,pred,n=4,hist_range=(0,3)):

    

    O = confusion_matrix(act,pred)

    O = np.divide(O,np.sum(O))

    

    W = np.zeros((n,n))

    for i in range(n):

        for j in range(n):

            W[i][j] = ((i-j)**2)/((n-1)**2)

            

    act_hist = np.histogram(act,bins=n,range=hist_range)[0]

    prd_hist = np.histogram(pred,bins=n,range=hist_range)[0]

    

    E = np.outer(act_hist,prd_hist)

    E = np.divide(E,np.sum(E))

    

    num = np.sum(np.multiply(W,O))

    den = np.sum(np.multiply(W,E))

        

    return 1-np.divide(num,den)
cb_optimizer = CatboostOptimizer(greedy_cb, X, y)

params_space = [Real(0.01, 0.8, name='learning_rate'),]

cb_optimal_values = cb_optimizer.optimize(params_space)
cb = catboost.CatBoostClassifier(n_estimators=4000,

                         one_hot_max_size=2,

                         loss_function='MultiClass',

                         eval_metric='WKappa',

                         task_type='CPU',                

                         random_seed=5, 

                         use_best_model=True,

                         silent=True

                        )
one_cb_optimizer = CatboostOptimizer(cb, X, y)

params_space = [Real(0.01, 0.8, name='learning_rate'), 

                Integer(2, 10, name='max_depth'), 

                Real(0.5, 1.0, name='colsample_bylevel'), 

                Real(0.0, 100, name='bagging_temperature'), 

                Real(0.0, 100, name='random_strength'), 

                Real(1.0, 100, name='reg_lambda')]

one_cb_optimal_values = one_cb_optimizer.optimize(params_space, max_evals=40, n_random_starts=4)
one_cb_optimizer.model.get_params()
def make_classifier():

    clf = catboost.CatBoostClassifier(

            n_estimators = 4000,

            task_type = 'CPU',

            one_hot_max_size = 2,

            random_seed = 31,

            loss_function = 'MultiClass',

            learning_rate = 0.8,

            max_depth = 6,

            colsample_bylevel = 0.5,

            bagging_temperature = 28.635664398579774,

            random_strength = 100.0,

            reg_lambda = 100.0,

            early_stopping_rounds=500,

    )

    return clf

oof = np.zeros(len(X))
from sklearn.model_selection import KFold

oof = np.zeros(len(X))

NFOLDS = 5

folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=2019)



training_start_time = time.time()

for fold, (trn_idx, test_idx) in enumerate(folds.split(X, y)):

    start_time = time.time()

    print(f'Training on fold {fold+1}')

    clf = make_classifier()

    clf.fit(X.loc[trn_idx, all_features], y.loc[trn_idx], eval_set=(X.loc[test_idx, all_features], y.loc[test_idx]),

                          use_best_model=True, verbose=500, cat_features=cat_features)    

    oof[test_idx] = clf.predict(X.loc[test_idx, all_features]).reshape(len(test_idx))

    print('Fold {} finished in {}'.format(fold + 1, str(datetime.timedelta(seconds=time.time() - start_time))))

    

print('-' * 30)

print('OOF QWK:', qwk(y, oof))

print('-' * 30)
# train model on all data once

clf = make_classifier()

clf.fit(X, y, verbose=500, cat_features=cat_features)
# process test set

new_test = []

for ins_id, user_sample in tqdm(test.groupby('installation_id', sort=False), total=1000):

    a = get_data(user_sample, test_set=True)

    new_test.append(a)

    

X_test = pd.DataFrame(new_test)

del test
# make predictions on test set once

preds = clf.predict(X_test)

del X_test
sample_submission['accuracy_group'] = np.round(preds).astype('int')

sample_submission.to_csv('submission.csv', index=None)

sample_submission.head()
sample_submission['accuracy_group'].plot(kind='hist')
labels['accuracy_group'].plot(kind='hist')
pd.Series(oof).plot(kind='hist')
clf = deepcopy(one_cb_optimizer.model)

pool = catboost.Pool(X, y, cat_features=cat_features)

clf.set_params(use_best_model=False, reg_lambda=1.0)

clf.fit(pool, use_best_model=False)

interactions = clf.get_feature_importance(pool, fstr_type=catboost.EFstrType.Interaction, prettified=True)

shap_values = clf.get_feature_importance(pool, fstr_type=catboost.EFstrType.ShapValues,prettified=True)
feature_interaction = [[X.columns[interaction[0]], X.columns[interaction[1]], interaction[2]] for i,interaction in interactions.iterrows()]

feature_interaction_df = pd.DataFrame(feature_interaction, columns=['feature1', 'feature2', 'interaction_strength'])

feature_interaction_df.head(10)
pd.Series(index=zip(feature_interaction_df['feature1'], feature_interaction_df['feature2']), data=feature_interaction_df['interaction_strength'].values, name='interaction_strength').head(10).plot(kind='barh', figsize=(18, 10), fontsize=16, color='b')
shap.initjs()

shap.summary_plot(shap_values[:, 0, :-1], X, feature_names=X.columns.tolist())
shap.initjs()

shap.summary_plot(shap_values[:, 1, :-1], X, feature_names=X.columns.tolist())
shap.initjs()

shap.summary_plot(shap_values[:, 2, :-1], X, feature_names=X.columns.tolist())
shap.initjs()

shap.summary_plot(shap_values[:, 3, :-1], X, feature_names=X.columns.tolist())
shap.summary_plot(shap_values[:, 0,:-1], X, feature_names=X.columns.tolist(), plot_type="bar")
shap.dependence_plot("accumulated_accuracy", shap_values[:, 3, :-1], X)