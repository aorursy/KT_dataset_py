import os #to access files

import pandas as pd #to work with dataframes

import numpy as np #just a tradition

from sklearn.model_selection import StratifiedKFold #for cross-validation

from sklearn.metrics import roc_auc_score #this is we are trying to increase

import matplotlib.pyplot as plt #we will plot something at the end)

import seaborn as sns #same reason

import lightgbm as lgb #the model we gonna use
%%time

PATH_TO_DATA = '../input/'



df_train_features = pd.read_csv(os.path.join(PATH_TO_DATA, 

                                             'train_features.csv'), 

                                    index_col='match_id_hash')

df_train_targets = pd.read_csv(os.path.join(PATH_TO_DATA, 

                                            'train_targets.csv'), 

                                   index_col='match_id_hash')

df_test_features = pd.read_csv(os.path.join(PATH_TO_DATA, 'test_features.csv'), 

                                   index_col='match_id_hash')
df_train_features.head(2)
df_train_targets.head(2)
df_test_features.head(2)
#turn to X and y notations for train data and target

X = df_train_features.values

y = df_train_targets['radiant_win'].values #extract the colomn we need
#this is to make sure we have "ujson" and "tqdm"

try:

    import ujson as json

except ModuleNotFoundError:

    import json

    print ('Please install ujson to read JSON oblects faster')

    

try:

    from tqdm import tqdm_notebook

except ModuleNotFoundError:

    tqdm_notebook = lambda x: x

    print ('Please install tqdm to track progress with Python loops')
#a helper function, we will use it in next cell

def read_matches(matches_file):

    

    MATCHES_COUNT = {

        'test_matches.jsonl': 10000,

        'train_matches.jsonl': 39675,

    }

    _, filename = os.path.split(matches_file)

    total_matches = MATCHES_COUNT.get(filename)

    

    with open(matches_file) as fin:

        for line in tqdm_notebook(fin, total=total_matches):

            yield json.loads(line)
def add_new_features(df_features, matches_file):

    

    # Process raw data and add new features

    for match in read_matches(matches_file):

        match_id_hash = match['match_id_hash']



        # Counting ruined towers for both teams

        radiant_tower_kills = 0

        dire_tower_kills = 0

        for objective in match['objectives']:

            if objective['type'] == 'CHAT_MESSAGE_TOWER_KILL':

                if objective['team'] == 2:

                    radiant_tower_kills += 1

                if objective['team'] == 3:

                    dire_tower_kills += 1



        # Write new features

        df_features.loc[match_id_hash, 'radiant_tower_kills'] = radiant_tower_kills

        df_features.loc[match_id_hash, 'dire_tower_kills'] = dire_tower_kills

        df_features.loc[match_id_hash, 'diff_tower_kills'] = radiant_tower_kills - dire_tower_kills

        

        #let's add one more

        df_features.loc[match_id_hash, 'ratio_tower_kills'] = radiant_tower_kills / (0.01+dire_tower_kills)

        # ... here you can add more features ...

        
%%time

# copy the dataframe with features

df_train_features_extended = df_train_features.copy()

df_test_features_extended = df_test_features.copy()



# add new features

add_new_features(df_train_features_extended, os.path.join(PATH_TO_DATA, 'train_matches.jsonl'))

add_new_features(df_test_features_extended, os.path.join(PATH_TO_DATA, 'test_matches.jsonl'))
#Just a shorter names for data

newtrain=df_train_features_extended

newtest=df_test_features_extended

target=pd.DataFrame(y)
#lastly, check the shapes, Andrew Ng approved)

newtrain.shape,target.shape, newtest.shape
features=newtrain.columns
param = {

        'bagging_freq': 5,  #handling overfitting

        'bagging_fraction': 0.5,  #handling overfitting - adding some noise

        'boost_from_average':'false',

        'boost': 'gbdt',

        'feature_fraction': 0.05, #handling overfitting

        'learning_rate': 0.01,  #the changes between one auc and a better one gets really small thus a small learning rate performs better

        'max_depth': -1,  

        'metric':'auc',

        'min_data_in_leaf': 50,

        'min_sum_hessian_in_leaf': 10.0,

        'num_leaves': 10,

        'num_threads': 5,

        'tree_learner': 'serial',

        'objective': 'binary', 

        'verbosity': 1

    }
%%time

#divide training data into train and validaton folds

folds = StratifiedKFold(n_splits=5, shuffle=False, random_state=17)



#placeholder for out-of-fold, i.e. validation scores

oof = np.zeros(len(newtrain))



#for predictions

predictions = np.zeros(len(newtest))



#and for feature importance

feature_importance_df = pd.DataFrame()



#RUN THE LOOP OVER FOLDS

for fold_, (trn_idx, val_idx) in enumerate(folds.split(newtrain.values, target.values)):

    

    X_train, y_train = newtrain.iloc[trn_idx], target.iloc[trn_idx]

    X_valid, y_valid = newtrain.iloc[val_idx], target.iloc[val_idx]

    

    print("Computing Fold {}".format(fold_))

    trn_data = lgb.Dataset(X_train, label=y_train)

    val_data = lgb.Dataset(X_valid, label=y_valid)



    

    num_round = 5000 

    verbose=1000 

    stop=500 

    

    #TRAIN THE MODEL

    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=verbose, early_stopping_rounds = stop)

    

    #CALCULATE PREDICTION FOR VALIDATION SET

    oof[val_idx] = clf.predict(newtrain.iloc[val_idx], num_iteration=clf.best_iteration)

    

    #FEATURE IMPORTANCE

    fold_importance_df = pd.DataFrame()

    fold_importance_df["Feature"] = features

    fold_importance_df["importance"] = clf.feature_importance()

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    

    #CALCULATE PREDICTIONS FOR TEST DATA, using best_iteration on the fold

    predictions += clf.predict(newtest, num_iteration=clf.best_iteration) / folds.n_splits



#print overall cross-validatino score

print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))
cols = (feature_importance_df[["Feature", "importance"]]

        .groupby("Feature")

        .mean()

        .sort_values(by="importance", ascending=False)[:150].index)

best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]



plt.figure(figsize=(14,28))

sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False))

plt.title('Features importance (averaged/folds)')

plt.tight_layout()

plt.savefig('FI.png')
df_submission = pd.DataFrame({'radiant_win_prob': predictions}, 

                                 index=df_test_features.index)

import datetime

submission_filename = 'submission_{}.csv'.format(

    datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

df_submission.to_csv(submission_filename)

print('Submission saved to {}'.format(submission_filename))
from bayes_opt import BayesianOptimization
def lgb_eval(num_leaves, feature_fraction, bagging_fraction, max_depth, lambda_l1, lambda_l2, min_split_gain, min_child_weight):

    params = {'application':'binary','num_iterations':4000, 'learning_rate':0.05, 'early_stopping_round':100, 'metric':'auc'}

    params["num_leaves"] = round(num_leaves)

    params['feature_fraction'] = max(min(feature_fraction, 1), 0)

    params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)

    params['max_depth'] = round(max_depth)

    params['lambda_l1'] = max(lambda_l1, 0)

    params['lambda_l2'] = max(lambda_l2, 0)

    params['min_split_gain'] = min_split_gain

    params['min_child_weight'] = min_child_weight

    cv_result = lgb.cv(params, train_data, nfold=n_folds, seed=random_seed, stratified=True, verbose_eval =200, metrics=['auc'])

    return max(cv_result['auc-mean'])

lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (24, 45),

                                        'feature_fraction': (0.1, 0.9),

                                        'bagging_fraction': (0.8, 1),

                                        'max_depth': (5, 8.99),

                                        'lambda_l1': (0, 5),

                                        'lambda_l2': (0, 3),

                                        'min_split_gain': (0.001, 0.1),

                                        'min_child_weight': (5, 50)}, random_state=0)
X = newtrain.values

y = target.values.ravel()
def bayes_parameter_opt_lgb(X, y, init_round=15, opt_round=25, n_folds=5, random_seed=6, n_estimators=10000, learning_rate=0.05, output_process=False):

    # prepare data

    train_data = lgb.Dataset(data=X, label=y, free_raw_data=False) #categorical_feature = categorical_feats

    # parameters

    def lgb_eval(num_leaves, feature_fraction, bagging_fraction, max_depth, lambda_l1, lambda_l2, min_split_gain, min_child_weight):

        params = {'application':'binary', 'metric':'auc'} #'num_iterations': n_estimators, 'learning_rate':learning_rate, 'early_stopping_round':100

        params["num_leaves"] = int(round(num_leaves))

        params['feature_fraction'] = max(min(feature_fraction, 1), 0)

        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)

        params['max_depth'] = int(round(max_depth))

        params['lambda_l1'] = max(lambda_l1, 0)

        params['lambda_l2'] = max(lambda_l2, 0)

        params['min_split_gain'] = min_split_gain

        params['min_child_weight'] = min_child_weight

        cv_result = lgb.cv(params, train_data, nfold=n_folds, seed=random_seed, stratified=True, verbose_eval =200, metrics=['auc'])

        return max(cv_result['auc-mean'])

    # range 

    lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (24, 45),

                                            'feature_fraction': (0.1, 0.9),

                                            'bagging_fraction': (0.8, 1),

                                            'max_depth': (5, 8.99),

                                            'lambda_l1': (0, 5),

                                            'lambda_l2': (0, 3),

                                            'min_split_gain': (0.001, 0.1),

                                            'min_child_weight': (5, 50)}, random_state=0)

    # optimize

    lgbBO.maximize(init_points=init_round, n_iter=opt_round)

    

    # output optimization process

    if output_process==True: lgbBO.points_to_csv("bayes_opt_result.csv")

    

    # return best parameters

    return lgbBO



opt_params = bayes_parameter_opt_lgb(X, y, init_round=5, opt_round=10, n_folds=3, random_seed=6, n_estimators=100, learning_rate=0.05)
#print(opt_params.res['max']['max_params'])
print("Final result:", opt_params.max)