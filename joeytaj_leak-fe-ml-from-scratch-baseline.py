"""
this kernel uses code and data from these kernels:
https://www.kaggle.com/dfrumkin/a-simple-way-to-use-giba-s-features/notebook
https://www.kaggle.com/johnfarrell/giba-s-property-extended-result
https://www.kaggle.com/titericz/the-property-by-giba
"""
import numpy as np
import pandas as pd

import os
import datetime

import gc

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from math import sqrt
import math

import lightgbm as lgb

from tqdm import tqdm_notebook

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
#This code is borrowed from a kernel. Not sure
all_features = [c for c in test.columns if c not in ['ID']]
def has_ugly(row):
    for v in row.values[row.values > 0]:
        if str(v)[::-1].find('.') > 2:
            return True
    return False
test['has_ugly'] = test[all_features].apply(has_ugly, axis=1)
test_og = test[['ID']].copy()
test_og['nonzero_mean'] = test[[c for c in test.columns if c not in ['ID', 'has_ugly']]].apply(lambda x: np.expm1(np.log1p(x[x!=0]).mean()), axis=1)
test = test[test.has_ugly == False]

train_t = train.drop(['target'], axis = 1, inplace=False)
train_t.set_index('ID', inplace=True)
train_t = train_t.T
test_t = test.set_index('ID', inplace=False)
test_t = test_t.T
gc.collect()

features = ['f190486d6', '58e2e02e6', 'eeb9cd3aa', '9fd594eec', '6eef030c1',
        '15ace8c9f', 'fb0f5dbfe', '58e056e12', '20aa07010', '024c577b9',
        'd6bb78916', 'b43a7cfd5', '58232a6fb', '1702b5bf0', '324921c7b',
        '62e59a501', '2ec5b290f', '241f0f867', 'fb49e4212', '66ace2992',
        'f74e8f13d', '5c6487af1', '963a49cdc', '26fc93eb7', '1931ccfdd',
        '703885424', '70feb1494', '491b9ee45', '23310aa6f', 'e176a204a',
        '6619d81fc', '1db387535', 'fc99f9426', '91f701ba2', '0572565c2',
        '190db8488', 'adb64ff71', 'c47340d97', 'c5a231d81', '0ff32eb98']

extra_features = []
#run this iteratively until you have no more links. Then prune
def chain_pairs(ordered_items):
    ordered_chains = []
    links_found = 0
    for i_1, op_chain in enumerate(ordered_items.copy()[:]):
        if op_chain[0] != op_chain[1]:
            end_chain = op_chain[-1]
            for i_2, op in enumerate(ordered_items.copy()[:]):
                if (end_chain == op[0]):
                    links_found += 1
                    op_chain.extend(op[1:])
                    end_chain = op_chain[-1]

            ordered_chains.append(op_chain)
    return links_found, ordered_chains

def prune_chain(ordered_chain):
    
    ordered_chain = sorted(ordered_chain, key=len, reverse=True)
    new_chain = []
    id_lookup = {}
    for oc in ordered_chain:
        id_already_in_chain = False
        for idd in oc:
            if idd in id_lookup:
                id_already_in_chain = True
            id_lookup[idd] = idd

        if not id_already_in_chain:
            new_chain.append(oc)
    return sorted(new_chain, key=len, reverse=True)

def find_new_ordered_features(ordered_ids, data_t):
    data = data_t.copy()
    
    f1 = ordered_ids[0][:-1]
    f2 = ordered_ids[0][1:]
    for ef in ordered_ids[1:]:
        f1 += ef[:-1]
        f2 += ef[1:]
            
    d1 = data[f1].apply(tuple, axis=1).apply(hash).to_frame().rename(columns={0: 'key'})
    d1['ID'] = data.index
    gc.collect()
    d2 = data[f2].apply(tuple, axis=1).apply(hash).to_frame().rename(columns={0: 'key'})
    d2['ID'] = data.index
    gc.collect()
    d3 = d2[~d2.duplicated(['key'], keep=False)]
    d4 = d1[~d1.duplicated(['key'], keep=False)]
    d5 = d4.merge(d3, how='inner', on='key')

    d_feat = d1.merge(d5, how='left', on='key')
    d_feat.fillna(0, inplace=True)

    ordered_features = list(d_feat[['ID_x', 'ID_y']][d_feat.ID_x != 0].apply(list, axis=1))
    del d1,d2,d3,d4,d5,d_feat
    gc.collect()
    
    links_found = 1
    #print(ordered_features)
    while links_found > 0:
        links_found, ordered_features = chain_pairs(ordered_features)
        #print(links_found)
    
    ordered_features = prune_chain(ordered_features)
    
    #make lookup of all features found so far
    found = {}
    for ef in extra_features:
        found[ef[0]] = ef
        #print (ef[0])
    found [features[0]] = features

    #make list of sets of 40 that have not been found yet
    new_feature_sets = []
    for of in ordered_features:
        if len(of) >= 40:
            if of[0] not in found:
                new_feature_sets.append(of)
                
    return new_feature_sets

def add_new_feature_sets(data, data_t):
    
    print ('\nData Shape:', data.shape)
    f1 = features[:-1]
    f2 = features[1:]

    for ef in extra_features:
        f1 += ef[:-1]
        f2 += ef[1:]

    d1 = data[f1].apply(tuple, axis=1).apply(hash).to_frame().rename(columns={0: 'key'})
    d1['ID'] = data['ID']    
    gc.collect()
    d2 = data[f2].apply(tuple, axis=1).apply(hash).to_frame().rename(columns={0: 'key'})
    d2['ID'] = data['ID']
    gc.collect()
    #print('here')
    d3 = d2[~d2.duplicated(['key'], keep=False)]
    del d2
    d4 = d1[~d1.duplicated(['key'], keep=False)]
    #print('here')
    d5 = d4.merge(d3, how='inner', on='key')
    del d4
    d = d1.merge(d5, how='left', on='key')
    d.fillna(0, inplace=True)
    #print('here')
    ordered_ids = list(d[['ID_x', 'ID_y']][d.ID_x != 0].apply(list, axis=1))
    del d1,d3,d5,d
    gc.collect()

    links_found = 1
    while links_found > 0:
        links_found, ordered_ids = chain_pairs(ordered_ids)
        #print(links_found)

    print ('OrderedIds:', len(ordered_ids))
    #Make distinct ordered id chains
    ordered_ids = prune_chain(ordered_ids)
    print ('OrderedIds Pruned:', len(ordered_ids))

    #look for ordered features with new ordered id chains
    new_feature_sets = find_new_ordered_features(ordered_ids, data_t)    

    extra_features.extend(new_feature_sets)
    print('New Feature Count:', len(new_feature_sets))
    print('Extra Feature Count:', len(extra_features))
%%time

add_new_feature_sets(train,train_t)
add_new_feature_sets(test,test_t)
add_new_feature_sets(train,train_t)
add_new_feature_sets(test,test_t)
add_new_feature_sets(train,train_t)
with open("extra_features_{}.txt".format(len(extra_features)), "w") as text_file:
    for ef in extra_features:
        text_file.write(','.join(ef) + '\n')
del train_t, test_t, test
gc.collect()
#now that memory is cleared we can get back full test
test = pd.read_csv('../input/test.csv')
test['has_ugly'] = test[all_features].apply(has_ugly, axis=1)
test[test.has_ugly == True] = 0

def get_log_pred(data, feats, extra_feats, offset = 2):
    f1 = feats[:(offset * -1)]
    f2 = feats[offset:]
    for ef in extra_feats:
        f1 += ef[:(offset * -1)]
        f2 += ef[offset:]
        
    d1 = data[f1].apply(tuple, axis=1).apply(hash).to_frame().rename(columns={0: 'key'})
    d2 = data[f2].apply(tuple, axis=1).apply(hash).to_frame().rename(columns={0: 'key'})
    d2['pred'] = data[feats[offset-2]]
    d2 = d2[d2['pred'] != 0] # Keep?
    d3 = d2[~d2.duplicated(['key'], keep=False)]
    d4 = d1[~d1.duplicated(['key'], keep=False)]
    d5 = d4.merge(d3, how='inner', on='key')
        
    d = d1.merge(d5, how='left', on='key')
    return np.log1p(d.pred).fillna(0)

end_offset = 39
pred_test = []
pred_train = []
efs = extra_features
for o in tqdm_notebook(list(range(2, end_offset))):
    print('Offset:', o)

    log_pred = get_log_pred(train, features, extra_features, o)
    pred_train.append(np.expm1(log_pred))
    have_data = log_pred != 0
    train_count = have_data.sum()
    score = sqrt(mean_squared_error(np.log1p(train.target[have_data]), log_pred[have_data]))
    print(f'Score = {score} on {have_data.sum()} out of {train.shape[0]} training samples')


    log_pred_test = get_log_pred(test, features, efs, o)
    pred_test.append(np.expm1(log_pred_test))
    have_data = log_pred_test != 0
    test_count = have_data.sum()
    print(f'Have predictions for {have_data.sum()} out of {test.shape[0]} test samples')

    
pred_train_final = pred_train[0].copy()
for r in range(1,len(pred_train)):
    pred_train_final[pred_train_final == 0] = pred_train[r][pred_train_final == 0]

train_leak_match_count = (pred_train_final!=0).sum();
no_match_count = (pred_train_final==0).sum();
print ("Train leak count: ", train_leak_match_count, "Train no leak count: ",  no_match_count)

pred_train_temp = pred_train_final.copy()
train["nonzero_mean"] = train[[f for f in train.columns if f not in ["ID", "target","nonzero_mean"]]].apply(lambda x: np.expm1(np.log1p(x[x!=0]).mean()), axis=1)
pred_train_temp[pred_train_temp==0] = train['nonzero_mean'][pred_train_temp==0]
print(f'Baseline Train Score = {sqrt(mean_squared_error(np.log1p(train.target), np.log1p(pred_train_temp)))}')
pred_test_final = pred_test[0].copy()
for r in range(1,len(pred_test)):
    pred_test_final[pred_test_final == 0] = pred_test[r][pred_test_final == 0]
    
##https://www.kaggle.com/rsakata/21st-place-solution-bug-fixed-private-0-52785
pred_test_final[(4e+07 < pred_test_final)] = 4e+07
pred_test_final[((pred_test_final < 29000) & (pred_test_final > 0))] = 30000
##https://www.kaggle.com/c/santander-value-prediction-challenge/discussion/63931
pred_test_final[test.ID == 'd72fad286'] = 1560000
pred_test_final[test.ID == 'a304cde42'] = 320000.0

test_leak_match_count = (pred_test_final!=0).sum();
no_match_count = (pred_test_final==0).sum();
print ("Test leak count: ", test_leak_match_count, "Test no leak count: ",  no_match_count)
##Make Leak Baseline
pred_test_temp = pred_test_final.copy()
test_og["nonzero_mean"] = test_og[[f for f in test_og.columns if f not in ["ID", "target", "nonzero_mean", "has_ugly"]]].apply(lambda x: np.expm1(np.log1p(x[x!=0]).mean()), axis=1)
pred_test_temp[pred_test_temp==0] = test_og['nonzero_mean'][pred_test_temp==0]
test_og['target']=pred_test_temp
test_og[['ID', 'target']].to_csv('leak_baseline_{}.csv'.format(test_leak_match_count), index=False)


test_leaks = pd.read_csv('../input/sample_submission.csv')
del test_leaks['target']
test_leaks['target']=pred_test_final
test_leaks.to_csv('leak_only_{}.csv'.format(test_leak_match_count), index=False)

extra_features_list = []

for ef in extra_features:
    extra_features_list.extend(ef)

extra_features_list.extend(features)
len(extra_features_list)
#This makes the 100 40 length feature groups into 40 100 length feature groups. 
#These 100 size vectors is what I would have liked to feed into an LSTM\CNN but I never got a chance to try this
feats = pd.DataFrame(extra_features) 
time_features = []
for c in feats.columns[:]:    
    time_features.append([f for f in feats[c].values if f is not None])
    

#Make a bunch of different feature groups to build aggregates from
agg_features = []
all_cols = train.columns.drop(['ID', 'target', 'nonzero_mean'])
agg_features.append(all_cols)
agg_features.append([c for c in all_cols if c not in extra_features_list])
agg_features.append(extra_features_list)
agg_features.extend(time_features)
agg_features.extend(extra_features)
 
#I made more aggregate feature to select from in model\feature selection. 
#See this thread for some more aggregate ideas
#https://www.kaggle.com/c/santander-value-prediction-challenge/discussion/62446

def add_new_features(source, dest, feats):
    #dest['high_{}_{}'.format(feats[0], len(feats))] = source[feats].max(axis=1)
    #dest['mean_{}_{}'.format(feats[0], len(feats))] = source[feats].replace(0, np.nan).mean(axis=1)
    #dest['low_{}_{}'.format(feats[0], len(feats))] = source[feats].replace(0, np.nan).min(axis=1)
    #dest['median_{}_{}'.format(feats[0], len(feats))] = source[feats].replace(0, np.nan).median(axis=1)
    #dest['sum_{}_{}'.format(feats[0], len(feats))] = source[feats].sum(axis=1)
    #dest['stddev_{}_{}'.format(feats[0], len(feats))] = source[feats].std(axis=1)
    
    dest['mean_log_{}_{}'.format(feats[0], len(feats))] = np.log1p(source[feats].replace(0, np.nan).mean(axis=1))    
    dest['first_nonZero_{}_{}'.format(feats[0], len(feats))] = np.log1p(source[feats].replace(0, np.nan).bfill(1).iloc[:, 0])
    dest['last_nonZero_{}_{}'.format(feats[0], len(feats))] = np.log1p(source[feats[::-1]].replace(0, np.nan).bfill(1).iloc[:, 0])    
    
    #dest['nb_nans_{}_{}'.format(feats[0], len(feats))] =  source[feats].replace(0, np.nan).isnull().sum(axis=1)
    #dest['unique_{}_{}'.format(feats[0], len(feats))] = source[feats].nunique(axis=1)
#now that leak is done we should get back ugly data for feature engineering. This might not be necessary.
del test
gc.collect
test = pd.read_csv('../input/test.csv')
train_feats = pd.DataFrame()
test_feats =pd.DataFrame()

for i, ef in tqdm_notebook(list(enumerate(agg_features))):        
    add_new_features(train, train_feats, ef)
    add_new_features(test, test_feats, ef)
# I made a general model runner but in this kernel it is hard coded to just the LGB class. 
# I left some of the generalness in the kernel incase it is of interest

class MyModel():
    def __init__(self, X_tr, y_tr, X_val, y_val, X_test):
        self.X_tr = X_tr
        self.y_tr = y_tr
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.params = {}
    def predict_val(self):
        return self.model.predict(self.X_val)
    def predict_test(self):
        return self.model.predict(self.X_test)
    
class LgbBoostModel(MyModel):
    def train(self):          
        
        
        self.params = { 'objective': 'regression', 'metric': 'rmse', 'boosting': 'gbdt', 'seed':seed, 'is_training_metric': True
                  ,'max_bin': 350 #,'max_bin': 150
                  ,'learning_rate': .005
                  ,'max_depth': -1                  
                  ,'num_leaves': 48
                  ,'feature_fraction': 0.1
                  ,'reg_alpha': 0
                  ,'reg_lambda': 0.2
                  ,'min_child_weight': 10}
        
        self.model = lgb.train(self.params, lgb.Dataset(self.X_tr, label=self.y_tr), 30000, 
                            [lgb.Dataset(self.X_tr, label=self.y_tr), lgb.Dataset(self.X_val, label=self.y_val)], 
                               verbose_eval=200, early_stopping_rounds=200)


#Make training data from the original training plus the leak. This is key to getting a good score.
cols = train_feats.columns
train_feat_final = pd.concat([train_feats[cols], test_feats[cols][test_leaks.target != 0]], axis = 0)
train_feat_id = pd.concat([train['ID'], test['ID'][test_leaks.target != 0]], axis = 0)
test_feat_final = test_feats[cols]    
y = np.array(list(np.log1p(train.target.values)) + list(np.log1p(test_leaks['target'][test_leaks.target != 0])))

X = train_feat_final.values
X_test = test_feat_final.values

print(X.shape)
print(X_test.shape)
n_splits = 5
seed = 42

kf = KFold(n_splits=n_splits, random_state=seed, shuffle=True)

rmse_scores = {}
oof_preds = {}
sub_preds = {}
model_params = {}

model_types = ['lgb']

for model_type in model_types:
    rmse_scores[model_type] = list()
    oof_preds[model_type] = np.zeros((X.shape[0],))
    sub_preds[model_type] = np.zeros((X_test.shape[0],))

print('{} fold..'.format(n_splits))

for fold, (train_index, test_index) in tqdm_notebook(list(enumerate(list(kf.split(y))[:]))):

    # print("TRAIN:", train_index, "TEST:", test_index)
    X_tr, X_val = X[train_index], X[test_index]
    y_tr, y_val = y[train_index], y[test_index]

    for model_type in model_types:
        print ('\n*** ' + model_type)
        #model = get_model_class(model_type,  X_tr, y_tr, X_val, y_val, X_test)
        model = LgbBoostModel(X_tr, y_tr, X_val, y_val, X_test)

        model.train()

        oof_preds[model_type][test_index] = model.predict_val()
        sub_preds[model_type] += model.predict_test() / n_splits        
        rmse = mean_squared_error(y_val, model.predict_val())**0.5
        rmse_scores[model_type].append(rmse)

        model.params['cv'] = n_splits
        #model.params['fold_by_target'] = fold_by_target
        model.params['seed'] = seed            
        model_params[model_type] = model.params

        print('Fold %d: %s Mean Squared Error %f'%(fold, model_type, rmse))

def mean(values):
    return float(sum(values)) / max(len(values), 1)

def sum_of_square_deviation(values, mean):
    return float(1/len(values) * sum((x - mean)** 2 for x in values))    

def export_results(model_type):
    subm = pd.read_csv('../input/sample_submission.csv')
    subm['target'] = np.expm1(sub_preds[model_type])
    
    oof = pd.DataFrame(train_feat_id.copy())
    oof['target'] = np.expm1(y)
    oof['prediction'] = np.expm1(oof_preds[model_type])
    mean_rmse = mean(rmse_scores[model_type])
    standard_deviation_rmse = math.sqrt(sum_of_square_deviation(rmse_scores[model_type],mean_rmse))
    
    #key = '{}_{}_{}'.format(model_type, int(mean_rmse * 10000), int(standard_deviation_rmse * 10000))
    key = '{}'.format(model_type)
    print( '{} Mean Squared Error {}'.format(model_type ,mean_rmse))
    print( '{} Stdev Squared Error {}'.format(model_type, standard_deviation_rmse))
    
    file_name = 'subm_{}_ml_base.csv'.format(key)                                 
    subm.to_csv(file_name, index=False, float_format="%.8f")
    
    #file_name = 'subm_{}_with_leaks.csv'.format(key)    
    file_name = 'submission.csv'.format(key)    
    subm['target'][test_leaks.target != 0] = test_leaks['target'][test_leaks.target != 0]
    subm.to_csv(file_name, index=False, float_format="%.8f")
    
    
    file_name = 'subm_{}_oof.csv'.format(key)    
    oof.to_csv(file_name, index=False, float_format="%.8f")
    model_params[model_type]['cv_score'] = int(mean_rmse * 10000)
    model_params[model_type]['cv_stddev'] = int(standard_deviation_rmse * 10000)
    model_params[model_type]['train_row_count'] = X.shape[0]
    model_params[model_type]['train_feature_count'] = X.shape[1]
    model_params[model_type]['test_leak_count'] = (test_leaks.target != 0).sum()
    with open('subm_{}_params.txt'.format(key) , "w") as text_file:
        params = str(model_params[model_type])
        print(f"{params}", file=text_file)

for model_type in model_types:
    export_results(model_type)


