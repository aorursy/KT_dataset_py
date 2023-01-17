import numpy as np 

import pandas as pd

from xgboost import XGBClassifier

from sklearn.model_selection import KFold,StratifiedKFold

from numba import jit
@jit

def eval_gini(y_true, y_prob):

    y_true = np.asarray(y_true)

    y_true = y_true[np.argsort(y_prob)]

    ntrue = 0

    gini = 0

    delta = 0

    n = len(y_true)

    for i in range(n-1, -1, -1):

        y_i = y_true[i]

        ntrue += y_i

        gini += y_i * delta

        delta += 1 - y_i

    gini = 1 - 2 * gini / (ntrue * (n - ntrue))

    return gini



def gini_xgb(preds, dtrain):

    labels = dtrain.get_label()

    preds -= preds.min()

    preds / preds.max()

    gini_score = -eval_gini(labels, preds)

    return [('gini', gini_score)]





def add_noise(series, noise_level):

    return series * (1 + noise_level * np.random.randn(len(series)))





def target_encode(trn_series=None,    # Revised to encode validation series

                  val_series=None,

                  tst_series=None,

                  target=None,

                  min_samples_leaf=1,

                  smoothing=1,

                  noise_level=0):

    """

    Smoothing is computed like in the following paper by Daniele Micci-Barreca

    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf

    trn_series : training categorical feature as a pd.Series

    tst_series : test categorical feature as a pd.Series

    target : target data as a pd.Series

    min_samples_leaf (int) : minimum samples to take category average into account

    smoothing (int) : smoothing effect to balance categorical average vs prior

    """

    assert len(trn_series) == len(target)

    assert trn_series.name == tst_series.name

    temp = pd.concat([trn_series, target], axis=1)

    # Compute target mean

    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])

    # Compute smoothing

    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))

    # Apply average function to all target data

    prior = target.mean()

    # The bigger the count the less full_avg is taken into account

    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing

    averages.drop(["mean", "count"], axis=1, inplace=True)

    # Apply averages to trn and tst series

    ft_trn_series = pd.merge(

        trn_series.to_frame(trn_series.name),

        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),

        on=trn_series.name,

        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)

    # pd.merge does not keep the index so restore it

    ft_trn_series.index = trn_series.index

    ft_val_series = pd.merge(

        val_series.to_frame(val_series.name),

        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),

        on=val_series.name,

        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)

    # pd.merge does not keep the index so restore it

    ft_val_series.index = val_series.index

    ft_tst_series = pd.merge(

        tst_series.to_frame(tst_series.name),

        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),

        on=tst_series.name,

        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)

    # pd.merge does not keep the index so restore it

    ft_tst_series.index = tst_series.index

    return add_noise(ft_trn_series, noise_level), add_noise(ft_val_series, noise_level), add_noise(ft_tst_series, noise_level)
strdirectory = '../input/'

train = pd.read_csv(strdirectory+'train.csv')

test = pd.read_csv(strdirectory+'test.csv')





test.insert(1,'target',0)

print(train.shape)

print(test.shape)



x = pd.concat([train,test])

x = x.reset_index(drop=True)

unwanted = x.columns[x.columns.str.startswith('ps_calc_')]

x.drop(unwanted,inplace=True,axis=1)



x.loc[:,'ps_reg_03'] = pd.cut(x['ps_reg_03'], 50,labels=False)

x.loc[:,'ps_car_12'] = pd.cut(x['ps_car_12'], 50,labels=False)

x.loc[:,'ps_car_13'] = pd.cut(x['ps_car_13'], 50,labels=False)

x.loc[:,'ps_car_14'] =  pd.cut(x['ps_car_14'], 50,labels=False)

x.loc[:,'ps_car_15'] =  pd.cut(x['ps_car_15'], 50,labels=False)



test = x.iloc[train.shape[0]:].copy()

train = x.iloc[:train.shape[0]].copy()

features = train.columns[2:]

ranktestpreds = np.zeros(test.shape[0])

kf = KFold(n_splits=5,shuffle=True,random_state=2017)

for i, (train_index, test_index) in enumerate(kf.split(list(train.index))):

    print('Fold: ',i)

    myfeatures = list(features[:])

    blindtrain = train.iloc[test_index].copy()

    vistrain = train.iloc[train_index].copy()

    mytest = test.copy()

    for column in features:

        vis, blind, tst = target_encode(trn_series=vistrain[column],

                                        val_series=blindtrain[column],

                                        tst_series=mytest[column],

                                        target=vistrain.target,

                                        min_samples_leaf=200,

                                        smoothing=10,

                                        noise_level=0)

        vistrain['te_' + column] = vis

        blindtrain['te_' + column] = blind

        mytest['te_' + column] = tst

        myfeatures = myfeatures + list(['te_' + column])

        

    clf = XGBClassifier(n_estimators=2000,

                        objective="rank:pairwise",

                        learning_rate = 0.04,

                        max_depth = 5,

                        min_child_weight = 9,

                        subsample = 0.8,

                        colsample_bytree = 0.8,

                        reg_alpha = 10.4,

                        reg_lambda = 0.59,

                        seed = 2017,

                        nthread = 8,

                        silent = 1)

    

    eval_set=[(blindtrain[myfeatures],blindtrain.target)]

    model = clf.fit(vistrain[myfeatures], vistrain.target, 

                    eval_set=eval_set,

                    eval_metric=gini_xgb,

                    early_stopping_rounds=70,

                    verbose=False)

    

    print( "  Best N trees = ", model.best_ntree_limit )

    print( "  Best gini = ", model.best_score )

    trainpreds = model.predict_proba(blindtrain[myfeatures],ntree_limit=model.best_ntree_limit)[:,1]

    print( "  Best gini = ", eval_gini(blindtrain.target,trainpreds))

    ranktestpreds += model.predict_proba(mytest[myfeatures],ntree_limit=model.best_ntree_limit)[:,1]

ranktestpreds /= 5

ranktestpreds -= ranktestpreds.min()

ranktestpreds /= ranktestpreds.max()

xgbtestpreds = np.zeros(test.shape[0])

kf = KFold(n_splits=5,shuffle=True,random_state=2017)

for i, (train_index, test_index) in enumerate(kf.split(list(train.index))):

    print('Fold: ',i)

    myfeatures = list(features[:])

    blindtrain = train.iloc[test_index].copy()

    vistrain = train.iloc[train_index].copy()

    mytest = test.copy()

    for column in features:

        vis, blind, tst = target_encode(trn_series=vistrain[column],

                                        val_series=blindtrain[column],

                                        tst_series=mytest[column],

                                        target=vistrain.target,

                                        min_samples_leaf=200,

                                        smoothing=10,

                                        noise_level=0)

        vistrain['te_' + column] = vis

        blindtrain['te_' + column] = blind

        mytest['te_' + column] = tst

        myfeatures = myfeatures + list(['te_' + column])

        

    clf = XGBClassifier(n_estimators=2000,

                        learning_rate = 0.04,

                        max_depth = 5,

                        min_child_weight = 9,

                        subsample = 0.8,

                        colsample_bytree = 0.8,

                        reg_alpha = 10.4,

                        reg_lambda = 0.59,

                        seed = 2017,

                        nthread = 8,

                        silent = 1)

    

    eval_set=[(blindtrain[myfeatures],blindtrain.target)]

    model = clf.fit(vistrain[myfeatures], vistrain.target, 

                    eval_set=eval_set,

                    eval_metric=gini_xgb,

                    early_stopping_rounds=70,

                    verbose=False)

    

    print( "  Best N trees = ", model.best_ntree_limit )

    print( "  Best gini = ", model.best_score )

    trainpreds = model.predict_proba(blindtrain[myfeatures],ntree_limit=model.best_ntree_limit)[:,1]

    print( "  Best gini = ", eval_gini(blindtrain.target,trainpreds))

    xgbtestpreds += model.predict_proba(mytest[myfeatures],ntree_limit=model.best_ntree_limit)[:,1]

xgbtestpreds /= 5

xgbtestpreds -= xgbtestpreds.min()

xgbtestpreds /= xgbtestpreds.max()
rankdata = pd.DataFrame()

rankdata['xgbnormal'] = xgbtestpreds

rankdata['xgbrank'] = ranktestpreds
sub = pd.read_csv('../input/sample_submission.csv')

sub.target = (rankdata.xgbnormal.rank()+rankdata.xgbrank.rank())

sub.target -= sub.target.min()

sub.target /= sub.target.max()

sub.to_csv('xgbsubmission.csv', index = False)