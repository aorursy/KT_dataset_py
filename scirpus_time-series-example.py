import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, roc_curve
def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

def target_encode(trn_series=None, 
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
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)
stype = 'previous_application'
num_rows = None
train = pd.read_csv('../input/application_train.csv', nrows= num_rows)
test = pd.read_csv('../input/application_test.csv', nrows= num_rows)
test.insert(1,'TARGET',-1)
trainsub = train[['SK_ID_CURR','TARGET']]
testsub = test[['SK_ID_CURR','TARGET']]
del train, test
gc.collect()
pr = pd.read_csv('../input/'+stype+'.csv', nrows = num_rows)
trainsub = trainsub.merge(pr,on='SK_ID_CURR',how='left')
testsub = testsub.merge(pr,on='SK_ID_CURR',how='left')
gc.collect()
floattypes = []
inttypes = []
stringtypes = []
for c in trainsub.columns[1:]:
    if(trainsub[c].dtype=='object'):
        trainsub[c] = trainsub[c].astype('str')
        testsub[c] = testsub[c].astype('str')
        stringtypes.append(c)
    elif(trainsub[c].dtype=='int64'):
        trainsub[c] = trainsub[c].astype('int32')
        testsub[c] = testsub[c].astype('int32')
        inttypes.append(c)
    else:
        trainsub[c] = trainsub[c].astype('float32')
        testsub[c] = testsub[c].astype('float32')
        floattypes.append(c)
stringtypes
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for col in stringtypes:
    print(col)
    trainsub['te_'+col] = 0.
    testsub['te_'+col] = 0.
    SMOOTHING = testsub[~testsub[col].isin(trainsub[col])].shape[0]/testsub.shape[0]
    for f, (vis_index, blind_index) in enumerate(kf.split(trainsub)):
        _, trainsub.loc[blind_index, 'te_'+col] = target_encode(trainsub.loc[vis_index, col], 
                                                            trainsub.loc[blind_index, col], 
                                                            target=trainsub.loc[vis_index,'TARGET'], 
                                                            min_samples_leaf=100,
                                                            smoothing=SMOOTHING,
                                                            noise_level=0.0)
        _, x = target_encode(trainsub.loc[vis_index, col], 
                                              testsub[col], 
                                              target=trainsub.loc[vis_index,'TARGET'], 
                                              min_samples_leaf=100,
                                              smoothing=SMOOTHING,
                                              noise_level=0.0)
        testsub['te_'+col] += (.2*x)
    trainsub.drop(col,inplace=True,axis=1)
    testsub.drop(col,inplace=True,axis=1)
alldata = trainsub.append(testsub)
del trainsub, testsub
gc.collect()
x = alldata['SK_ID_CURR'].value_counts().reset_index(drop=False)
x.columns = ['SK_ID_CURR','cnt']
x.head()
alldata['SK_ID_PREV'] = alldata['SK_ID_PREV'].fillna(-1)
alldata.sort_values(['SK_ID_CURR','SK_ID_PREV'],inplace=True,ascending=False)
alldata['cc'] = alldata[['SK_ID_CURR']].groupby(['SK_ID_CURR']).cumcount()
alldata = alldata.groupby(['SK_ID_CURR'])[alldata.columns].head(20)
feats = [f for f in alldata.columns if f not in ['SK_ID_CURR','SK_ID_PREV','TARGET','cc']]
alldata= pd.pivot_table(alldata,index=['SK_ID_CURR','TARGET'],values=feats,columns=['cc'])
alldata.columns = [x+"_"+str(y) for x,y in zip(alldata.columns.get_level_values(0),alldata.columns.get_level_values(1))]
alldata.reset_index(drop=False,inplace=True)
alldata['nans'] = alldata.isnull().sum(axis=1)
alldata = alldata.merge(x,on='SK_ID_CURR')
trainsub = alldata[alldata.TARGET!=-1].copy()
testsub = alldata[alldata.TARGET==-1].copy()
del alldata
gc.collect()
feats = [f for f in trainsub.columns if f not in ['SK_ID_CURR','SK_ID_PREV','TARGET','cc']]
feats
from lightgbm import LGBMClassifier
folds = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(trainsub.shape[0])
sub_preds = np.zeros(testsub.shape[0])
feats = [f for f in trainsub.columns if f not in ['SK_ID_CURR','TARGET']]

for n_fold, (trn_idx, val_idx) in enumerate(folds.split(trainsub)):
    trn_x, trn_y = trainsub[feats].iloc[trn_idx], trainsub.iloc[trn_idx]['TARGET']
   
    val_x, val_y = trainsub[feats].iloc[val_idx], trainsub.iloc[val_idx]['TARGET']
    
    clf = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.1,
        num_leaves=30,
        colsample_bytree=.8,
        subsample=.9,
        max_depth=7,
        reg_alpha=.1,
        reg_lambda=.1,
        min_split_gain=.01,
        min_child_weight=100,
        silent=-1,
        verbose=-1,
    )
    
    clf.fit(trn_x, trn_y, 
            eval_set= [(trn_x, trn_y), (val_x, val_y)], 
            eval_metric='auc', verbose=100, early_stopping_rounds=100  #30
           )
    
    oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
    sub_preds += clf.predict_proba(testsub[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits
    
    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
    del clf, trn_x, trn_y, val_x, val_y
    gc.collect()
Submission = pd.DataFrame({ 'SK_ID_CURR': testsub.SK_ID_CURR.values,'TARGET': sub_preds })
Submission.to_csv('ts.csv', index=False)