import pandas as pd
from sklearn.model_selection import KFold
from sklearn.svm import SVR
import tqdm 
import numpy as np
from sklearn.linear_model import Ridge
import dask
fnc = pd.read_csv('../input/trends-assessment-prediction/fnc.csv')
loading = pd.read_csv('../input/trends-assessment-prediction/loading.csv')
labels = pd.read_csv('../input/trends-assessment-prediction/train_scores.csv')
fnc_features, loading_features = list(fnc.columns[1:]), list(loading.columns[1:])
df = fnc.merge(loading, on="Id")

labels["is_train"] = True
df = df.merge(labels, on="Id", how="left")

test_df = df[df["is_train"] != True].copy()
df = df[df["is_train"] == True].copy()

df.shape, test_df.shape
FNC_SCALE = 1/600

df[fnc_features] *= FNC_SCALE
test_df[fnc_features] *= FNC_SCALE
def metric(y_true, y_pred):
    return np.mean(np.sum(np.abs(y_true - y_pred), axis=0)/np.sum(y_true, axis=0))
def svm_ridge(target, c, ww, ff):

    NUM_FOLDS = 7
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=0)


    features = loading_features + fnc_features

    overal_score = 0
    y_oof = np.zeros(df.shape[0])
    y_test = np.zeros((test_df.shape[0], NUM_FOLDS))
    
    cnt = 0
    for f, (train_ind, val_ind) in (enumerate(kf.split(df, df))):
        train_df, val_df = df.iloc[train_ind], df.iloc[val_ind]
        train_df = train_df[train_df[target].notnull()]

        # train both models at once
        pipelines = [SVR(C=c), Ridge(alpha=0.0001)]  
        pipelines_ = [dask.delayed(pl).fit(train_df[features], train_df[target]) for pl in pipelines]
        fit_pipelines = dask.compute(*pipelines_)
        
        # inference for val
        inference_ = [dask.delayed(m).predict(val_df[features]) for m in fit_pipelines]
        preds = (dask.compute(*inference_))
        val_pred_1 = preds[0]
        val_pred_2 = preds[1]
        
        # inference for test
        inference_ = [dask.delayed(m).predict(test_df[features]) for m in fit_pipelines]
        preds = (dask.compute(*inference_))
        test_pred_1 = preds[0]
        test_pred_2 = preds[1]
        
        val_pred = np.array((1-ff)*val_pred_1+ff*val_pred_2)
        val_pred = val_pred.flatten()
        
        test_pred = np.array((1-ff)*test_pred_1+ff*test_pred_2)
        test_pred = test_pred.flatten()
        
        y_oof[val_ind] = val_pred
        y_test[:, f] = test_pred
        
        
        print(target + " iter " + str(cnt))
        cnt+=1
        
    df["pred_{}".format(target)] = y_oof
    test_df[target] = y_test.mean(axis=1)
    
    score = metric(df[df[target].notnull()][target].values, df[df[target].notnull()]["pred_{}".format(target)].values)
    print(target + " " + str(score))
    return ww*score, test_df[target]
%%time 
results = []
L = [("age", 60, 0.3, 0.5), ("domain1_var1", 12, 0.175, 0.2), ("domain1_var2", 8, 0.175, 0.2), ("domain2_var1", 9, 0.175, 0.22), ("domain2_var2", 15, 0.175, 0.22)]
for x in L:
    print(x)
    y = dask.delayed(svm_ridge)(x[0], x[1], x[2], x[3])
    results.append(y)

results = dask.compute(*results)
r_sum = 0
for i in range(0, 5):
    r_sum += results[i][0]
r_sum

preds = pd.concat([results[0][1], results[1][1], results[2][1], results[3][1], results[4][1]], axis=1)
# Function from https://www.kaggle.com/nischaydnk/beginners-trends-neuroimaging-decent-score
def make_sub(predictions):
    features = ('age', 'domain1_var1', 'domain1_var2','domain2_var1','domain2_var2')
    _columns = (0,1,2,3,4)
    tests = predictions.rename(columns=dict(zip(features, _columns)))
    tests = tests.melt(id_vars='Id',value_vars=_columns,value_name='Predicted')
    tests['target'] = tests.variable.map(dict(zip(_columns, features)))
    tests['Id_'] = tests[['Id', 'target']].apply(lambda x: '_'.join((str(x[0]), str(x[1]))), axis=1)
  
    return tests.sort_values(by=['Id', 'variable'])\
              .drop(['Id', 'variable', 'target'],axis=1)\
              .rename(columns={'Id_':'Id'})\
              .reset_index(drop=True)\
              [['Id', 'Predicted']]
preds = pd.concat([test_df['Id'], preds], axis=1)
preds = make_sub(preds)
preds.to_csv('dask.csv', index=False)
