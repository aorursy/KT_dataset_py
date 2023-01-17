import numpy as np 
import pandas as pd 
df_raw = pd.read_csv('../input/train.csv')
df =  df_raw.copy()
kaggle_sub = pd.read_csv('../input/test.csv')
df['SalePrice'] = np.log(df['SalePrice'])
df.shape
df = df.append(kaggle_sub)
df.shape
df.fillna(-1, inplace=True)
cat_vars = [col for col in df.columns if df[col].dtype == 'object']
for col in cat_vars:
    df[col] = df[col].astype('category').cat.codes
df, kaggle_sub = df.iloc[:1460], df.iloc[1460:]
removed_cols = ['SalePrice']
feats = [c for c in df.columns if c not in removed_cols]
from sklearn.model_selection import train_test_split
train, valid = train_test_split(df, random_state=42)
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=42)
rf.fit(train[feats], train['SalePrice'])
train_preds = rf.predict(train[feats])
valid_preds = rf.predict(valid[feats])
from sklearn.metrics import mean_squared_error
mean_squared_error(train['SalePrice'], train_preds) ** (1/2)
mean_squared_error(valid['SalePrice'], valid_preds) ** (1/2)
rf = RandomForestRegressor(random_state=42)
rf.fit(df[feats], df['SalePrice'])
kaggle_sub['SalePrice'] = np.exp(rf.predict(kaggle_sub[feats]))
kaggle_sub[['Id', 'SalePrice']].to_csv('submission.csv', index=False)
rf = RandomForestRegressor(random_state=42, n_estimators=200, n_jobs=-1)
rf.fit(train[feats], train['SalePrice'])
train_preds_rf = rf.predict(train[feats])
valid_preds_rf = rf.predict(valid[feats])
from sklearn.metrics import mean_squared_error
mean_squared_error(train['SalePrice'],train_preds_rf) ** (1/2)
mean_squared_error(valid['SalePrice'], valid_preds_rf) ** (1/2)
rf = RandomForestRegressor(random_state=42, n_estimators=200, max_features=.9, n_jobs=-1)
rf.fit(df[feats], df['SalePrice'])
kaggle_sub['SalePrice'] = np.exp(rf.predict(kaggle_sub[feats]))
kaggle_sub[['Id', 'SalePrice']].to_csv('submission_rf_opt.csv', index=False)
fi = pd.Series(rf.feature_importances_, index=feats)
fi.sort_values(ascending=False)
fi.sort_values().plot.barh(figsize=(20,20))
fi[fi > 0.002].shape
feats = fi[fi > 0.002].index.tolist() 
rf = RandomForestRegressor(random_state=42, n_estimators=200, max_features=.9, n_jobs=-1)
rf.fit(train[feats], train['SalePrice'])
train_preds_rf = rf.predict(train[feats])
valid_preds_rf = rf.predict(valid[feats])
from sklearn.metrics import mean_squared_error
mean_squared_error(train['SalePrice'],train_preds_rf) ** (1/2)
mean_squared_error(valid['SalePrice'], valid_preds_rf) ** (1/2)
fi = pd.Series(rf.feature_importances_, index=feats)
fi.sort_values().plot.barh(figsize=(20,10))
object_cols = [c for c in fi.index if df_raw[c].dtype == 'object' and df_raw[c].nunique() < 8]
df[object_cols].nunique()
def one_hot_encode(df, df_raw, object_cols, max_cats=None):
    df.reset_index(inplace=True)
    for col in object_cols:
        df_raw[col].astype('category')
        if max_cats and df_raw[col].nunique() > max_cats:
            df_raw[col].cat.codes = df_raw[col].cat.codes.apply(lambda x : x if x <= max_cats else max_cats + 1)   
        df = pd.concat([df, pd.get_dummies(df_raw[col], prefix=col, dummy_na=True)], axis=1)        
        del df[col]        
    return df
df = one_hot_encode(df, df_raw, object_cols)
removed_cols = ['SalePrice']
feats = [c for c in df.columns if c not in removed_cols]
train, valid = train_test_split(df, random_state=42)
rf = RandomForestRegressor(random_state=11, n_estimators=200,  n_jobs=-1)
rf.fit(train[feats], train['SalePrice'])
train_preds_rf = rf.predict(train[feats])
valid_preds_rf = rf.predict(valid[feats])
mean_squared_error(train['SalePrice'], train_preds_rf) ** (1/2)
mean_squared_error(valid['SalePrice'], valid_preds_rf) ** (1/2)
from sklearn.metrics import r2_score
r2_score(valid['SalePrice'], valid_preds_rf) 
fi = pd.Series(rf.feature_importances_, index=feats)
fi.sort_values().plot.barh(figsize=(20,30))
fi[fi > 0.005].shape
feats = fi[fi > 0.005].index.tolist() 
feats
rf = RandomForestRegressor(random_state=42, n_estimators=200,  n_jobs=-1)
rf.fit(train[feats], train['SalePrice'])
train_preds_rf = rf.predict(train[feats])
valid_preds_rf = rf.predict(valid[feats])
mean_squared_error(train['SalePrice'],train_preds_rf) ** (1/2)
mean_squared_error(valid['SalePrice'], valid_preds_rf) ** (1/2)
r2_score(valid['SalePrice'], valid_preds_rf) 
pd.Series(rf.feature_importances_, index=feats).sort_values().plot.barh(figsize=(20,10))
preds = np.stack([t.predict(df[feats]) for t in rf.estimators_])
preds.shape
x = df[['OverallQual','SalePrice']].copy()
x['pred_std'] = np.std(preds, axis=0)
x['pred_mean'] = np.mean(preds, axis=0)
x.groupby('OverallQual').mean()
x.groupby('OverallQual', as_index=False).mean().plot('OverallQual', 'pred_mean', 'barh', xerr='pred_std', figsize=(20,10))
def cv(df, k, feats):
    preds = []
    score = []
    fis = []
    chunk = df.shape[0] // k
    for i in range(k):
        if i + 1 < k:
            valid = df.iloc[i*chunk: (i+1)*chunk]
            train = df.iloc[: i*chunk].append(df.iloc[(i+1)*chunk:])
            
        else:
            valid = df.iloc[i*chunk:]
            train = df.iloc[: i*chunk] 
        
        print(train.shape, valid.shape)

        rf = RandomForestRegressor(random_state=42, n_estimators=100, min_samples_leaf=2)
        rf.fit(train[feats], train['SalePrice'])
        score.append(mean_squared_error(valid['SalePrice'], rf.predict(valid[feats]))**(1/2))        
        fis.append(rf.feature_importances_)
    return pd.Series(score), pd.Series(preds).mean(), fis
score, preds, fis = cv(df, 5, feats)
score.mean()
