import numpy as np
import pandas as pd
import time

import lightgbm as lgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
my_aggs = {
           'passband': ['mean', 'std', 'var'],
           'flux': ['min', 'max', 'mean', 'median', 'std'],
           'flux_err': ['min', 'max', 'mean', 'median', 'std'],
           'detected': ['mean'],
           'flux_ratio_sq': ['sum'],
           'flux_by_flux_ratio_sq':['sum']
          }
def transform_ts(data_ts, aggs):
    #copy
    df_ts = data_ts.copy()
    
    #add
    df_ts = df_ts.assign(flux_ratio_sq = np.power(df_ts['flux'] / df_ts['flux_err'], 2.0))
    df_ts = df_ts.assign(flux_by_flux_ratio_sq = df_ts['flux'] * df_ts['flux_ratio_sq'])

    #aggregate
    df_ts_agg = df_ts.groupby(['object_id']).agg(aggs)
    df_ts_agg.columns = ['_'.join((col[0],col[1])) for col in df_ts_agg.columns]
    df_ts_agg = df_ts_agg.reset_index()
    
    return df_ts_agg
    
def transform_ts_chunk(from_file, to_file, aggs, chunk_size=5000000):
    
    remain_df=None
    start = time.time()

    for i, df in enumerate(pd.read_csv(from_file, chunksize=chunk_size, iterator=True)):
        
        # set aside data of last object_id (may be implete)
        remain_id = df.iloc[-1]['object_id']
        new_remain_df = df[df['object_id'] == remain_id].copy()
        df = df[~(df['object_id'] == remain_id)].copy()
        
        # add remain_df of last iteration if exists. 
        if remain_df is not None:
            df = pd.concat([remain_df, df], axis=0)
        remain_df = new_remain_df
        
        # apply transformations and save
        df_ts_agg = transform_ts(df, aggs)
        if i == 0:
            df_ts_agg.to_csv(to_file, header=True, index=False)
        else:
            df_ts_agg.to_csv(to_file, header=False, index=False, mode='a')
    
        # print progress
        print("chunk", i, "done in", int(time.time() - start), "sec")
        start = time.time()
    
    # add remaining object
    df_ts_agg = transform_ts(remain_df, aggs)
    df_ts_agg.to_csv(to_file, header=False, index=False, mode='a')
    
train_ts = pd.read_csv("../input/training_set.csv")
train_ts_agg = transform_ts(train_ts, my_aggs)
train_ts_agg.to_csv("training_set_agg.csv", header=True, index=False)
train_meta = pd.read_csv("../input/training_set_metadata.csv")
train_total  = train_meta.merge(train_ts_agg, on=['object_id'], how='left')
train_total.to_csv("training_total.csv", header=True, index=False)
transform_ts_chunk("../input/test_set.csv", "test_set_agg.csv", my_aggs)
test_meta = pd.read_csv("../input/test_set_metadata.csv")
test_ts_agg = pd.read_csv("test_set_agg.csv")
test_total = test_meta.merge(test_ts_agg, on=['object_id'], how='left')
test_total.to_csv("test_total.csv", header=True, index=False)
del test_ts_agg, test_meta, test_total
def lgb_multi_weighted_logloss(y_true, y_preds, scorer=False):
    """@author olivier https://www.kaggle.com/ogrellier"""
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
    if len(np.unique(y_true)) > 14:
        classes.append(99)
        class_weight[99] = 2
    y_p = y_preds.reshape(y_true.shape[0], len(classes), order='F')

    y_ohe = pd.get_dummies(y_true)
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
    y_p_log = np.log(y_p)
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos

    loss = - np.sum(y_w) / np.sum(class_arr)
    return 'wloss', loss, False
class imputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.means = dict()

    def fit(self, X, y=None):
        print('Fitting the imputer...')
                
        numeric_columns = list(X.select_dtypes(include=[np.number]).columns)
        for col in numeric_columns:
            try:
                self.means[col] = int(X.loc[:,col].mean())
            except:
                self.means[col] = 0
        
        return self

    def transform(self, X):
        print('Transforming the data with the imputer...')
        X_new = X.copy()
        non_numeric_columns = list(X_new.select_dtypes(exclude=[np.number]).columns)
        for col in non_numeric_columns:
            na_value = "unknown"
            X_new[col].fillna(na_value, inplace=True)
        
        numeric_columns = list(X_new.select_dtypes(include=[np.number]).columns)
        for col in numeric_columns:
            na_value = self.means[col]
            X_new[col].fillna(na_value, inplace=True)
        return X_new

class addFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self 

    def transform(self, X):
        print('Adding features...')
        X_new = X.copy()
        
        X_new['flux_diff'] = X_new['flux_max'] - X_new['flux_min']
        X_new['flux_rel_diff'] = (X_new['flux_max'] - X_new['flux_min']) / X_new['flux_mean']
        X_new['flux_w_mean'] = X_new['flux_by_flux_ratio_sq_sum'] / X_new['flux_ratio_sq_sum']
        X_new['flux_rel_diff2'] = (X_new['flux_max'] - X_new['flux_min']) / X_new['flux_w_mean']
        return X_new

class dropFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self 

    def transform(self, X):
        print('Dropping features...')
        X_new = X.copy()
        for col in self.cols:
            X_new = X_new.drop(col, axis=1)
        return X_new
    
class customModel(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.clfs = []
        self.lgb_params = {
            'boosting_type': 'gbdt',
            'objective': 'multiclass',
            'num_class': 14,
            'metric': 'multi_logloss',
            'learning_rate': 0.03,
            'subsample': .9,
            'colsample_bytree': .7,
            'reg_alpha': .01,
            'reg_lambda': .01,
            'min_split_gain': 0.01,
            'min_child_weight': 10,
            'n_estimators': 1000,
            'silent': -1,
            'verbose': -1,
            'max_depth': 3
        }
        
    def fit(self, X, y):
        print('Fitten the model...')
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        
        w = y.value_counts()
        weights = {i: np.sum(w) / w[i] for i in w.index}
        
        for fold_, (trn_, val_) in enumerate(folds.split(y, y)):
            trn_x, trn_y = X.iloc[trn_], y.iloc[trn_]
            val_x, val_y = X.iloc[val_], y.iloc[val_]

            clf = lgb.LGBMClassifier(**self.lgb_params)
            clf.fit(
                trn_x, trn_y,
                eval_set=[(trn_x, trn_y), (val_x, val_y)],
                eval_metric=lgb_multi_weighted_logloss,
                verbose=100,
                early_stopping_rounds=50,
                sample_weight=trn_y.map(weights)
            )
            self.clfs.append(clf)
        return self
        
    def predict(self, X):
        print('Predicting...')
        preds_ = None
        for clf in self.clfs:
            if preds_ is None:
                preds_ = clf.predict_proba(X) / len(self.clfs)
            else:
                preds_ += clf.predict_proba(X) / len(self.clfs)
        return preds_
        
df_train = pd.read_csv("training_total.csv")
X_train = df_train.drop(['target'],1)
y_train = df_train['target']

estimator_lgb = Pipeline(steps = [
        ('imputer', imputer()),
        ('add_features', addFeatures()),
        ('drop_features', dropFeatures(['object_id', 'hostgal_specz'])),
        ('customModel', customModel())])

estimator_lgb.fit(X_train, y_train)
X_test = pd.read_csv("test_total.csv")
splits = np.array_split(X_test['object_id'].unique(), 50)

normalize_rows_to_one = True
 
for i, split in enumerate(splits):
    print("split:", i)
    chunk = X_test[X_test['object_id'].isin(split)]
    chunk_pred = estimator_lgb.predict(chunk)

    # prob of class 99 is probability of not other clases
    preds_99 = np.ones(chunk_pred.shape[0])
    for j in range(chunk_pred.shape[1]):
        preds_99 *= (1 - chunk_pred[:, j])
    preds_99 = np.expand_dims(preds_99,1)
    preds_99 = 0.14 * preds_99 / np.mean(preds_99)
    chunk_pred = np.append(chunk_pred, preds_99, axis=1)

    # rescale such that all probs in one row add up to 1.
    if(normalize_rows_to_one):
        row_sums = np.expand_dims(np.sum(chunk_pred, axis = 1),1)
        chunk_pred = chunk_pred/row_sums

    # alles lekker aan elkaar plakken
    if i==0:
        y_test_pred = chunk_pred
    else:
        y_test_pred = np.append(y_test_pred, chunk_pred, axis=0)

#y_test_pred = y_test_pred.astype('float16')
targets = estimator_lgb.named_steps.customModel.clfs[0].classes_
targets = np.append(targets,'99')
y_test_pred_df = pd.DataFrame(index=X_test['object_id'], data=y_test_pred, columns=['class_'+i for i in targets])
y_test_pred_df.reset_index(inplace = True)
y_test_pred_df.to_csv("submission.csv", index = False)

