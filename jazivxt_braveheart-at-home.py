from ml import simple #add custom Package the1owl
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import linear_model, metrics, decomposition

train = pd.read_csv('../input/train.csv')
train['SalePrice'] = np.log1p(train['SalePrice'])
test = pd.read_csv('../input/test.csv')
sub = pd.read_csv('../input/sample_submission.csv')
print(train.shape, test.shape)
col = [c for c in train.columns if c not in ['Id', 'SalePrice']]

train['null_count'] = np.sum((train[col]==np.nan).values, axis=1)
test['null_count'] = np.sum((test[col]==np.nan).values, axis=1)
for c in col:
    train[c + '_null'] = (train[c]==np.nan).astype(int)
    test[c + '_null'] = (test[c]==np.nan).astype(int)
    if train[c].dtype == 'O':
        lbl0 = {k: v for k, v in train.groupby(by=[c], as_index=False)['SalePrice'].mean().values} 
        lbl1 = {k: v for k, v in train.groupby(by=[c], as_index=False)['SalePrice'].max().values} 
        lbl2 = {k: v for k, v in train.groupby(by=[c], as_index=False)['SalePrice'].min().values} 
        lbl3 = {k:v for v, k in enumerate(train[c].value_counts().index)}
        lbl4 = {k: np.log1p(v) for k, v in train[c].value_counts().reset_index().values}
        lbl5 = {k:v for v, k in enumerate(pd.concat((train[c], test[c])).value_counts().index)}
        lbl6 = {k: np.log1p(v) for k, v in pd.concat((train[c], test[c])).value_counts().reset_index().values}
        if len(lbl1)>100:
            for k in lbl1:
                train[c+'_'+str(k)] = (train[c] == k).astype(int)
                test[c+'_'+str(k)] = (test[c] == k).astype(int)
        for i in range(6):
            train[c+'_key_'+str(i)] = train[c].map(eval('lbl'+str(i)))
            test[c+'_key_'+str(i)] = test[c].map(eval('lbl'+str(i)))
        train[c] = train[c].map(lbl6)
        test[c] = test[c].map(lbl6)
    else:
        slope, intercept, r_value, p_value, std_err = stats.linregress(train[c],train['SalePrice'])
        train[c+'_slope'] = train[c] * slope
        test[c+'_slope'] = test[c] * slope
        
        train[c+'_log1p'] = np.log1p(train[c])
        test[c+'_log1p'] = np.log1p(test[c])

train.fillna(-1, inplace=True)
test.fillna(-1, inplace=True)

feature_cnt = 25
col = [c for c in train.columns if c not in ['Id', 'SalePrice']]
dim_reduction_pca = decomposition.PCA(n_components=feature_cnt, random_state=10)
dim_reduction_svd = decomposition.TruncatedSVD(n_components=feature_cnt, random_state=11)
dim_reduction_pca_train = dim_reduction_pca.fit_transform(train[col])
dim_reduction_svd_train = dim_reduction_svd.fit_transform(train[col])
dim_reduction_pca_test = dim_reduction_pca.transform(test[col])
dim_reduction_svd_test = dim_reduction_svd.transform(test[col])
for i in range(feature_cnt):
    train['dim_reduction_pca_'+str(i)] = dim_reduction_pca_train[:,i]
    test['dim_reduction_pca_'+str(i)] = dim_reduction_pca_test[:,i]
    train['dim_reduction_svd_'+str(i)] = dim_reduction_svd_train[:,i]
    test['dim_reduction_svd_'+str(i)] = dim_reduction_svd_test[:,i]

print(train.shape, test.shape)
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split

col = [c for c in train.columns if c not in ['Id', 'SalePrice']]
col = [c for c in col if train[c].dtype != 'O']

x1, x2, y1, y2 = train_test_split(train[col].fillna(-1), train['SalePrice'], test_size=0.2, random_state=3)
etr = ExtraTreesRegressor(n_jobs=-1, random_state=3)
etr.fit(x1, y1)
print(np.sqrt(metrics.mean_squared_error(y2, etr.predict(x2))))
feature_importances = pd.DataFrame({'features':col, 'importance': etr.feature_importances_}).sort_values(by=['importance'], ascending=False)
feature_importances.head()
#Testing Outlier Exclusions - need similar approach for categorical and dates
d_imp = {k:v for k, v in feature_importances.values}

def f_hl_test(a): #test more options with the target values
    q1 = a.quantile(0.25)
    q3 = a.quantile(0.75)
    iqr = q3 - q1
    iqr_h = q3 + iqr*1.5
    iqr_l = q1 - iqr*1.5
    #a.min(), sum(a < iqr_l), iqr_l, q1, q3, iqr_h, sum(a > iqr_h), a.max()
    return iqr_l, iqr_h

train['outlier'] = 0.0
test['outlier'] = 0.0
for c in feature_importances.features:
    l, h = f_hl_test(train[c])
    #print(c, len(train[(train[c] < l) | (train[c] > h)]), l, h)
    #train = train[(train[c] >= l) & (train[c] <= h)].reset_index(drop=True)
    train['outlier'] += (((train[c] < l).astype(int) + (train[c] > h).astype(int)) > 0).astype(int) * d_imp[c]
    test['outlier'] += (((test[c] < l).astype(int) + (test[c] > h).astype(int)) > 0).astype(int) * d_imp[c]

#train = train.sort_values(by=['outlier'], ascending=False).reset_index(drop=True)
#ocut = -30
#outliers = train[ocut:].copy().reset_index(drop=True)
#train = train[:ocut].copy().reset_index(drop=True)
#train.drop(columns=['outlier'], inplace=True)
print(train.shape, test.shape)
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

sk_if = IsolationForest(n_jobs=-1, random_state=3)
sk_lof = LocalOutlierFactor(novelty=True, n_jobs=-1)
train['outlier_sk_if'] = sk_if.fit_predict(train[col].fillna(-1), train['SalePrice'])
sk_lof.fit(train[col].fillna(-1), train['SalePrice'])
train['outlier_sk_lof'] = sk_lof.predict(train[col].fillna(-1))
test['outlier_sk_if'] = sk_if.predict(test[col].fillna(-1))
test['outlier_sk_lof'] = sk_lof.predict(test[col].fillna(-1))
col = [c for c in train.columns if c not in ['Id', 'SalePrice']]

#using ml.simple shortcuts here
data = simple.Data(train, test, 'Id', 'SalePrice')
params = {'learning_rate': 0.005, 'max_depth': -1, 'boosting': 'gbdt', 'objective': 'regression', 'metric':'rmse', 'seed': 3, 'num_iterations': 5000, 'early_stopping_round': 200, 'verbose_eval': 300, 'num_leaves': 64, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5}
sub = simple.Model(data, 'LGB', params, 0.2, 7).PRED
sub['SalePrice'] = np.expm1(sub['SalePrice'])
sub.to_csv('submission_lgb.csv', index=False)
lr = linear_model.Ridge(random_state=3)
lr.fit(train[col], train['SalePrice'])
print('Ridge', np.sqrt(metrics.mean_squared_error(train['SalePrice'], lr.predict(train[col]))))
#print('Ridge on Outliers', np.sqrt(metrics.mean_squared_error(outliers['SalePrice'], lr.predict(outliers[col]))))
test['SalePrice'] = np.expm1(lr.predict(test[col]))
test[['Id','SalePrice']].to_csv('submission_ridge.csv', index=False)

sub = simple.Blend(['submission_lgb.csv', 'submission_ridge.csv'], 'Id', 'SalePrice').BLEND.reset_index()
xtrain = train.copy()
for i in range(3):
    test.drop(columns=['SalePrice'], inplace=True)
    sub['SalePrice'] = np.log1p(sub['SalePrice'])
    test = pd.merge(test, sub[['Id','SalePrice']], how='left', on='Id')
    train = pd.concat((xtrain, test), ignore_index=True, sort=False)

    for c in feature_importances.features[:5]:
        l, h = f_hl_test(train[c])
        print(c, len(train[(train[c] < l) | (train[c] > h)]))
        train = train[(train[c] >= l) & (train[c] <= h)].reset_index(drop=True)
    
    data = simple.Data(train, test, 'Id', 'SalePrice')
    params = {'learning_rate': 0.02, 'max_depth': -1, 'boosting': 'gbdt', 'objective': 'regression', 'metric':'rmse', 'seed': 3, 'num_iterations': 1000, 'early_stopping_round': 200, 'verbose_eval': 300, 'num_leaves': 64, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5}
    sub = simple.Model(data, 'LGB', params, 0.2, 7).PRED
    sub['SalePrice'] = np.expm1(sub['SalePrice'])
    sub.to_csv('submission_lgb.csv', index=False)
    
    lr.fit(train[col], train['SalePrice'])
    print('Ridge', np.sqrt(metrics.mean_squared_error(train['SalePrice'], lr.predict(train[col]))))
    #print('Ridge on Outliers', np.sqrt(metrics.mean_squared_error(outliers['SalePrice'], lr.predict(outliers[col]))))
    test['SalePrice'] = np.expm1(lr.predict(test[col]))
    test[['Id','SalePrice']].to_csv('submission_ridge.csv', index=False)

    sub = simple.Blend(['submission_lgb.csv', 'submission_ridge.csv'], 'Id', 'SalePrice', 'blend' + str(i+2).zfill(2) + '.csv').BLEND.reset_index()
testing_linear = """
import signal
def handler(signum, frame):
    print ('Time out...', signum)
    raise 'Time out...'
signal.signal(signal.SIGALRM, handler)

col = [c for c in train.columns if c not in ['Id', 'SalePrice']]
x1, x2, y1, y2 = train_test_split(train[col], train['SalePrice'], test_size=0.2, random_state=3)

models=[]
for m in linear_model.__all__:
    try:
        signal.alarm(10)
        model = eval('linear_model.' + m + '(random_state=3)')
        model.fit(x1, y1)
        signal.alarm(0) 
        score = np.sqrt(metrics.mean_squared_error(y2, model.predict(x2)))
        models.append([score, m, model])
        print(m, score)
    except:
        print('\t'*2, m)
        
models = sorted(models)
print([[s, m] for s, m, model in models])
"""