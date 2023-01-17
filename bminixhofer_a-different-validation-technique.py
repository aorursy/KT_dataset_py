import lightgbm as lgb
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold, BaseCrossValidator
from sklearn.decomposition import TruncatedSVD, FastICA, FactorAnalysis
from sklearn.random_projection import SparseRandomProjection
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import Ridge
from sklearn.preprocessing import scale
from scipy.stats import skew, kurtosis, gmean, ks_2samp
import gc
import psutil
from tqdm._tqdm_notebook import tqdm_notebook as tqdm

tqdm.pandas()
sns.set(style="white", color_codes=True)
%matplotlib inline
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
def aggregate_row(row):
    non_zero_values = row.iloc[row.nonzero()].astype(float)
    if non_zero_values.empty:
        aggs = {
            'non_zero_mean': np.nan,
            'non_zero_std': np.nan,
            'non_zero_max': np.nan,
            'non_zero_min': np.nan,
            'non_zero_sum': np.nan,
            'non_zero_skewness': np.nan,
            'non_zero_kurtosis': np.nan,
            'non_zero_median': np.nan,
            'non_zero_q1': np.nan,
            'non_zero_q3': np.nan,
            'non_zero_gmean': np.nan,
            'non_zero_log_mean': np.nan,
            'non_zero_log_std': np.nan,
            'non_zero_log_max': np.nan,
            'non_zero_log_min': np.nan,
            'non_zero_log_sum': np.nan,
            'non_zero_log_skewness': np.nan,
            'non_zero_log_kurtosis': np.nan,
            'non_zero_log_median': np.nan,
            'non_zero_log_q1': np.nan,
            'non_zero_log_q3': np.nan,
            'non_zero_log_gmean': np.nan,
            'non_zero_count': np.nan,
            'non_zero_fraction': np.nan
        }
    else:
        aggs = {
            'non_zero_mean': non_zero_values.mean(),
            'non_zero_std': non_zero_values.std(),
            'non_zero_max': non_zero_values.max(),
            'non_zero_min': non_zero_values.min(),
            'non_zero_sum': non_zero_values.sum(),
            'non_zero_skewness': skew(non_zero_values),
            'non_zero_kurtosis': kurtosis(non_zero_values),
            'non_zero_median': non_zero_values.median(),
            'non_zero_q1': np.percentile(non_zero_values, q=25),
            'non_zero_q3': np.percentile(non_zero_values, q=75),
            'non_zero_gmean': gmean(non_zero_values),
            'non_zero_log_mean': np.log1p(non_zero_values).mean(),
            'non_zero_log_std': np.log1p(non_zero_values).std(),
            'non_zero_log_max': np.log1p(non_zero_values).max(),
            'non_zero_log_min': np.log1p(non_zero_values).min(),
            'non_zero_log_sum': np.log1p(non_zero_values).sum(),
            'non_zero_log_skewness': skew(np.log1p(non_zero_values)),
            'non_zero_log_kurtosis': kurtosis(np.log1p(non_zero_values)),
            'non_zero_log_median': np.log1p(non_zero_values).median(),
            'non_zero_log_q1': np.percentile(np.log1p(non_zero_values), q=25),
            'non_zero_log_q3': np.percentile(np.log1p(non_zero_values), q=75),
            'non_zero_log_gmean': gmean(np.log1p(non_zero_values)),
            'non_zero_count': non_zero_values.count(),
            'non_zero_fraction': non_zero_values.count() / row.count()
        }
    return pd.Series(aggs, index=list(aggs.keys()))
eng_features = train.iloc[:, 2:].progress_apply(aggregate_row, axis=1)
eng_features_test = test.iloc[:, 1:].progress_apply(aggregate_row, axis=1)
train_matrix = np.hstack([
    eng_features.values
])

test_matrix = np.hstack([
    eng_features_test.values
])
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.1,
    'num_leaves': 16,
    'max_depth': -1,
    'min_child_samples': 1,
    'max_bin': 300,
    'subsample': 1.0,
    'subsample_freq': 1,
    'colsample_bytree': 0.5,
    'min_child_weight': 10,
    'reg_lambda': 0.1,
    'reg_alpha': 0.0,
    'scale_pos_weight': 1,
    'zero_as_missing': False,
    'num_threads': -1,
}

adversarial_x = np.vstack([
    train_matrix,
    test_matrix
])
adversarial_y = np.ones(len(adversarial_x))
adversarial_y[:len(train_matrix)] = 0

cv = KFold(n_splits=5, random_state=100, shuffle=True)
train_preds = np.zeros((len(adversarial_x)))

for i, (train_index, valid_index) in enumerate(cv.split(adversarial_y)):
    print(f'Fold {i}')
    
    dtrain = lgb.Dataset(adversarial_x[train_index], 
                         label=adversarial_y[train_index])
    dvalid = lgb.Dataset(adversarial_x[valid_index], 
                         label=adversarial_y[valid_index])
    
    evals_result = {}
    model = lgb.train(lgb_params, dtrain,
                      num_boost_round=10000, 
                      valid_sets=[dtrain, dvalid],
                      valid_names=['train', 'valid'],
                      early_stopping_rounds=100, 
                      verbose_eval=2000, 
                      evals_result=evals_result)

    valid_preds = model.predict(adversarial_x[valid_index])
    train_preds[valid_index] = valid_preds
    
print('Overall ROC AUC', roc_auc_score(adversarial_y, train_preds))
predictions = pd.Series(train_preds[:len(train)])
predictions_sample = predictions.sample(frac=1)
sns.jointplot(predictions_sample.index, predictions_sample, size=10, stat_func=None,
              marginal_kws=dict(bins=15), joint_kws=dict(s=3))
np.save('weights.npy', predictions.values)
weights = predictions + 0.1
weights = weights / weights.mean()

def weighted_rmsle(y_true, y_pred, index):
    errors = (np.log1p(y_true) - np.log1p(y_pred)) ** 2
    errors = errors * weights[index]
    
    return np.sqrt(np.mean(errors))

def rmsle(y_true, y_pred):
    errors = (np.log1p(y_true) - np.log1p(y_pred)) ** 2

    return np.sqrt(np.mean(errors))
x_train = train.iloc[:, 2:].values
x_test = test.iloc[:, 1:].values
y_train = np.log(train['target'])

_, unique_indices = np.unique(x_train, return_index=True, axis=1)
variance_greater_zero = x_train.var(axis=0) > 0

mask = np.zeros(x_train.shape[1], dtype=bool)
mask[unique_indices] = True
mask[variance_greater_zero] = True

x_train = x_train[:, mask]
x_test = x_test[:, mask]

x_train.shape, x_test.shape
decomposer = FeatureUnion([
    ('svd', TruncatedSVD(n_components=50, random_state=100)),
    ('ica', FastICA(n_components=20, random_state=100))
])

decomposed_train = decomposer.fit_transform(x_train)
decomposed_test = decomposer.transform(x_test)
train_matrix = np.hstack([
    eng_features.values,
    decomposed_train
])
test_matrix = np.hstack([
    eng_features_test.values,
    decomposed_test
])
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.01,
    'num_leaves': 16,
    'max_depth': -1,
    'min_child_samples': 1,
    'max_bin': 300,
    'subsample': 1.0,
    'subsample_freq': 1,
    'colsample_bytree': 0.5,
    'min_child_weight': 10,
    'reg_lambda': 0.1,
    'reg_alpha': 0.0,
    'scale_pos_weight': 1,
    'zero_as_missing': False,
    'num_threads': -1,
}
class KFoldByTargetValue(BaseCrossValidator):
    def __init__(self, n_splits=3, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def _iter_test_indices(self, X, y=None, groups=None):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        sorted_idx_vals = sorted(zip(indices, X), key=lambda x: x[1])
        indices = [idx for idx, val in sorted_idx_vals]

        for split_start in range(self.n_splits):
            split_indeces = indices[split_start::self.n_splits]
            yield split_indeces

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_split
cv = KFoldByTargetValue(n_splits=5, shuffle=True, random_state=100)

train_preds = np.zeros((len(train)))
test_preds = np.zeros((len(test)))

for i, (train_index, valid_index) in enumerate(cv.split(y_train)):
    print(f'Fold {i}')
    
    dtrain = lgb.Dataset(train_matrix[train_index], 
                         label=y_train[train_index])
    dvalid = lgb.Dataset(train_matrix[valid_index], 
                         label=y_train[valid_index])
    
    evals_result = {}
    model = lgb.train(lgb_params, dtrain,
                      num_boost_round=10000, 
                      valid_sets=[dtrain, dvalid],
                      valid_names=['train', 'valid'],
                      early_stopping_rounds=100, 
                      verbose_eval=2000, 
                      evals_result=evals_result)
    
    valid_preds = np.exp(model.predict(train_matrix[valid_index]))
    test_preds += np.exp(model.predict(test_matrix)) / cv.n_splits
    
    train_preds[valid_index] = valid_preds
    
    print('RMSLE: ', rmsle(np.exp(y_train[valid_index]), valid_preds))
    print('Weighted RMSLE: ', weighted_rmsle(np.exp(y_train[valid_index]), valid_preds, valid_index))
print()
print('Overall RMSLE: ', rmsle(np.exp(y_train), train_preds))
print('Overall Weighted RMSLE: ', weighted_rmsle(np.exp(y_train), train_preds, np.arange(len(train_preds))))
submission = pd.DataFrame()
submission['ID'] = test['ID']
submission['target'] = test_preds
submission.to_csv('submission.csv', index=False)