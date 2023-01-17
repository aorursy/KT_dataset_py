import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import matplotlib as mpl
# some matplotlib niceness
mpl.rcParams['axes.color_cycle'] = ['e6194b', '3cb44b', 'ffe119', '0082c8',
                                    'f58231', '911eb4', '46f0f0', 'f032e6',
                                    'd2f53c', 'fabebe', '008080', 'e6beff',
                                    '800000', 'aaffc3', 'ffd8b1']
mpl.rcParams['figure.figsize'] = (12.8, 7.2)
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['figure.facecolor'] = '#303030'
mpl.rcParams['figure.edgecolor'] = '#303030'
mpl.rcParams['axes.facecolor'] = '#303030'
mpl.rcParams['axes.edgecolor'] = 'white'
mpl.rcParams['axes.labelcolor'] = 'white'
mpl.rcParams['axes.linewidth'] = 0.4
mpl.rcParams['xtick.major.width'] = 0.4
mpl.rcParams['xtick.minor.width'] = 0.2
mpl.rcParams['ytick.major.width'] = 0.4
mpl.rcParams['ytick.minor.width'] = 0.2
mpl.rcParams['grid.linestyle'] = '--'
mpl.rcParams['grid.linewidth'] = 0.4
mpl.rcParams['grid.alpha'] = 0.8
mpl.rcParams['xtick.color'] = 'white'
mpl.rcParams['ytick.color'] = 'white'
mpl.rcParams['text.color'] = 'white'
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.fancybox'] = False
# load the data and concat train and test features into one DF
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
submission = pd.read_csv('../input/sample_submission.csv')
all_data = pd.concat((train.iloc[:, 2:], test.iloc[:, 1:]), ignore_index=True)
# as you already know the data is quite sparse
# but it also of the same scale like target
# so i suppose the data is encoded timeseries
print('mean of std and max of feature values:')
all_data.describe().loc[['std', 'max']].mean(axis=1)
# The class is for preprocessing
# it computes row-based statistics on full data
class StatsDatasetCreator(object):
    def get_nonzero(self, data):
        '''
        Transforms original data to np.array of np.arrays
        dropping zero values.
        '''
        numerical_cols = [x for x in data.columns  \
                          if x not in ['ID', 'target']]
        values = data[numerical_cols] \
                 .apply(lambda x: np.log1p(x.values[np.nonzero(x)]),
                        axis=1) \
                 .values
        return values

    def create_stats_dataset(self, data, target, maxlen=1989):
        '''
        Main function, returns full processed dataset
        '''
        values = self.get_nonzero(data)
        data_length = len(data)
        stats_array = np.zeros((data_length, 108))
        stats_array[:, 0] = np.vectorize(np.count_nonzero)(values)
        stats_array[:, 1] = np.vectorize(np.mean)(values)
        stats_array[:, 2] = np.vectorize(np.std)(values)
        for i, v in enumerate(values):
            stats_array[i, 3:104] = np.percentile(v,
                                                  np.linspace(0, 100, 101))

        values_pad = pad_sequences(values, maxlen=maxlen)
        values_diff = values_pad[:, 1:] - values_pad[:, :-1]
        stats_array[:, 104] = values_diff.min(axis=1)
        stats_array[:, 105] = values_diff.max(axis=1)
        stats_array[:, 106] = values_diff.mean(axis=1)
        stats_array[:, 107] = np.nanmedian(np.where(values_diff == 0,
                                                    np.nan,
                                                    values_diff),
                                           axis=1)
        
        hist = self.get_hist(values)
        probs = self.get_probs(values, hist)
        probs_array = np.zeros((len(data), 103))
        probs_array[:, 0] = np.vectorize(np.mean)(probs)
        probs_array[:, 1] = np.vectorize(np.std)(probs)
        for i, p in enumerate(probs):
            probs_array[i, 2:103] = np.percentile(p, np.linspace(0, 100, 101))
        
        transformed_data = np.concatenate((stats_array, probs_array), axis=1)
        return transformed_data[:len(target)], transformed_data[len(target):]

    def get_hist(self, values):
        '''
        Returns histogram of values in dataset
        '''
        values_ravel = np.concatenate([*values])
        vals, bins = np.histogram(values_ravel,
                                  density=True,
                                  bins=100)
        probs = vals * (bins[1:] - bins[:-1])
        hist = [vals, bins, probs]
        return hist

    def digitizer(self, val, hist):
        '''
        Returns probability of val
        '''
        vals, bins, probs = hist
        b = np.digitize(val, bins[:-1]) - 1
        v = vals[b]
        p_less = probs[:b].sum()
        return p_less

    def get_probs(self, values, hist):
        '''
        Transforms values to their probabilities
        '''
        probs = []
        for i, v in enumerate(values):
            probs.append(np.vectorize(lambda x: self.digitizer(x, hist))(v))
        return probs
y = np.log1p(train['target'].copy().values)
ds_creator = StatsDatasetCreator()
x, x_test = ds_creator.create_stats_dataset(all_data, y)
corrs = np.apply_along_axis(lambda x: np.corrcoef(y, x)[0][1], axis=0, arr=x)

fig, ax = plt.subplots()
ax.hist(corrs, bins=100, edgecolor='k')
ax.set_xlabel('Correlation')
ax.set_title('New features correlation hist')
ax.set_ylabel('Frequency');
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

features = ['nonzero', 'mean', 'std'] + ['{}_percentile'.format(x) for x in range(0, 101)] + \
           ['min_diff', 'max_diff', 'mean_diff', 'median_diff'] + ['mean_probs', 'std_probs'] + \
           ['{}_probs_percentile'.format(x) for x in range(0, 101)]
rseed = np.random.RandomState(0)
folds = rseed.randint(0, 10, size=x.shape[0])

res = []
x_prediction = np.zeros((len(train), ))
x_prediction_test = np.zeros((len(test), ))


all_feature_importance_df  = pd.DataFrame()
params = {'task': 'train',
          'objective': 'regression',
          'metric': 'rmse',
          'colsample_bytree': 0.3,
          'learning_rate': 0.05}
evals_result = {}
for fold_idx in np.unique(folds):

    print('Fold {} start'.format(fold_idx))    
    val_mask = folds == fold_idx
    x_train = x[~val_mask]
    x_val = x[val_mask]
    y_train = y[~val_mask]
    y_val = y[val_mask]
    
    evals_result[fold_idx] = {}
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_val = lgb.Dataset(x_val, y_val, reference=lgb_train)
    
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=10000,
                    valid_sets= (lgb_train, lgb_val),
                    verbose_eval=500,
                    evals_result=evals_result[fold_idx],
                    early_stopping_rounds=200)
    
    y_val_pred = gbm.predict(x_val, num_iteration=gbm.best_iteration)
    x_prediction[val_mask] = y_val_pred
    r = mean_squared_error(y_val_pred, y_val)
    res.append(np.sqrt(r))
    
    y_test_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration)
    x_prediction_test += y_test_pred

    # Feature Importance
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = gbm.feature_importance()
    all_feature_importance_df = pd.concat([all_feature_importance_df, fold_importance_df], axis=0)

print('CV results', np.mean(res), np.std(res))
import seaborn as sns
cols = all_feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(
    by="importance", ascending=False)[:50].index
best_features = all_feature_importance_df.loc[all_feature_importance_df.feature.isin(cols)]
plt.figure(figsize=(8,10))
sns.barplot(x="importance", y="feature", 
            data=best_features.sort_values(by="importance", ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
submission['target'] = np.expm1(x_prediction_test / len(np.unique(folds)))
submission.to_csv('lgbm_on_stats.csv', index=False)