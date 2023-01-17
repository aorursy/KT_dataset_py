import pandas as pd

data = pd.read_csv('../input/257k-gaiadr2-sources-with-photometry.csv', dtype={'source_id': str})
len(data)
_seen = set()
should_remove_set = set()
for _source_id in data['source_id']:
    if _source_id in _seen:
        should_remove_set.add(_source_id)
    else:
        _seen.add(_source_id)
data = data[~data['source_id'].isin(should_remove_set)].reset_index(drop=True)
len(data)
assert len(data) == len(set(data['source_id']))
import inspect

pd_concat_argspec = inspect.getfullargspec(pd.concat)
pd_concat_has_sort = 'sort' in pd_concat_argspec.args

def pd_concat(frames):
    # Due to Pandas versioning issue
    new_frame = pd.concat(frames, sort=False) if pd_concat_has_sort else pd.concat(frames)
    new_frame.reset_index(inplace=True, drop=True)
    return new_frame
import types
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler 

np.random.seed(201808011)

def model_results(data_frame, label_extractor, var_extractor, trainer_factory, id_column='source_id', n_splits=2, n_runs=3, scale=False, max_n_training=None):
    '''
    Returns a frame with source_id, response and residual columns, with the same ordering as data_frame.
    '''
    sum_series = pd.Series([0] * len(data_frame))
    for r in range(n_runs):
        shuffled_frame = data_frame.sample(frac=1)
        shuffled_frame.reset_index(inplace=True, drop=True)
        response_frame = pd.DataFrame(columns=[id_column, 'response'])
        kf = KFold(n_splits=n_splits)
        for train_idx, test_idx in kf.split(shuffled_frame):
            train_frame = shuffled_frame.iloc[train_idx]
            if max_n_training is not None:
                train_frame = train_frame.sample(max_n_training)
            test_frame = shuffled_frame.iloc[test_idx]
            train_labels = label_extractor(train_frame) if isinstance(label_extractor, types.FunctionType) else train_frame[label_extractor]
            train_vars = var_extractor(train_frame)
            test_vars = var_extractor(test_frame)
            if scale:
                scaler = StandardScaler()  
                scaler.fit(train_vars)
                train_vars = scaler.transform(train_vars)  
                test_vars = scaler.transform(test_vars) 
            trainer = trainer_factory()
            fold_model = trainer.fit(train_vars, train_labels)
            test_responses = fold_model.predict(test_vars)
            test_id = test_frame[id_column]
            assert len(test_id) == len(test_responses)
            fold_frame = pd.DataFrame({id_column: test_id, 'response': test_responses})
            response_frame = pd_concat([response_frame, fold_frame])
        response_frame.sort_values(id_column, inplace=True)
        response_frame.reset_index(inplace=True, drop=True)
        assert len(response_frame) == len(data_frame), 'len(response_frame)=%d' % len(response_frame)
        sum_series += response_frame['response']
    cv_response = sum_series / n_runs
    assert len(cv_response) == len(data_frame)
    sorted_result = pd.DataFrame({
        id_column: np.sort(data_frame[id_column].values), 
        'response': cv_response})
    data_frame_partial = pd.DataFrame({id_column: data_frame[id_column]})
    merged_frame = pd.merge(data_frame_partial, sorted_result, how='inner', on=id_column, sort=False)
    data_frame_labels = label_extractor(data_frame) if isinstance(label_extractor, types.FunctionType) else data_frame[label_extractor]
    merged_frame['residual'] = data_frame_labels - merged_frame['response']
    assert len(merged_frame) == len(data_frame)
    return merged_frame
import math
import scipy.stats as stats

def print_evaluation(data_frame, label_column, response_frame):
    _response = response_frame['response']
    _label = label_column(data_frame) if isinstance(label_column, types.FunctionType) else data_frame[label_column]
    _error = _label - _response
    assert sum(response_frame['residual'] == _error) == len(data_frame)
    _rmse = math.sqrt(np.sum(_error ** 2) / len(data_frame))
    _correl = stats.pearsonr(_response, _label)[0]
    print('RMSE: %.4f | Correlation: %.4f' % (_rmse, _correl,), flush=True)
def extract_mag_model_vars(data_frame):
    distance = (1000.0 / data_frame['parallax']).values
    log_distance = np.log(distance)

    feature_list = [log_distance, distance]
    feature_list.append(data_frame['phot_g_mean_mag'] - data_frame['phot_rp_mean_mag'])
    feature_list.append(data_frame['phot_bp_mean_mag'] - data_frame['phot_g_mean_mag'])
    
    return np.transpose(feature_list)    
LABEL_COLUMN = 'phot_g_mean_mag'
MAX_N_TRAINING = 40000
from sklearn.neural_network import MLPRegressor

def get_mag_trainer():
    return MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=500, alpha=0.1, random_state=np.random.randint(1, 10000))
mag_results = model_results(data, LABEL_COLUMN, extract_mag_model_vars, get_mag_trainer, n_runs=2, scale=True, max_n_training=MAX_N_TRAINING)
print_evaluation(data, LABEL_COLUMN, mag_results)
NUM_OUTLIERS = 1000
def produce_outliers():
    global data_dim_outliers
    global data_bright_outliers
    global idx_dim_outlier
    global idx_bright_outlier
    
    idx_res_sort = np.argsort(mag_results['residual'].values)
    idx_dim_outlier = idx_res_sort[-NUM_OUTLIERS:]
    idx_bright_outlier = idx_res_sort[:NUM_OUTLIERS]
    data_dim_outliers = data.iloc[idx_dim_outlier]
    data_bright_outliers = data.iloc[idx_bright_outlier]
import matplotlib.pyplot as plt
def plot_outliers():
    plt.rcParams['figure.figsize'] = (14, 7)
    plt.scatter(data_dim_outliers['ra'], data_dim_outliers['dec'], color='blue', s=8)
    plt.scatter(data_bright_outliers['ra'], data_bright_outliers['dec'], color='red', s=8)
    plt.title('Dim (blue) and bright (red) outliers')
    plt.xlabel('Right Ascension (degrees)')
    plt.ylabel('Declination (degrees)')
    plt.show()
    plt.scatter(data_dim_outliers['l'], data_dim_outliers['b'], color='blue', s=8)
    plt.scatter(data_bright_outliers['l'], data_bright_outliers['b'], color='red', s=8)
    plt.title('Dim (blue) and bright (red) outliers')
    plt.xlabel('Galactic longitude (degrees)')
    plt.ylabel('Galactic latitude (degrees)')
    plt.show()
    
produce_outliers()    
plot_outliers()
S_RA_MIN = 65
S_RA_MAX = 115
S_DEC_MIN = -68
S_DEC_MAX = -64
data_ra = data['ra']
data_dec = data['dec']
s_mask_bottom_left = (data_ra >= S_RA_MIN) & (data_ra <= S_RA_MAX) & (data_dec >= S_DEC_MIN) & (data_dec <= S_DEC_MAX)
s_mask_top_right = (data_ra >= S_RA_MIN + 180) & (data_ra <= S_RA_MAX + 180) & (data_dec >= -S_DEC_MAX) & (data_dec <= -S_DEC_MIN)
s_mask = s_mask_bottom_left | s_mask_top_right
should_remove_set = should_remove_set.union(set(data[s_mask]['source_id']))
len(should_remove_set)
data = data[~s_mask]
data.reset_index(inplace=True, drop=True)
mag_results = mag_results[~s_mask]
mag_results.reset_index(inplace=True, drop=True)
len(data)
assert np.sum(mag_results['source_id'] == data['source_id']) == len(data)
produce_outliers()
plot_outliers()
should_remove_frame = pd.DataFrame({
    'source_id': list(should_remove_set)
})
should_remove_frame.to_csv('257k-gaiadr2-should-remove.csv', index=False)