import pandas as pd

file_folder = '../input/multi-stellar-seti-candidate-selection-part-1/'
file_name = 'stars_with_trimmed_skewness_of_anomaly.csv'
work_data = pd.read_csv(file_folder + file_name, dtype={'source_id': str})
len(work_data)
work_data.columns
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
np.random.seed(2018090003)
import types
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler 

def get_cv_model_transform(data_frame, label_extractor, var_extractor, trainer_factory, response_column='response', 
                           id_column='source_id', n_runs=2, n_splits=2, max_n_training=None, scale = False,
                           trim_fraction=None, classification=False):
    '''
    Creates a transform function that results from training a regression model with cross-validation.
    The transform function takes a frame and adds a response column to it.
    '''
    default_model_list = []
    sum_series = pd.Series([0] * len(data_frame))
    for r in range(n_runs):
        shuffled_frame = data_frame.sample(frac=1)
        shuffled_frame.reset_index(inplace=True, drop=True)
        response_frame = pd.DataFrame(columns=[id_column, 'response'])
        kf = KFold(n_splits=n_splits)
        first_fold = True
        for train_idx, test_idx in kf.split(shuffled_frame):
            train_frame = shuffled_frame.iloc[train_idx]
            if trim_fraction is not None:
                helper_labels = label_extractor(train_frame) if isinstance(label_extractor, types.FunctionType) else train_frame[label_extractor] 
                train_label_ordering = np.argsort(helper_labels)
                orig_train_len = len(train_label_ordering)
                head_tail_len_to_trim = int(round(orig_train_len * trim_fraction * 0.5))
                assert head_tail_len_to_trim > 0
                trimmed_ordering = train_label_ordering[head_tail_len_to_trim:-head_tail_len_to_trim]
                train_frame = train_frame.iloc[trimmed_ordering]
            if max_n_training is not None:
                train_frame = train_frame.sample(max_n_training)
            train_labels = label_extractor(train_frame) if isinstance(label_extractor, types.FunctionType) else train_frame[label_extractor]
            test_frame = shuffled_frame.iloc[test_idx]
            train_vars = var_extractor(train_frame)
            test_vars = var_extractor(test_frame)
            scaler = None
            if scale:
                scaler = StandardScaler()  
                scaler.fit(train_vars)
                train_vars = scaler.transform(train_vars)  
                test_vars = scaler.transform(test_vars) 
            trainer = trainer_factory()
            fold_model = trainer.fit(train_vars, train_labels)
            test_responses = fold_model.predict_proba(test_vars)[:,1] if classification else fold_model.predict(test_vars)
            test_id = test_frame[id_column]
            assert len(test_id) == len(test_responses)
            fold_frame = pd.DataFrame({id_column: test_id, 'response': test_responses})
            response_frame = pd.concat([response_frame, fold_frame], sort=False)
            if first_fold:
                first_fold = False
                default_model_list.append((scaler, fold_model,))
        response_frame.sort_values(id_column, inplace=True)
        response_frame.reset_index(inplace=True, drop=True)
        assert len(response_frame) == len(data_frame), 'len(response_frame)=%d' % len(response_frame)
        sum_series += response_frame['response']
    cv_response = sum_series / n_runs
    assert len(cv_response) == len(data_frame)
    assert len(default_model_list) == n_runs
    response_map = dict()
    sorted_id = np.sort(data_frame[id_column].values) 
    for i in range(len(cv_response)):
        response_map[str(sorted_id[i])] = cv_response[i]
    response_id_set = set(response_map)
    
    def _transform(_frame):
        _in_trained_set = _frame[id_column].astype(str).isin(response_id_set)
        _trained_frame = _frame[_in_trained_set].copy()
        _trained_frame.reset_index(inplace=True, drop=True)
        if len(_trained_frame) > 0:
            _trained_id = _trained_frame[id_column]
            _tn = len(_trained_id)
            _response = pd.Series([None] * _tn)
            for i in range(_tn):
                _response[i] = response_map[str(_trained_id[i])]
            _trained_frame[response_column] = _response
        _remain_frame = _frame[~_in_trained_set].copy()
        _remain_frame.reset_index(inplace=True, drop=True)
        if len(_remain_frame) > 0:
            _unscaled_vars = var_extractor(_remain_frame)
            _response_sum = pd.Series([0] * len(_remain_frame))
            for _model_tuple in default_model_list:
                _scaler = _model_tuple[0]
                _model = _model_tuple[1]
                _vars = _unscaled_vars if _scaler is None else _scaler.transform(_unscaled_vars)
                _response = _model.predict_proba(_vars)[:,1] if classification else _model.predict(_vars)
                _response_sum += _response
            _remain_frame[response_column] = _response_sum / len(default_model_list)
        _frames_list = [_trained_frame, _remain_frame]
        _result = pd.concat(_frames_list, sort=False)
        _result.reset_index(inplace=True, drop=True)
        return _result
    return _transform
import scipy.stats as stats

def print_reg_evaluation(data_frame, label_column, response_column):
    response = response_column(data_frame) if isinstance(response_column, types.FunctionType) else data_frame[response_column]
    label = label_column(data_frame) if isinstance(label_column, types.FunctionType) else data_frame[label_column]
    residual = label - response
    rmse = np.sqrt(sum(residual ** 2) / len(data_frame))
    correl = stats.pearsonr(response, label)[0]
    print('RMSE: %.4f | Correlation: %.4f' % (rmse, correl,), flush=True)
from sklearn.metrics import roc_auc_score

def print_cls_evaluation(data_frame, label_column, response_column):
    response = response_column(data_frame) if isinstance(response_column, types.FunctionType) else data_frame[response_column]
    label = label_column(data_frame) if isinstance(label_column, types.FunctionType) else data_frame[label_column]
    print('AUC: %.4f' % (roc_auc_score(label, response),))
def get_pos_features(data_frame):
    return data_frame[['x', 'y', 'z']]
from sklearn.ensemble import RandomForestRegressor

def get_pos_trainer():
    return RandomForestRegressor(n_estimators=60, max_depth=16, min_samples_split=10, random_state=np.random.randint(0, 10000))
MAX_N_TRAINING = 50000

pos_transform = get_cv_model_transform(work_data, 'best_std_skewness', get_pos_features, get_pos_trainer, 
    response_column='smooth_skewness', n_runs=2, n_splits=2, max_n_training=MAX_N_TRAINING)
work_data = pos_transform(work_data)
print_reg_evaluation(work_data, 'best_std_skewness', 'smooth_skewness')
n_to_show = 2000
low_outliers = work_data.sort_values('smooth_skewness', ascending=True).head(n_to_show)
import plotly.plotly as py
import plotly.offline as py
import plotly.graph_objs as go
import warnings

py.init_notebook_mode(connected=False)
def plot_pos_frame(pos_frame, star_color, sun_color = 'blue', bstar_color = 'black'):    
    star_color = [(bstar_color if row['source_id'] == '2081900940499099136' else (sun_color if row['source_id'] == 'sun' else star_color)) for _, row in pos_frame.iterrows()]
    trace1 = go.Scatter3d(
        x=pos_frame['x'],
        y=pos_frame['y'],
        z=pos_frame['z'],
        mode='markers',
        text=pos_frame['source_id'],
        marker=dict(
            size=3,
            color=star_color,
            opacity=0.5
        )
    )
    scatter_data = [trace1]
    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        ),
        scene=dict(
            annotations=[
                dict(
                    x=0,
                    y=0,
                    z=0,
                    text='Earth'
                )
            ]
        )
    )
    fig = go.Figure(data=scatter_data, layout=layout)
    py.iplot(fig)
def get_sun():
    new_frame = pd.DataFrame(columns=['source_id', 'x', 'y', 'z'])
    new_frame.loc[0] = ['sun', 0.0, 0.0, 0.0]
    new_frame.reset_index(inplace=True, drop=True)
    return new_frame
bstar_frame = work_data[work_data['source_id'] == '2081900940499099136']
low_outliers_with_extras = pd.concat([low_outliers, bstar_frame, get_sun()], sort=False)
plot_pos_frame(low_outliers_with_extras, 'red')
MAG_COLUMNS = ['allwise_w4', 'allwise_w3', 'allwise_w2', 
               'tmass_ks_m', 'tmass_h_m', 'tmass_j_m', 
               'phot_rp_mean_mag', 'phot_g_mean_mag',
               'tycho2_vt_mag', 'phot_bp_mean_mag', 
               'tycho2_bt_mag']
def load_orig_data():
    orig_data = pd.read_csv('../input/257k-gaia-dr2-stars/257k-gaiadr2-sources-with-photometry.csv', dtype={'source_id': str})
    result_data = orig_data.loc[:,MAG_COLUMNS]
    result_data['source_id'] = orig_data['source_id']
    result_data['parallax'] = orig_data['parallax']
    return result_data
orig_data = load_orig_data()
work_data.drop(['phot_g_mean_mag'], axis=1, inplace=True)
work_data = pd.merge(work_data, orig_data, how='inner', on='source_id')
def transform_absmag(data_frame):
    new_frame = data_frame.copy()
    distance = 1000.0 / new_frame['parallax']
    dist_mod = 5 * (np.log10(distance) - 1)
    for mc in MAG_COLUMNS:
        # Convert to absolute magnitude
        new_frame[mc] = new_frame[mc] - dist_mod
    new_frame['distance'] = distance
    return new_frame
work_data = transform_absmag(work_data)
def get_cls_features(data_frame):
    return data_frame[MAG_COLUMNS]
LEFT_SKEWNESS_FRACTION = 0.03
left_count = int(len(work_data) * LEFT_SKEWNESS_FRACTION)
in_region_data = work_data.sort_values('smooth_skewness').head(left_count)
len(in_region_data)
from sklearn.neighbors import BallTree

def get_matched_control_group(data_pool, exp_group, matching_columns, sample_fraction=0.5, k=15, n_factor=1):
    # Only a sample of the dataset is used to come up with a matched control group
    data_for_search = data_pool.sample(int(len(data_pool) * sample_fraction)).reset_index(drop=True)
    # Set of source_id's that are already in the matched control group
    source_id_set = set()

    ball_tree = BallTree(data_for_search[matching_columns])
    data_source_id = data_for_search['source_id']
    shuffled_exp_group = exp_group.sample(frac=1)    
    idx_source_id = shuffled_exp_group.columns.get_loc('source_id')
    matching_col_indexes = [shuffled_exp_group.columns.get_loc(cn) for cn in matching_columns]
    results = pd.DataFrame(columns=data_for_search.columns)
    for row in shuffled_exp_group.itertuples(index=False):
        source_pos = [row[i] for i in matching_col_indexes]
        _, index_matrix = ball_tree.query([source_pos], k=k)
        indexes = index_matrix[0]
        count = 0
        for i in range(k):
            data_index = indexes[i]
            source_id = data_source_id[data_index]
            if source_id not in source_id_set:
                source_id_set.add(source_id)
                results.loc[len(results)] = data_for_search.iloc[data_index]
                count += 1
                if count >= n_factor:
                    break
    return results
not_in_region_data = work_data[~work_data['source_id'].isin(set(in_region_data['source_id']))]
region_control_group = get_matched_control_group(not_in_region_data, in_region_data, ['distance', 'z'])
in_region_data['distance'].describe()
region_control_group['distance'].describe()
in_region_data['z'].describe()
region_control_group['z'].describe()
in_region_data['region_label'] = [1] * len(in_region_data)
region_control_group['region_label'] = [0] * len(region_control_group)
cls_work_data = pd.concat([in_region_data, region_control_group], sort=False)
len(cls_work_data)
from sklearn.neural_network import MLPClassifier

def get_nn_trainer():
    return MLPClassifier(solver='lbfgs', activation='tanh', alpha=0.001, max_iter=350,
        hidden_layer_sizes=(8,2,), random_state=np.random.randint(1,10000))
transform_nn = get_cv_model_transform(cls_work_data, 'region_label', get_cls_features, get_nn_trainer, 
    response_column='cls_response', n_runs=3, n_splits=5, max_n_training=None, classification=True,
    scale=True)
cls_work_data = transform_nn(cls_work_data)
print_cls_evaluation(cls_work_data, 'region_label', 'cls_response')
work_data = transform_nn(work_data)
def transform_logit(data_frame, prior = 0.5, prior_weight = 0.001):
    new_frame = data_frame.copy()
    damped_p = new_frame['cls_response'].astype(float) * (1 - prior_weight) + prior * prior_weight
    new_frame['cls_logit'] = np.log(damped_p / (1 - damped_p))
    return new_frame
work_data = transform_logit(work_data)
from scipy.stats import binned_statistic 

def plot_binned(x, y, x_label, y_label, bins=100):
    statistic, bin_edges, _ = binned_statistic(x.astype(float), y.astype(float), bins=bins)
    bin_x = (bin_edges[1:] + bin_edges[:-1]) / 2
    trace = go.Scatter(
        x = bin_x,
        y = statistic
    )
    layout = go.Layout(
        xaxis=dict(title=x_label),
        yaxis=dict(title=y_label)
    )
    fig = dict(data=[trace], layout=layout)
    py.iplot(fig)
plot_binned(work_data['distance'], work_data['cls_logit'], 'Distance', 'Classifier Logit')
from sklearn.linear_model import LinearRegression

def get_distcorr_trainer():
    return LinearRegression()
def get_distcorr_features(data_frame):
    d = data_frame['distance']
    log_d = np.log(d)
    return np.transpose([d, d ** 2])
transform_distcorr = get_cv_model_transform(work_data, 'cls_logit', get_distcorr_features, get_distcorr_trainer, 
    response_column='cls_logit_by_distance', n_runs=2, n_splits=3, max_n_training=None)
work_data = transform_distcorr(work_data)
def transform_distcorr_residual(data_frame):
    new_frame = data_frame.copy()
    new_frame['cls_distcorr_logit'] = new_frame['cls_logit'] - new_frame['cls_logit_by_distance']
    new_frame.drop(['cls_logit_by_distance'], axis=1, inplace=True)
    return new_frame
work_data = transform_distcorr_residual(work_data)
plot_binned(work_data['distance'], work_data['cls_distcorr_logit'], 'Distance', 'Corrected Classifier Logit')
N_CANDIDATES = 150
candidates = work_data.sort_values('cls_distcorr_logit', ascending=False).iloc[:N_CANDIDATES]
not_candidates = work_data[~work_data['source_id'].isin(set(candidates['source_id']))]
cand_matched_control_group = get_matched_control_group(not_candidates, candidates, ['distance', 'z'])
candidates[['source_id', 'cls_distcorr_logit']].to_csv('altered-spectra-candidates.csv', index=False)
cand_matched_control_group[['source_id', 'cls_distcorr_logit']].to_csv('matched-control-group.csv', index=False)
kep_src_raw = pd.read_csv('../input/gaia-dr2-stars-in-kepler-field/gaia-dr2-kepler-field-3-extra-dbs.csv', dtype={'source_id': str})
kep_src = transform_absmag(kep_src_raw)
kep_src = transform_nn(kep_src)
kep_src = transform_logit(kep_src)
kep_src = transform_distcorr(kep_src)
kep_src = transform_distcorr_residual(kep_src)
len(kep_src)
def giant_separation_y(x):
    return x * 40.0 - 25
def color_index(data_frame):
    return data_frame['phot_bp_mean_mag'] - data_frame['phot_rp_mean_mag']
def transform_rm_giants(data_frame):
    abs_mag_ne = data_frame['phot_g_mean_mag'] # Already transformed to absolute
    new_frame = data_frame[abs_mag_ne ** 2 >= giant_separation_y(color_index(data_frame))]
    new_frame.reset_index(inplace=True, drop=True)
    return new_frame
kep_src = transform_rm_giants(kep_src)
len(kep_src)
kep_src_no_bstar = kep_src[kep_src['kepid'] != 8462852]
kep_candidates = kep_src_no_bstar.sort_values('cls_distcorr_logit', ascending=False).iloc[:100]
assert '2081900940499099136' not in kep_candidates['source_id']
not_kep_candidates = kep_src_no_bstar[~kep_src_no_bstar['source_id'].isin(set(kep_candidates['source_id']))]
kep_cand_matched_control_group = get_matched_control_group(not_kep_candidates, kep_candidates, ['distance'])
kep_candidates[['source_id', 'kepid', 'cls_distcorr_logit']].to_csv('kepler-altered-spectra-candidates.csv', index=False)
kep_cand_matched_control_group[['source_id', 'kepid', 'cls_distcorr_logit']].to_csv('kepler-matched-control-group.csv', index=False)