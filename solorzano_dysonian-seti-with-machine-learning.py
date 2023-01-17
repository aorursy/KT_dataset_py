import random
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
import types
import inspect
import plotly.plotly as py
import plotly.offline as py
import plotly.graph_objs as go
import warnings

from matplotlib import cm
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

warnings.simplefilter(action='ignore', category=DeprecationWarning)
py.init_notebook_mode(connected=False)

np.random.seed(201807002)

pd_concat_argspec = inspect.getfullargspec(pd.concat)
pd_concat_has_sort = 'sort' in pd_concat_argspec.args

def pd_concat(frames):
    # Due to Pandas versioning issue
    new_frame = pd.concat(frames, sort=False) if pd_concat_has_sort else pd.concat(frames)
    new_frame.reset_index(inplace=True, drop=True)
    return new_frame
    
def plt_hist(x, bins=30):
    # plt.hist() can be very slow.
    histo, edges = np.histogram(x, bins=bins)
    plt.bar(0.5 * edges[1:] + 0.5 * edges[:-1], histo, width=(edges[-1] - edges[0])/(len(edges) + 1))
data = pd.read_csv('../input/gaia-dr2-rave-35.csv', dtype={'source_id': str})
len(data)
train_mask = np.random.rand(len(data)) < 0.9
work_data = data[train_mask]
work_data.reset_index(inplace=True, drop=True)
test_data = data[~train_mask]
test_data.reset_index(inplace=True, drop=True)
data.columns
def get_cv_model_transform(data_frame, label_extractor, var_extractor, trainer, response_column='Response', id_column='source_id', n_splits=2, scale = False):
    shuffled_frame = data_frame.sample(frac=1).reset_index(drop=True)
    nrow = len(data_frame)
    kf = KFold(n_splits=n_splits)
    response_map = dict()
    default_model = None
    default_scaler = None
    split_idx = 0
    for train_idx, test_idx in kf.split(shuffled_frame):
        train_frame = shuffled_frame.iloc[train_idx]
        test_frame = shuffled_frame.iloc[test_idx]
        train_labels = label_extractor(train_frame) if isinstance(label_extractor, types.FunctionType) else train_frame[label_extractor]
        train_vars = var_extractor(train_frame)
        test_vars = var_extractor(test_frame)
        if scale:
            default_scaler = StandardScaler()  
            default_scaler.fit(train_vars)
            train_vars = default_scaler.transform(train_vars)  
            test_vars = default_scaler.transform(test_vars) 
        default_model = trainer.fit(train_vars, train_labels)
        test_frame.reset_index(inplace=True, drop=True)
        test_responses = default_model.predict(test_vars)
        test_id = test_frame[id_column]
        assert len(test_id) == len(test_responses)
        for i in range(len(test_id)):
            response = test_responses[i]
            key = str(test_id[i])
            response_map[key] = response
        split_idx += 1
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
        if len(_remain_frame) > 0:
            _vars = var_extractor(_remain_frame)
            if default_scaler is not None:
                _vars = default_scaler.transform(_vars)
            _response = default_model.predict(_vars)
            _remain_frame[response_column] = _response
        _frames_list = [_trained_frame, _remain_frame]
        _concat_frame = pd_concat(_frames_list)
        _concat_frame.reset_index(inplace=True, drop=True)
        return _concat_frame
    return _transform
def print_evaluation(data_frame, label_column, response_column):
    _response = response_column(data_frame) if isinstance(response_column, types.FunctionType) else data_frame[response_column]
    _label = label_column(data_frame) if isinstance(label_column, types.FunctionType) else data_frame[label_column]
    _error = _response - _label
    _rmse = math.sqrt(sum(_error ** 2) / len(data_frame))
    _correl = stats.pearsonr(_response, _label)[0]
    print('RMSE: %.4f | Correlation: %.4f' % (_rmse, _correl,), flush=True)
    
MIN_PARALLAX = 2.5
MAX_PARALLAX_ERROR = 0.2
def transform_init(data_frame):    
    parallax = data_frame['parallax']
    parallax_error = data_frame['parallax_error']
    new_frame = data_frame[(parallax >= MIN_PARALLAX) & (parallax_error <= MAX_PARALLAX_ERROR)].copy()
    new_frame.reset_index(inplace=True, drop=True)
    distance = 1000.0 / new_frame['parallax']
    new_frame['distance'] = distance
    new_frame['abs_mag_ne'] = new_frame['phot_g_mean_mag'] - 5 * (np.log10(distance) - 1)
    return new_frame
work_data = transform_init(work_data)
len(work_data)
wd_distance_mod = 5 * (np.log10(work_data['distance']) - 1)
wd_photo_distance_mod = 5 * (np.log10(1000.0 / work_data['r_parallax']) - 1)
np.sqrt(mean_squared_error(wd_distance_mod, wd_photo_distance_mod))
GR = 100
def extract_model_vars(data_frame):
    distance = data_frame['distance'].values
    log_distance = np.log(distance)
    g_mag = data_frame['phot_g_mean_mag']
    bp_mag = data_frame['phot_bp_mean_mag']
    rp_mag = data_frame['phot_rp_mean_mag']
    longitude_raw = data_frame['l'].values
    longitude = [(lng if lng <= 180 else lng - 360) for lng in longitude_raw]
    latitude = data_frame['b'].values
    sin_lat = np.sin(np.deg2rad(latitude))
    lat_ext_metric_prelim = np.abs(GR / sin_lat)
    lat_ext_metric = [min(distance[i], lat_ext_metric_prelim[i]) for i in range(len(data_frame))]
    metallicity = data_frame['r_metallicity']
    radial_velocity = data_frame['r_hrv']
    mg = data_frame['r_mg']
    si = data_frame['r_si']
    fe = data_frame['r_fe']
    jmag = data_frame['r_jmag_2mass']
    hmag = data_frame['r_hmag_2mass']
    kmag = data_frame['r_kmag_2mass']
    aw_m1 = data_frame['r_w1mag_allwise']
    aw_m2 = data_frame['r_w2mag_allwise']
    aw_m3 = data_frame['r_w3mag_allwise']
    aw_m4 = data_frame['r_w4mag_allwise']
    denis_imag = data_frame['r_imag_denis']
    denis_jmag = data_frame['r_jmag_denis']
    denis_kmag = data_frame['r_kmag_denis']    
    apass_bmag = data_frame['r_bmag_apassdr9']
    apass_vmag = data_frame['r_vmag_apassdr9']
    apass_rpmag = data_frame['r_rpmag_apassdr9']
    apass_ipmag = data_frame['r_ipmag_apassdr9']
    
    color1 = hmag - jmag
    color2 = kmag - hmag
    color3 = rp_mag - kmag
    color4 = g_mag - rp_mag
    color5 = bp_mag - g_mag
    color6 = aw_m2 - aw_m1
    color7 = aw_m3 - aw_m2
    color8 = aw_m4 - aw_m3
    color9 = rp_mag - aw_m4
    color10 = g_mag - denis_imag
    color11 = denis_imag - denis_jmag
    color12 = denis_jmag - denis_kmag    
    color13 = apass_rpmag - apass_ipmag
    color14 = apass_vmag - apass_rpmag
    color15 = apass_bmag - apass_vmag
    color16 = g_mag - apass_bmag
    
    return np.transpose([log_distance, distance,
            color1, color2, color3, color4, color5,
            color6, color7, color8, color9,
            color10, color11, color12,
            color13, color14, color15, color16,
            mg, si, fe, metallicity,
            latitude, longitude, lat_ext_metric
            ])    
LABEL_COLUMN = 'phot_g_mean_mag'
transform_linear = get_cv_model_transform(work_data, LABEL_COLUMN, extract_model_vars, linear_model.LinearRegression(), n_splits=2, response_column='linear_' + LABEL_COLUMN, scale=True)
work_data = transform_linear(work_data)
print_evaluation(work_data, LABEL_COLUMN, 'linear_' + LABEL_COLUMN)
def get_gbm_trainer():
    return xgb.XGBRegressor(n_estimators=550, learning_rate=0.05, gamma=0.01, subsample=0.75,
                           colsample_bytree=1.0, max_depth=8, random_state=np.random.randint(1,10000))
def get_gbm_transform(label_column):
    return get_cv_model_transform(work_data, label_column, extract_model_vars, 
                get_gbm_trainer(), 
                n_splits=2, response_column='gbm_' + label_column)
transform_gbm = get_gbm_transform(LABEL_COLUMN)
work_data = transform_gbm(work_data)
print_evaluation(work_data, LABEL_COLUMN, 'gbm_' + LABEL_COLUMN)
def get_gbm2_trainer():
    return xgb.XGBRegressor(n_estimators=500, learning_rate=0.07, gamma=0.003, subsample=0.80,
                           colsample_bytree=1.0, max_depth=7, random_state=np.random.randint(1,10000))
def get_gbm2_transform(label_column):
    return get_cv_model_transform(work_data, label_column, extract_model_vars, 
                get_gbm2_trainer(), 
                n_splits=2, response_column='gbm2_' + label_column)
transform_gbm2 = get_gbm2_transform(LABEL_COLUMN)
work_data = transform_gbm2(work_data)
print_evaluation(work_data, LABEL_COLUMN, 'gbm2_' + LABEL_COLUMN)
nn_seed = np.random.randint(1,10000)
def get_nn_trainer():
    return MLPRegressor(hidden_layer_sizes=(30, 10), max_iter=500, alpha=0.1, random_state=nn_seed)
def get_nn_transform(label_column):
    return get_cv_model_transform(work_data, label_column, extract_model_vars, get_nn_trainer(), 
        n_splits=3, response_column='nn_' + label_column, scale=True)
transform_nn = get_nn_transform(LABEL_COLUMN)
work_data = transform_nn(work_data)
print_evaluation(work_data, LABEL_COLUMN, 'nn_' + LABEL_COLUMN)
def extract_blend_vars(data_frame):
    gbm_responses = data_frame['gbm_' + LABEL_COLUMN].values
    gbm2_responses = data_frame['gbm2_' + LABEL_COLUMN].values
    nn_responses = data_frame['nn_' + LABEL_COLUMN].values
    linear_responses = data_frame['linear_' + LABEL_COLUMN].values
    return np.transpose([gbm_responses, gbm2_responses, nn_responses, linear_responses])
def get_blend_trainer():
    return linear_model.LinearRegression()
def get_blend_transform(label_column):
    return get_cv_model_transform(work_data, label_column, extract_blend_vars, get_blend_trainer(), 
                n_splits=5, response_column='blend_' + label_column)
transform_blend = get_blend_transform(LABEL_COLUMN)
work_data = transform_blend(work_data)
print_evaluation(work_data, LABEL_COLUMN, 'blend_' + LABEL_COLUMN)
MODEL_PREFIX = 'blend_'
def error(data_frame, label_column):
    return data_frame[label_column] - data_frame[MODEL_PREFIX + label_column]
def transform_error(data_frame):
    new_frame = data_frame.copy()
    new_frame['error_' + LABEL_COLUMN] = error(data_frame, LABEL_COLUMN)
    return new_frame
work_data = transform_error(work_data)
def get_abs_error_label(data_frame):
    return np.abs(data_frame['error_' + LABEL_COLUMN])
def extract_error_vars(data_frame):
    parallax = data_frame['parallax']
    parallax_error = data_frame['parallax_error']
    parallax_high = parallax + parallax_error
    parallax_low = parallax - parallax_error
    var_error_diff = np.log(parallax_high) - np.log(parallax_low)
    distance = data_frame['distance']
    longitude = data_frame['l']
    latitude = data_frame['b']
    radial_velocity = data_frame['r_hrv']
    return np.transpose([
        var_error_diff,
        distance,
        longitude,
        latitude,
        radial_velocity
    ])
def get_error_trainer():
    return RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=2, min_samples_leaf=2)
transform_expected_error = get_cv_model_transform(work_data, get_abs_error_label, extract_error_vars, get_error_trainer(), 
                n_splits=3, response_column='expected_error_' + LABEL_COLUMN)
work_data = transform_expected_error(work_data)
print_evaluation(work_data, get_abs_error_label, 'expected_error_' + LABEL_COLUMN)
def transform_anomaly(data_frame):
    new_frame = data_frame.copy()
    new_frame['anomaly_' + LABEL_COLUMN] = new_frame['error_' + LABEL_COLUMN] / new_frame['expected_error_' + LABEL_COLUMN]
    return new_frame
work_data = transform_anomaly(work_data)
transform_list = [transform_init, 
                  transform_linear, transform_gbm, transform_gbm2, transform_nn, 
                  transform_blend,
                  transform_error, transform_expected_error, transform_anomaly]
def combined_transform(data_frame):
    _frame = data_frame
    for t in transform_list:
        _frame = t(_frame)
    return _frame
test_data = combined_transform(test_data)
np.std(test_data['error_' + LABEL_COLUMN])
data = combined_transform(data)
CAND_SD_THRESHOLD = 3.0
data_anomalies = data['anomaly_' + LABEL_COLUMN]
anomaly_std = np.std(data_anomalies)
cand_threshold = anomaly_std * CAND_SD_THRESHOLD
candidates = data[data_anomalies >= cand_threshold]
len(candidates)
bright_control_group = data.sort_values('anomaly_' + LABEL_COLUMN, ascending=True).head(len(candidates))
normal_control_group = data[(data_anomalies < anomaly_std) & (data_anomalies > -anomaly_std)].sample(len(candidates))
data_sample = data.sample(1000)
def abs_mag_value(data_frame, mag_column):
    _distance_mod = 5 * np.log10(data_frame['distance']) - 5
    return data_frame[mag_column] - _distance_mod
MODEL_RESPONSE_COLUMN = MODEL_PREFIX + LABEL_COLUMN

plt.rcParams['figure.figsize'] = (10, 5)
plt.scatter(abs_mag_value(data_sample, MODEL_RESPONSE_COLUMN), abs_mag_value(data_sample, LABEL_COLUMN), color=(0.5,0.5,0.5,0.5,), s=1)
plt.scatter(abs_mag_value(candidates, MODEL_RESPONSE_COLUMN), abs_mag_value(candidates, LABEL_COLUMN), color='green', s=6)
plt.scatter(abs_mag_value(bright_control_group, MODEL_RESPONSE_COLUMN), abs_mag_value(bright_control_group, LABEL_COLUMN), color='orange', s=2)
plt.ylim(-1, 11)
plt.gca().invert_yaxis()
plt.title('Model Diagram')
plt.xlabel('Modeled absolute magnitude')
plt.ylabel('Observed absolute magnitude')
plt.show()
def color_value(data_frame):
    return data_frame['phot_bp_mean_mag'] - data_frame['phot_rp_mean_mag']
def separation_y(color_index):
    return -0.1 + 4.6 * color_index
plt.rcParams['figure.figsize'] = (10, 5)
plt.scatter(color_value(data_sample), abs_mag_value(data_sample, LABEL_COLUMN), color=(0.5,0.5,0.5,0.5,), s=1)
plt.scatter(color_value(bright_control_group), abs_mag_value(bright_control_group, LABEL_COLUMN), color='orange', s=2)
plt.scatter(color_value(candidates), abs_mag_value(candidates, LABEL_COLUMN), color='green', s=6)
sep_x = np.linspace(0.5, 2.1, 100)
sep_y = separation_y(sep_x)
plt.plot(sep_x, sep_y, '--', color='blue')
plt.ylim(-1, 11)
plt.xlim(0.5, 2.1)
plt.gca().invert_yaxis()
plt.title('H-R Diagram')
plt.xlabel('BP - RP color index')
plt.ylabel('Absolute magnitude')
plt.show()
mainseq_mask = abs_mag_value(candidates, LABEL_COLUMN) > separation_y(color_value(candidates))
candidates_mainseq = candidates[mainseq_mask]
candidates_bright = candidates[~mainseq_mask]
len(candidates_mainseq)
len(candidates_bright)
def get_position_frame(data_frame):
    new_frame = pd.DataFrame()
    new_frame['source_id'] = data_frame['source_id'].values
    distance = data_frame['distance'].values
    latitude = np.deg2rad(data_frame['b'].values)
    longitude = np.deg2rad(data_frame['l'].values)
    new_frame['z'] = distance * np.sin(latitude)
    projection = distance * np.cos(latitude)
    new_frame['x'] = projection * np.cos(longitude)
    new_frame['y'] = projection * np.sin(longitude)
    new_frame['is_mainseq'] = (abs_mag_value(candidates, LABEL_COLUMN) > separation_y(color_value(candidates))).values
    return new_frame
candidates_pos_frame = get_position_frame(candidates)
def plot_pos_frame(pos_frame, mainseq_color = 'red', other_color= 'red'):
    star_color = [(mainseq_color if v else other_color) for v in pos_frame['is_mainseq'].values]
    trace1 = go.Scatter3d(
        x=pos_frame['x'],
        y=pos_frame['y'],
        z=pos_frame['z'],
        mode='markers',
        text=candidates['source_id'],
        marker=dict(
            size=4,
            color=star_color,
            opacity=0.67
        )
    )
    scatter_data = [trace1]
    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )
    fig = go.Figure(data=scatter_data, layout=layout)
    py.iplot(fig)
plot_pos_frame(candidates_pos_frame, 'blue', 'green')
bright_control_group_pos_frame = get_position_frame(bright_control_group)
plot_pos_frame(bright_control_group_pos_frame, 'red', 'red')
candidates['distance'].describe()
bright_control_group['distance'].describe()
anomalous_pos_frame = pd_concat([candidates_pos_frame, bright_control_group_pos_frame])
apf_len = len(anomalous_pos_frame)
apf_source_id_idx = anomalous_pos_frame.columns.get_loc('source_id')
apf_x_idx = anomalous_pos_frame.columns.get_loc('x')
apf_y_idx = anomalous_pos_frame.columns.get_loc('y')
apf_z_idx = anomalous_pos_frame.columns.get_loc('z')
new_row_list = []
for i in range(apf_len):
    row1 = anomalous_pos_frame.iloc[i]
    source1 = row1[apf_source_id_idx]
    x1 = row1[apf_x_idx]
    y1 = row1[apf_y_idx]
    z1 = row1[apf_z_idx]
    for j in range(i + 1, apf_len):
        row2 = anomalous_pos_frame.iloc[j]
        source2 = row2[apf_source_id_idx]
        x2 = row2[apf_x_idx]
        y2 = row2[apf_y_idx]
        z2 = row2[apf_z_idx]
        distance_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2
        new_row_list.append([source1, source2, distance_sq])
cross_distance_frame = pd.DataFrame(new_row_list, columns=['source1', 'source2', 'distance_sq'])
cross_distance_frame.sort_values('distance_sq', inplace=True)
candidate_source_set = set(candidates['source_id'])
cross_distance_frame['source1_dim'] = cross_distance_frame['source1'].isin(candidate_source_set)
cross_distance_frame['source2_dim'] = cross_distance_frame['source2'].isin(candidate_source_set)
cross_distance_frame.head(5)
len(cross_distance_frame[cross_distance_frame['source1_dim'] & cross_distance_frame['source2_dim']]) / len(cross_distance_frame)
dim_match_frequency = pd.DataFrame(columns=['count', 'frequency'])
for ss in range(5, 2000, 10):
    sub_frame = cross_distance_frame.iloc[:ss]
    count = len(sub_frame)
    freq = len(sub_frame[sub_frame['source1_dim'] & sub_frame['source2_dim']]) / count
    dim_match_frequency.loc[len(dim_match_frequency)] = [count, freq]
plt.rcParams['figure.figsize'] = (10, 5)
exp_line_x = [0, 2000]
exp_line_y = [0.25, 0.25]
hs_counts = dim_match_frequency['count']
tt_freqs = dim_match_frequency['frequency']
# standard margin of error
std_moe = np.sqrt(0.25 * (1 - 0.25) / hs_counts)
tt_freqs_low_95 = tt_freqs - 1.96 * std_moe
tt_freqs_low_99 = tt_freqs - 2.575 * std_moe
plt.plot(exp_line_x, exp_line_y, '--', color='orange')
plt.plot(hs_counts, tt_freqs, color='black', linewidth=5)
plt.plot(hs_counts, tt_freqs_low_95, color='red', linewidth=1)
plt.plot(hs_counts, tt_freqs_low_99, color='blue', linewidth=1)
plt.yticks(np.linspace(0, 1, 11))
plt.ylim(0, 1.0)
plt.xlim(0, 2000)
#plt.grid(color=(0.9, 0.9, 0.9,))
plt.title('Results of clustering test')
plt.xlabel('Size of headset of ordered cross-distance frame')
plt.ylabel('Frequency of True-True rows')
plt.show()
SAVED_COLUMNS = ['source_id', 'ra', 'dec', 'pmra', 'pmdec', 'l', 'b', 'distance', 'abs_mag_ne', 
                 'error_' + LABEL_COLUMN, 'anomaly_' + LABEL_COLUMN]
def save_data(data_frame, file_name):
    data_frame[SAVED_COLUMNS].to_csv(file_name)
save_data(data, 'all-sources.csv')
save_data(candidates, 'dim-candidates.csv')
save_data(candidates_mainseq, 'dim-candidates-mainseq.csv')
save_data(bright_control_group, 'bright-controls.csv')
save_data(normal_control_group, 'normal-controls.csv')
def sc(data_frame):
    new_frame = data_frame[['source_id', 'ra', 'dec', 'pmra', 'pmdec', 'distance', 'abs_mag_ne', 'error_phot_g_mean_mag', 'anomaly_phot_g_mean_mag']]
    new_frame.reset_index(inplace=True, drop=True)
    return new_frame
sc(candidates[abs_mag_value(candidates, LABEL_COLUMN) >= 1.45 + separation_y(color_value(candidates))])
closest_cand = candidates.sort_values('distance').head(15)
sc(closest_cand[closest_cand['ra'] >= 300])
data_close = data[data['distance'] < 75]
np.mean(np.abs(data_close['pmra']))
np.mean(np.abs(data_close['pmdec']))