import pandas as pd



work_data = pd.read_csv('../input/hipparcos-gaia-data.csv')
len(work_data)
work_data.columns
import types

import numpy as np

import warnings

from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler 



warnings.simplefilter(action='ignore', category=FutureWarning)

np.random.seed(2019050001)



def get_cv_model_transform(data_frame, label_extractor, var_extractor, trainer_factory, response_column='response', 

                           id_column='source_id', n_runs=2, n_splits=2, max_n_training=None, scale=False,

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



def print_evaluation(data_frame, label_column, response_column):

    '''

    Compares a label with a model response and prints RMSE and correlation statistics.

    '''

    response = response_column(data_frame) if isinstance(response_column, types.FunctionType) else data_frame[response_column]

    label = label_column(data_frame) if isinstance(label_column, types.FunctionType) else data_frame[label_column]

    residual = label - response

    rmse = np.sqrt(sum(residual ** 2) / len(data_frame))

    correl = stats.pearsonr(response, label)[0]

    print('RMSE: %.5f | Correlation: %.4f' % (rmse, correl,), flush=True)
def extract_vars(data_frame):

    return data_frame[['phot_g_mean_mag','phot_bp_mean_mag','phot_rp_mean_mag']]
LABEL_COLUMN = 'hpmag'



def extract_label(data_frame):

    return data_frame[LABEL_COLUMN]
from sklearn.linear_model import LinearRegression



def get_base_trainer():

    return LinearRegression()
base_transform = get_cv_model_transform(work_data, extract_label, extract_vars, get_base_trainer, 

        n_runs=3, n_splits=5, max_n_training=None, response_column='base_response' , scale=False)
work_data = base_transform(work_data)
print_evaluation(work_data, LABEL_COLUMN, 'base_response')
def base_residual_transform(data_frame):

    new_frame = data_frame.copy()

    new_frame['base_residual'] = new_frame[LABEL_COLUMN] - new_frame['base_response']

    return new_frame
work_data = base_residual_transform(work_data)
def extract_res_vars(data_frame):

    g_mag = data_frame['phot_g_mean_mag']

    bp_mag = data_frame['phot_bp_mean_mag']

    rp_mag = data_frame['phot_rp_mean_mag']

    btmag = data_frame['btmag']

    hpmag = data_frame['hpmag']

    vmag = data_frame['vmag']

    vtmag = data_frame['vtmag']

    return np.transpose([

        g_mag - bp_mag, 

        rp_mag - g_mag,

        btmag - vmag,

        vmag - vtmag,

        hpmag - vmag,

        hpmag - vtmag,

        btmag - vtmag,

    ])
def extract_res_label(data_frame):

    return data_frame['base_residual']
from sklearn.neural_network import MLPRegressor



def get_res_trainer():

    return MLPRegressor(hidden_layer_sizes=(20, 20), max_iter=300, alpha=0.1, random_state=np.random.randint(1,10000))
res_transform = get_cv_model_transform(work_data, extract_res_label, extract_res_vars, get_res_trainer, 

        n_runs=3, n_splits=3, max_n_training=None, response_column='modeled_residual' , scale=True)
work_data = res_transform(work_data)
print_evaluation(work_data, 'base_residual', 'modeled_residual')
def mag_change_transform(data_frame):

    new_frame = data_frame.copy()

    new_frame['mag_change_estimate'] = new_frame['modeled_residual'] - new_frame['base_residual']

    return new_frame
work_data = mag_change_transform(work_data)
work_data[['source_id', 'parallax', 'l', 'b', 'base_residual', 'mag_change_estimate']].to_csv('hipparcos-gaia-mag-change-estimate.csv', index=False)