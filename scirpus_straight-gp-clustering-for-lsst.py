import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
import gc
import os
import matplotlib.pyplot as plt
import seaborn as sns 
import lightgbm as lgb
import itertools
import pickle, gzip
import glob
from sklearn.preprocessing import StandardScaler
from tsfresh.feature_extraction import extract_features
def photoztodist(data) :
    return ((((((np.log((((data["hostgal_photoz"]) + (np.log((((data["hostgal_photoz"]) + (np.sqrt((np.log((np.maximum(((3.0)), ((((data["hostgal_photoz"]) * 2.0))))))))))))))))) + ((12.99870681762695312)))) + ((1.17613816261291504)))) * (3.0))

fcp = {'fft_coefficient': [{'coeff': 0, 'attr': 'abs'},{'coeff': 1, 'attr': 'abs'}],'kurtosis' : None, 'skewness' : None}
def get_inputs(data, metadata):
    agg_df_ts = None
    agg_df_ts_detected = None
    for i in range(10):
        uniqueids = data.object_id.unique()[i::20]
        df_ts = extract_features(data.loc[data.object_id.isin(uniqueids)], column_id='object_id', column_sort='mjd', column_kind='passband', column_value = 'flux', default_fc_parameters = fcp, n_jobs=4)
        df_ts.index.rename('object_id',inplace=True)
        df_ts.reset_index(drop=False,inplace=True)
        df_ts_detected = extract_features(data.loc[(data.detected==1)&data.object_id.isin(uniqueids)], column_id='object_id', column_sort='mjd', column_kind='passband', column_value = 'flux', default_fc_parameters = fcp, n_jobs=4)
        df_ts_detected.index.rename('object_id',inplace=True)
        df_ts_detected.reset_index(drop=False,inplace=True)
        if(agg_df_ts is None):
            agg_df_ts = df_ts.copy()
            agg_df_ts_detected = df_ts_detected.copy()
        else:
            agg_df_ts = pd.concat([agg_df_ts,df_ts.fillna(0)],sort=False)
            agg_df_ts_detected = pd.concat([agg_df_ts_detected,df_ts_detected.fillna(0)],sort=False)
        del df_ts, df_ts_detected
        gc.collect()
    for d in [0,1]:
        for pb in range(6):
            x = None
            if(d==0):
                x = data[(data.passband==pb)][['object_id','flux']].groupby(['object_id']).flux.mean().reset_index(drop=False)
            else:
                x = data[(data.passband==pb)&(data.detected==1)][['object_id','flux']].groupby(['object_id']).flux.mean().reset_index(drop=False)
            x.columns = ['object_id','flux_d'+str(d)+'_pb'+str(pb)]  
            metadata = metadata.merge(x,on='object_id',how='left')
            del x
            gc.collect()
    
    data['flux_ratio_sq'] = np.power(data['flux'] / data['flux_err'], 2.0)
    data['flux_by_flux_ratio_sq'] = data['flux'] * data['flux_ratio_sq']
    aggs = {
        'mjd': ['min', 'max', 'size'],
        'flux': ['min', 'max', 'mean', 'median', 'std','skew'],
        'flux_err': ['min', 'max', 'mean', 'median', 'std','skew'],
        'flux_ratio_sq':['sum','skew'],
        'flux_by_flux_ratio_sq':['sum','skew'],
    }
    x = data[data.detected==1].groupby(['object_id']).agg(aggs)
    new_columns = [
            k + '_' + agg for k in aggs.keys() for agg in aggs[k]
        ]
    x.columns = new_columns

    x['mjd_diff'] = x['mjd_max'] - x['mjd_min']
    x['flux_diff'] = x['flux_max'] - x['flux_min']
    x['flux_dif2'] = (x['flux_max'] - x['flux_min']) / x['flux_mean']
    x['flux_w_mean'] = x['flux_by_flux_ratio_sq_sum'] / x['flux_ratio_sq_sum']
    x['flux_dif3'] = (x['flux_max'] - x['flux_min']) / x['flux_w_mean']
    del x['mjd_max'], x['mjd_min']
    x.columns = ['detected_'+c for c in x.columns]
    x = x.reset_index(drop=False)
    aggs = {
        'mjd': ['min', 'max', 'size'],
        'flux': ['min', 'max', 'mean', 'median', 'std','skew'],
        'flux_err': ['min', 'max', 'mean', 'median', 'std','skew'],
        'detected': ['mean'],
        'flux_ratio_sq':['sum','skew'],
        'flux_by_flux_ratio_sq':['sum','skew'],
    }

    agg_data = data.groupby(['object_id']).agg(aggs)
    new_columns = [
        k + '_' + agg for k in aggs.keys() for agg in aggs[k]
    ]
    agg_data.columns = new_columns
    agg_data['mjd_diff'] = agg_data['mjd_max'] - agg_data['mjd_min']
    agg_data['flux_diff'] = agg_data['flux_max'] - agg_data['flux_min']
    agg_data['flux_dif2'] = (agg_data['flux_max'] - agg_data['flux_min']) / agg_data['flux_mean']
    agg_data['flux_w_mean'] = agg_data['flux_by_flux_ratio_sq_sum'] / agg_data['flux_ratio_sq_sum']
    agg_data['flux_dif3'] = (agg_data['flux_max'] - agg_data['flux_min']) / agg_data['flux_w_mean']
    del agg_data['mjd_max'], agg_data['mjd_min']
    agg_data.reset_index(drop=False,inplace=True)
    full_data = agg_data.merge(right=metadata, how='outer', on=['object_id'])
    del full_data['hostgal_specz']
    del full_data['ra'], full_data['decl'], full_data['gal_l'],full_data['gal_b']
    del data, metadata
    gc.collect()
    full_data = full_data.merge(x,on=['object_id'],how='left')
    full_data = full_data.merge(agg_df_ts,on=['object_id'],how='left')
    full_data = full_data.merge(agg_df_ts_detected,on=['object_id'],how='left')
    full_data.loc[full_data.distmod.isnull()&(full_data.hostgal_photoz!=0),'distmod'] = photoztodist(full_data.loc[full_data.distmod.isnull()&(full_data.hostgal_photoz!=0)])
    full_data.loc[full_data.distmod.isnull()&(full_data.hostgal_photoz==0),'distmod'] = 0
    return full_data
customscale = np.array([0.16, 1.06, 1.14, 1.92, 1.71, 1.14, 0.71, 0.53, 0.55, 0.5 , 0.48,
                        0.5 , 0.19, 0.2 , 1.73, 0.44, 6.78, 0.91, 0.1 , 1.06, 2.57, 2.51,
                        1.22, 0.1 , 0.45, 0.28, 1.19, 0.16, 1.92, 2.07, 2.03, 2.02, 2.04,
                        2.23, 1.65, 2.59, 2.58, 2.68, 2.81, 2.36, 0.94, 3.09, 1.79, 2.36,
                        2.52, 1.58, 0.54, 0.57, 0.99, 0.73, 0.71, 0.87, 0.56, 2.09, 0.56,
                        6.74, 0.63, 1.57, 1.86, 0.95, 2.52, 0.81, 1.51, 1.2 , 0.87, 0.57,
                        1.99, 1.78, 1.12, 0.69, 1.73, 1.52, 1.09, 0.62, 1.61, 1.37, 1.07,
                        0.62, 1.59, 1.26, 0.97, 0.6 , 1.55, 1.14, 0.81, 0.53, 1.91, 1.47,
                        0.23, 0.16, 2.67, 2.34, 0.33, 0.25, 2.38, 2.55, 0.51, 0.4 , 2.59,
                        2.51, 0.44, 0.35, 3.03, 2.5 , 0.37, 0.27, 2.82, 2.26, 0.27, 0.17])

custommean = np.array([ 4.86e+00, -4.44e+00,  4.87e+00,  1.42e+00,  5.97e-01,  3.51e+00,
                        5.03e-01,  1.09e+00,  4.12e+00,  2.76e+00,  2.48e+00,  2.59e+00,
                        8.87e-01,  9.83e-02,  7.06e+00,  1.97e+00,  9.01e+00,  1.82e+00,
                        6.88e+00,  5.43e+00,  2.95e+00,  3.22e+00,  1.52e+00,  1.00e-02,
                        5.16e-01,  1.54e-01,  3.34e+00,  9.19e-02,  3.00e-01,  7.93e-01,
                        1.34e+00,  1.49e+00,  1.40e+00,  1.18e+00,  2.50e-01,  1.28e+00,
                        2.62e+00,  2.52e+00,  1.76e+00,  6.51e-01,  1.90e+00,  2.19e+00,
                        4.29e+00,  3.37e+00,  3.26e+00,  3.12e+00,  5.48e-02,  1.48e+00,
                        2.44e+00,  2.00e+00,  1.91e+00,  1.33e+00,  3.57e-01,  6.53e+00,
                        4.30e-01,  8.79e+00,  4.04e-01,  3.55e+00,  3.86e+00,  5.18e-01,
                        3.47e+00,  4.59e-01,  3.61e+00,  3.81e+00,  2.59e-01,  1.25e-01,
                        3.35e+00,  3.39e+00,  8.29e-01,  5.41e-01,  4.69e+00,  4.62e+00,
                        1.32e+00,  8.51e-01,  4.88e+00,  4.83e+00,  1.07e+00,  7.17e-01,
                        5.15e+00,  5.19e+00,  7.99e-01,  4.77e-01,  5.37e+00,  5.52e+00,
                        5.02e-01,  1.88e-01,  6.41e-01,  3.55e-01, -5.65e-03,  1.40e-02,
                        2.16e+00,  1.01e+00, -6.87e-03,  2.92e-02,  3.95e+00,  2.12e+00,
                       -2.85e-02,  3.19e-02,  3.70e+00,  1.87e+00, -3.37e-02,  1.06e-02,
                        2.74e+00,  1.40e+00, -3.77e-02,  4.01e-03,  1.36e+00,  7.93e-01,
                       -3.63e-02,  6.69e-03])
meta_train = pd.read_csv('../input/training_set_metadata.csv')
train = pd.read_csv('../input/training_set.csv')
full_train = get_inputs(train,meta_train)
if 'target' in full_train:
    y = full_train['target']
    del full_train['target']

features = full_train.columns[1:]
alldata = full_train[features].copy()
features = alldata.columns
for c in features:
    print(c)
    if(alldata[c].min()<0):
        alldata.loc[~alldata[c].isnull(),c] = np.sign(alldata.loc[~alldata[c].isnull(),c])*np.log1p(np.abs(alldata.loc[~alldata[c].isnull(),c]))
    elif((alldata[c].max()-alldata[c].min())>10):
        alldata.loc[~alldata[c].isnull(),c] = np.log1p(alldata.loc[~alldata[c].isnull(),c])
alldata.fillna(0,inplace=True)
ss = StandardScaler()
ss.scale_=customscale
ss.mean_=custommean
alldata.loc[:,features] = ss.transform(alldata.loc[:,features])
full_train_ss = alldata
full_train_ss.columns = [c.replace('"','_')for c in full_train_ss.columns] #My program doesn't really like non-alpha numeric column names
def GPx(data):
    return ((((data["distmod"]) + (((data["distmod"]) + (((((data["detected_flux_by_flux_ratio_sq_skew"]) + (1.570796))) + (((0.636620) + (data["distmod"]))))))))) +
(((((data["detected_flux_min"]) + (1.0))) + ((((data["flux_min"]) + (1.0))/2.0)))) +
((((((((((0.95365786552429199)) + (data["distmod"]))) + (((data["distmod"]) + ((0.95365786552429199)))))) + ((0.95365786552429199)))) + ((0.95365786552429199)))) +
(np.minimum(((data["3__fft_coefficient__coeff_0__attr__abs__y"])), ((np.minimum(((data["flux_std"])), ((data["4__skewness_x"]))))))) +
(((((((data["detected_flux_min"]) + (data["detected_flux_min"]))) + (data["flux_by_flux_ratio_sq_skew"]))) + (((data["flux_by_flux_ratio_sq_skew"]) + (data["flux_skew"]))))) +
(((((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) + (data["0__fft_coefficient__coeff_1__attr__abs__x"]))/2.0)) + (data["flux_err_min"]))/2.0)) +
(((np.where(data["distmod"] > -1, data["distmod"], data["distmod"] )) + (((np.where(3.141593 > -1, 3.0, data["distmod"] )) + (data["distmod"]))))) +
(((data["distmod"]) + ((((((((data["detected_flux_min"]) + (data["detected_flux_min"]))/2.0)) + (data["distmod"]))) + (np.tanh((data["3__skewness_x"]))))))) +
(np.where(data["distmod"] > -1, (1.74795663356781006), ((np.minimum(((data["distmod"])), ((1.0)))) + (data["detected_flux_by_flux_ratio_sq_sum"])) )) +
((-1.0*((((data["flux_ratio_sq_skew"]) + (((((-1.0*((data["flux_ratio_sq_skew"])))) < (((data["flux_ratio_sq_skew"]) + (data["4__kurtosis_x"]))))*1.))))))) +
(((3.141593) + (((((data["distmod"]) + (data["distmod"]))) + (((3.141593) + (data["distmod"]))))))) +
(np.minimum(((data["flux_d1_pb0"])), ((((data["flux_d1_pb0"]) + (data["0__fft_coefficient__coeff_1__attr__abs__x"])))))) +
(((((data["flux_by_flux_ratio_sq_skew"]) * 2.0)) * 2.0)) +
(np.where(data["distmod"] > -1, ((((((data["flux_ratio_sq_skew"]) - (data["distmod"]))) - (2.0))) - (((data["detected_mjd_diff"]) * 2.0))), data["flux_by_flux_ratio_sq_skew"] )) +
(((((((data["flux_min"]) + (data["distmod"]))) + (data["detected_flux_min"]))) * 2.0)) +
(((np.minimum(((((((data["5__fft_coefficient__coeff_1__attr__abs__y"]) + (data["5__fft_coefficient__coeff_1__attr__abs__y"]))) + (data["5__fft_coefficient__coeff_1__attr__abs__y"])))), ((data["flux_err_mean"])))) + (((data["5__fft_coefficient__coeff_1__attr__abs__y"]) + (data["5__fft_coefficient__coeff_1__attr__abs__x"]))))) +
(((((data["detected_flux_min"]) + (data["2__skewness_x"]))) + (np.maximum(((((data["flux_skew"]) + (((data["flux_skew"]) + (data["4__skewness_x"])))))), ((data["1__kurtosis_x"])))))) +
(((((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) + (data["0__fft_coefficient__coeff_1__attr__abs__y"]))) + (((data["0__fft_coefficient__coeff_1__attr__abs__x"]) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))))) +
(((((data["4__fft_coefficient__coeff_1__attr__abs__y"]) + (((data["detected_mjd_size"]) + (data["detected_mjd_size"]))))) + (((data["detected_mean"]) * 2.0)))) +
(((np.where(data["5__fft_coefficient__coeff_1__attr__abs__y"] > -1, data["4__fft_coefficient__coeff_1__attr__abs__x"], np.where(data["5__fft_coefficient__coeff_1__attr__abs__y"] > -1, data["5__fft_coefficient__coeff_1__attr__abs__y"], np.minimum((((5.75092554092407227))), ((data["5__fft_coefficient__coeff_1__attr__abs__y"]))) ) )) * 2.0)) +
(((((data["distmod"]) + (1.0))) + (((data["flux_d1_pb4"]) + ((((data["distmod"]) + (data["detected_flux_min"]))/2.0)))))) +
(np.where(data["3__fft_coefficient__coeff_1__attr__abs__x"]>0, data["5__fft_coefficient__coeff_1__attr__abs__x"], np.minimum(((((data["detected_mjd_size"]) * 2.0))), ((data["5__fft_coefficient__coeff_1__attr__abs__x"]))) )) +
(((((((data["4__skewness_x"]) + (data["3__skewness_x"]))) + (np.maximum((((((((data["3__skewness_x"]) + (data["2__skewness_x"]))/2.0)) * 2.0))), ((data["3__skewness_x"])))))) * 2.0)) +
(np.minimum(((((data["flux_err_min"]) + (np.where(data["detected_flux_err_min"] > -1, np.tanh((data["0__skewness_y"])), ((data["flux_err_min"]) + (data["flux_err_min"])) ))))), ((data["5__fft_coefficient__coeff_0__attr__abs__y"])))) +
((((((((((data["3__skewness_x"]) * 2.0)) * 2.0)) + (data["4__skewness_x"]))/2.0)) + (((data["4__skewness_x"]) + (data["4__skewness_x"]))))) +
(((data["distmod"]) + (((((data["5__fft_coefficient__coeff_0__attr__abs__y"]) + (((((data["detected_mean"]) + (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) + (data["distmod"]))))) + (data["5__fft_coefficient__coeff_1__attr__abs__x"]))))) +
(((2.0) + (((((data["distmod"]) + (((1.570796) + (data["distmod"]))))) + (data["distmod"]))))) +
(((data["detected_flux_w_mean"]) + (((data["0__skewness_x"]) + (((((data["detected_flux_max"]) + (data["detected_flux_min"]))) + (data["detected_flux_min"]))))))) +
(np.where(data["2__fft_coefficient__coeff_0__attr__abs__x"]>0, ((data["0__fft_coefficient__coeff_0__attr__abs__y"]) * 2.0), ((((data["0__fft_coefficient__coeff_0__attr__abs__y"]) + (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) * 2.0) )) +
(((((data["1__skewness_x"]) + (data["1__skewness_x"]))) * 2.0)) +
(((((data["0__fft_coefficient__coeff_0__attr__abs__y"]) + (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) + (((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (data["detected_flux_std"]))) + (data["0__fft_coefficient__coeff_1__attr__abs__y"]))))) +
((((np.where(data["detected_mjd_size"] > -1, data["detected_mjd_diff"], data["detected_mjd_diff"] )) + (data["detected_mjd_diff"]))/2.0)) +
(((data["detected_mjd_size"]) + (np.minimum(((np.maximum(((data["detected_mjd_size"])), ((data["detected_mjd_size"]))))), ((data["detected_mjd_size"])))))) +
(((((((((((data["distmod"]) + (data["distmod"]))) + (data["detected_mjd_diff"]))) + (data["distmod"]))) + (data["distmod"]))) + (data["distmod"]))) +
(((data["flux_err_min"]) + (((((np.maximum(((data["flux_err_min"])), ((data["flux_err_min"])))) + (data["detected_mean"]))) + (((data["flux_err_min"]) + (data["flux_err_mean"]))))))) +
(((((data["distmod"]) + (((data["distmod"]) * 2.0)))) + (data["distmod"]))) +
(((data["hostgal_photoz"]) + (((data["hostgal_photoz"]) + (((data["hostgal_photoz"]) * 2.0)))))) +
(np.minimum(((data["detected_mjd_size"])), ((((((data["4__fft_coefficient__coeff_1__attr__abs__y"]) * 2.0)) + (data["4__fft_coefficient__coeff_1__attr__abs__y"])))))) +
(((((np.minimum(((data["detected_flux_min"])), ((data["detected_mjd_diff"])))) + (np.minimum(((data["flux_d1_pb0"])), ((data["flux_d1_pb0"])))))) + (data["distmod"]))) +
(((((data["5__fft_coefficient__coeff_0__attr__abs__y"]) - (2.0))) + (np.minimum(((np.minimum(((data["5__fft_coefficient__coeff_0__attr__abs__y"])), ((np.minimum(((data["flux_std"])), ((data["flux_err_mean"])))))))), ((data["flux_err_std"])))))) +
(((((data["detected_flux_mean"]) + ((((((((data["detected_flux_min"]) + (((data["detected_flux_min"]) + (data["detected_flux_min"]))))) + (data["detected_flux_min"]))/2.0)) * 2.0)))) * 2.0)) +
(((data["flux_err_mean"]) + (np.minimum(((np.minimum(((((((data["flux_err_std"]) + (data["4__skewness_x"]))) * 2.0))), ((data["2__skewness_y"]))))), ((data["3__kurtosis_y"])))))) +
((((data["hostgal_photoz"]) + (data["hostgal_photoz"]))/2.0)) +
(((data["detected_mjd_diff"]) + (np.tanh((data["detected_mjd_diff"]))))) +
((((((((np.where(data["distmod"] > -1, 1.570796, ((data["distmod"]) * 2.0) )) * 2.0)) + (1.0))/2.0)) * 2.0)) +
(((((np.where(data["detected_mjd_size"]>0, data["detected_mjd_size"], data["detected_mjd_size"] )) - (data["0__fft_coefficient__coeff_1__attr__abs__y"]))) * 2.0)) +
(np.minimum(((np.minimum(((data["1__fft_coefficient__coeff_1__attr__abs__y"])), ((data["3__skewness_y"]))))), ((data["flux_err_std"])))) +
(((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) + ((((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) + (data["distmod"]))/2.0)) + (np.minimum(((data["flux_mean"])), ((data["detected_flux_min"])))))))) - (data["detected_flux_std"]))) +
(np.minimum(((data["4__fft_coefficient__coeff_1__attr__abs__x"])), ((((data["4__skewness_x"]) + (((data["detected_flux_min"]) + (np.minimum(((data["detected_flux_min"])), ((data["detected_flux_min"]))))))))))) +
(((((((data["detected_mjd_diff"]) + (data["flux_err_min"]))) * 2.0)) + (((data["detected_mjd_diff"]) + (data["detected_mjd_diff"]))))) +
(np.where(data["detected_mjd_diff"] > -1, data["detected_mjd_diff"], data["detected_mjd_diff"] )) +
(np.minimum(((data["3__skewness_y"])), ((((((data["2__skewness_x"]) - (data["0__fft_coefficient__coeff_1__attr__abs__y"]))) + (data["3__skewness_y"])))))) +
((((((data["detected_mjd_diff"]) + ((((((data["hostgal_photoz"]) + (data["distmod"]))) + (data["hostgal_photoz"]))/2.0)))/2.0)) + (np.minimum(((data["hostgal_photoz"])), ((data["hostgal_photoz"])))))) +
(((((data["detected_mjd_diff"]) + (data["detected_flux_min"]))) * 2.0)) +
(np.minimum(((((np.tanh((data["flux_err_min"]))) * 2.0))), ((data["flux_err_min"])))) +
(((((data["detected_flux_min"]) + (data["3__fft_coefficient__coeff_1__attr__abs__x"]))) + (data["2__skewness_x"]))) +
(np.where(data["detected_mjd_diff"] > -1, data["detected_mjd_diff"], ((data["detected_mjd_diff"]) * 2.0) )) +
(np.where(data["0__fft_coefficient__coeff_0__attr__abs__x"] > -1, data["5__fft_coefficient__coeff_1__attr__abs__x"], ((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - (data["0__fft_coefficient__coeff_1__attr__abs__x"])) )) +
(np.tanh((((np.tanh((data["detected_mjd_diff"]))) / 2.0)))) +
(((data["2__skewness_x"]) + (((data["flux_ratio_sq_skew"]) + (data["2__skewness_x"]))))) +
(((((((((data["detected_flux_min"]) + (data["hostgal_photoz"]))) - (data["3__fft_coefficient__coeff_0__attr__abs__y"]))) + (data["flux_d0_pb0"]))) * 2.0)) +
(((data["flux_d0_pb0"]) - (np.where(data["flux_d0_pb0"]>0, data["3__fft_coefficient__coeff_1__attr__abs__y"], data["3__fft_coefficient__coeff_1__attr__abs__y"] )))) +
(np.where(((data["3__kurtosis_x"]) + (data["3__kurtosis_x"]))<0, ((np.minimum(((0.318310)), ((data["3__kurtosis_x"])))) - (data["1__fft_coefficient__coeff_1__attr__abs__x"])), data["3__kurtosis_x"] )) +
(((((data["flux_ratio_sq_skew"]) + ((((data["flux_by_flux_ratio_sq_skew"]) > (data["flux_ratio_sq_skew"]))*1.)))) + (((data["flux_by_flux_ratio_sq_skew"]) + (((data["flux_ratio_sq_skew"]) + (data["flux_ratio_sq_skew"]))))))) +
(np.where(((np.where(data["0__fft_coefficient__coeff_0__attr__abs__y"]>0, data["2__skewness_x"], data["0__fft_coefficient__coeff_0__attr__abs__y"] )) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))>0, data["0__fft_coefficient__coeff_0__attr__abs__y"], data["flux_by_flux_ratio_sq_skew"] )) +
((((((((data["flux_mean"]) + (((data["hostgal_photoz"]) * 2.0)))/2.0)) + (data["distmod"]))) * 2.0)) +
(np.where(data["0__kurtosis_y"] > -1, data["3__skewness_x"], ((data["flux_d1_pb5"]) + (data["4__fft_coefficient__coeff_0__attr__abs__x"])) )) +
(((data["flux_d0_pb4"]) - (((((data["2__kurtosis_x"]) - (data["flux_d0_pb2"]))) * (data["distmod"]))))) +
(((data["0__fft_coefficient__coeff_1__attr__abs__x"]) - ((((np.where(data["0__fft_coefficient__coeff_1__attr__abs__x"] > -1, data["5__fft_coefficient__coeff_0__attr__abs__y"], data["4__fft_coefficient__coeff_1__attr__abs__x"] )) + (data["3__fft_coefficient__coeff_1__attr__abs__x"]))/2.0)))) +
(((((data["flux_by_flux_ratio_sq_skew"]) - (data["flux_by_flux_ratio_sq_skew"]))) + (((((data["flux_ratio_sq_skew"]) - (data["detected_flux_by_flux_ratio_sq_skew"]))) + (data["flux_by_flux_ratio_sq_skew"]))))) +
(((((((data["4__kurtosis_x"]) + (data["4__kurtosis_x"]))) + (data["detected_flux_err_median"]))) + (((data["4__kurtosis_x"]) - (data["0__skewness_x"]))))) +
(((((((data["detected_flux_min"]) + (((data["distmod"]) + (((data["flux_d0_pb0"]) + (data["distmod"]))))))) + (data["distmod"]))) + (data["detected_flux_min"]))) +
(((np.where(data["distmod"] > -1, (-1.0*((((data["hostgal_photoz"]) * 2.0)))), ((data["hostgal_photoz"]) - ((-1.0*((data["distmod"]))))) )) - (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) +
(((np.minimum(((data["detected_mjd_diff"])), ((data["2__kurtosis_y"])))) + (((((data["detected_mjd_diff"]) - (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) + (data["3__kurtosis_x"]))))) +
(np.minimum(((np.minimum(((data["flux_err_min"])), ((((data["detected_flux_max"]) * (np.minimum(((data["flux_err_min"])), ((np.minimum(((data["detected_flux_min"])), ((data["detected_flux_min"]))))))))))))), ((data["flux_err_min"])))) +
(((((data["1__skewness_x"]) + (data["flux_err_mean"]))) + (data["flux_d0_pb0"]))) +
(np.minimum(((data["5__skewness_y"])), ((np.minimum(((np.minimum(((data["3__kurtosis_y"])), ((data["5__skewness_y"]))))), ((np.minimum(((data["detected_flux_err_mean"])), ((data["3__kurtosis_y"])))))))))) +
(((data["1__skewness_x"]) + (((((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) + ((-1.0*(((-1.0*((data["1__skewness_x"]))))))))) + (data["1__skewness_x"]))) + (data["1__skewness_x"]))))) +
(((((((data["detected_flux_median"]) + ((((data["detected_flux_err_min"]) + (data["flux_err_std"]))/2.0)))/2.0)) + (data["1__skewness_x"]))/2.0)) +
(((data["distmod"]) + (((data["distmod"]) + (((1.0) + (2.0))))))) +
(((((np.tanh((0.318310))) - (data["detected_mjd_diff"]))) - (data["detected_mjd_diff"]))) +
(((((((data["flux_d0_pb3"]) - (((data["detected_mean"]) - (data["2__fft_coefficient__coeff_1__attr__abs__x"]))))) - (data["flux_std"]))) - (data["flux_d0_pb0"]))) +
(((data["detected_mean"]) + (data["hostgal_photoz"]))) +
(np.where(data["distmod"] > -1, ((((((data["flux_by_flux_ratio_sq_skew"]) - (data["flux_d1_pb0"]))) - (data["distmod"]))) - (-1.0)), data["distmod"] )) +
((((((((data["1__skewness_x"]) + (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) + (((((data["detected_flux_max"]) + (data["2__skewness_x"]))) + (data["4__skewness_x"]))))) + (data["2__skewness_x"]))/2.0)) +
(((((((((3.0)) < (data["flux_d0_pb5"]))*1.)) * 2.0)) - (data["distmod"]))) +
(((((((((data["detected_mjd_diff"]) - (data["3__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["3__fft_coefficient__coeff_1__attr__abs__x"]))) + (((data["detected_mjd_diff"]) - (data["flux_ratio_sq_sum"]))))) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) +
(((((((data["5__kurtosis_y"]) + (data["ddf"]))) - (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) + (((data["detected_mjd_diff"]) + (data["detected_mjd_diff"]))))) +
(((((((((data["2__fft_coefficient__coeff_1__attr__abs__x"]) + (data["flux_d0_pb2"]))) + (((data["flux_d0_pb4"]) - (data["detected_flux_max"]))))) - (data["detected_flux_max"]))) * 2.0)) +
(((np.where(data["3__fft_coefficient__coeff_1__attr__abs__x"] > -1, data["5__kurtosis_x"], ((((data["5__kurtosis_x"]) - (data["5__kurtosis_x"]))) - (data["5__kurtosis_x"])) )) - (data["0__skewness_x"]))) +
(((((((((np.where(((((data["flux_skew"]) / 2.0)) * 2.0)>0, data["4__skewness_x"], data["4__fft_coefficient__coeff_0__attr__abs__y"] )) + (data["1__skewness_y"]))/2.0)) * 2.0)) + (data["flux_skew"]))/2.0)) +
(((np.where(0.636620 > -1, ((np.where(data["4__fft_coefficient__coeff_1__attr__abs__y"] > -1, data["4__fft_coefficient__coeff_1__attr__abs__x"], data["4__fft_coefficient__coeff_1__attr__abs__x"] )) + (data["distmod"])), data["4__fft_coefficient__coeff_1__attr__abs__y"] )) - (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) +
(((((data["detected_flux_by_flux_ratio_sq_sum"]) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["hostgal_photoz_err"]))) +
(((((data["detected_flux_min"]) + (((((np.minimum(((((data["flux_median"]) * 2.0))), ((data["detected_flux_min"])))) * 2.0)) * 2.0)))) * 2.0)) +
(np.minimum(((((data["flux_err_min"]) - (data["distmod"])))), ((((data["flux_err_min"]) + (((data["flux_err_min"]) - (data["flux_err_std"])))))))) +
(np.where(data["detected_mjd_diff"] > -1, np.where(data["hostgal_photoz"] > -1, ((((data["distmod"]) - (data["hostgal_photoz"]))) - (data["distmod"])), data["distmod"] ), data["hostgal_photoz"] )) +
(np.minimum(((((data["detected_mjd_diff"]) * (data["2__skewness_x"])))), ((((data["detected_mjd_diff"]) + (data["1__fft_coefficient__coeff_0__attr__abs__x"])))))) +
(np.where(((np.maximum(((data["hostgal_photoz"])), ((data["detected_mjd_diff"])))) * 2.0)>0, data["hostgal_photoz_err"], (((data["flux_min"]) + (data["flux_min"]))/2.0) )) +
(((np.minimum(((np.tanh((((np.tanh(((-1.0*((((data["mjd_diff"]) * 2.0))))))) / 2.0))))), (((-1.0*((data["4__fft_coefficient__coeff_0__attr__abs__x"]))))))) * (data["3__skewness_x"]))) +
(((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0)) - (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) +
((-1.0*((np.where(np.where(data["0__fft_coefficient__coeff_0__attr__abs__y"] > -1, data["0__fft_coefficient__coeff_0__attr__abs__y"], data["0__fft_coefficient__coeff_0__attr__abs__y"] ) > -1, data["0__fft_coefficient__coeff_0__attr__abs__y"], data["0__fft_coefficient__coeff_0__attr__abs__y"] ))))) +
((((((((data["detected_flux_err_mean"]) + ((((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) + (data["detected_flux_err_mean"]))/2.0)) - (data["0__fft_coefficient__coeff_1__attr__abs__x"]))))/2.0)) - (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["0__skewness_x"]))) +
(np.minimum(((data["3__skewness_x"])), ((np.tanh((data["hostgal_photoz"])))))) +
(((np.where(data["distmod"] > -1, ((data["flux_d0_pb4"]) + (np.tanh((data["flux_skew"])))), ((data["4__fft_coefficient__coeff_1__attr__abs__x"]) - (data["flux_skew"])) )) + (data["flux_d0_pb5"]))) +
(np.where(data["flux_d1_pb1"]<0, np.where(data["1__fft_coefficient__coeff_0__attr__abs__y"]<0, data["5__fft_coefficient__coeff_0__attr__abs__y"], data["1__fft_coefficient__coeff_0__attr__abs__y"] ), ((data["flux_d0_pb0"]) - (data["2__fft_coefficient__coeff_1__attr__abs__x"])) )) +
((((((((data["detected_flux_err_min"]) - (((data["detected_flux_err_max"]) + (data["distmod"]))))) + (((data["flux_err_min"]) - (data["detected_flux_err_median"]))))/2.0)) * 2.0)) +
(((np.minimum(((data["5__fft_coefficient__coeff_1__attr__abs__x"])), ((data["5__skewness_x"])))) + (((((((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (data["detected_flux_diff"]))) * 2.0)) - (data["detected_flux_diff"]))) * 2.0)))) +
(((((((((((((data["distmod"]) + (1.570796))) * 2.0)) + (1.570796))) * 2.0)) * 2.0)) * 2.0)) +
(((((data["detected_mjd_diff"]) + (((1.570796) + (data["detected_mjd_diff"]))))) + (((0.318310) - (data["flux_max"]))))) +
(((((data["4__fft_coefficient__coeff_1__attr__abs__x"]) - (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) + (((data["detected_mjd_diff"]) - (data["0__fft_coefficient__coeff_0__attr__abs__y"]))))) +
(((np.where(data["flux_d0_pb0"]>0, ((data["flux_d0_pb0"]) - (data["hostgal_photoz"])), ((data["mjd_diff"]) - (data["hostgal_photoz"])) )) - (data["5__fft_coefficient__coeff_1__attr__abs__y"]))) +
(np.where(((data["4__fft_coefficient__coeff_1__attr__abs__x"]) + (data["detected_flux_w_mean"])) > -1, ((((data["4__kurtosis_x"]) * (data["2__fft_coefficient__coeff_0__attr__abs__y"]))) / 2.0), np.tanh((data["2__fft_coefficient__coeff_0__attr__abs__x"])) )) +
(((((np.maximum(((data["2__skewness_x"])), ((data["2__skewness_x"])))) + (((data["1__skewness_x"]) + (((data["flux_std"]) - (data["detected_flux_ratio_sq_skew"]))))))) + (data["2__skewness_x"]))) +
(((np.minimum(((((data["flux_d0_pb5"]) - (data["detected_flux_std"])))), ((((data["5__skewness_x"]) + (data["detected_flux_min"])))))) + (((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (data["detected_flux_max"]))))) +
(((((data["1__fft_coefficient__coeff_0__attr__abs__x"]) - (np.maximum(((data["hostgal_photoz"])), (((-1.0*((data["1__fft_coefficient__coeff_0__attr__abs__x"]))))))))) - (np.maximum(((((data["hostgal_photoz"]) / 2.0))), ((data["5__fft_coefficient__coeff_1__attr__abs__x"])))))) +
(((((data["5__kurtosis_x"]) + (((data["5__kurtosis_x"]) - (data["2__skewness_x"]))))) - (data["5__kurtosis_x"]))) +
(((np.where(data["2__kurtosis_y"]>0, np.minimum(((data["1__skewness_y"])), ((data["3__kurtosis_x"]))), data["4__kurtosis_x"] )) - (data["5__kurtosis_y"]))) +
(((data["flux_err_min"]) - (((data["flux_err_median"]) - (((data["flux_err_min"]) - (((data["flux_err_median"]) - (data["flux_err_median"]))))))))) +
(((((((((data["distmod"]) - (data["hostgal_photoz"]))) - (data["hostgal_photoz"]))) + ((((data["detected_mjd_diff"]) + (data["distmod"]))/2.0)))) * 2.0)) +
(((np.maximum(((data["detected_flux_by_flux_ratio_sq_sum"])), ((data["mjd_diff"])))) * (data["mjd_diff"]))) +
(((np.minimum(((data["2__kurtosis_y"])), ((((data["2__kurtosis_y"]) + (data["4__skewness_y"])))))) + (((data["3__skewness_y"]) + (data["2__kurtosis_y"]))))) +
(((((((data["flux_d0_pb0"]) + (data["detected_flux_err_min"]))) + (((data["mjd_diff"]) + (data["flux_d0_pb0"]))))) + (data["detected_flux_min"]))) +
(((((((((data["flux_skew"]) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) * 2.0)) - (data["flux_median"]))) - (((data["flux_median"]) - (data["ddf"]))))) +
(np.where(np.tanh(((((data["2__kurtosis_x"]) < (data["2__kurtosis_x"]))*1.))) > -1, data["2__kurtosis_x"], data["2__kurtosis_x"] )) +
(np.minimum(((data["5__kurtosis_y"])), ((data["4__kurtosis_y"])))) +
(((np.where(data["5__fft_coefficient__coeff_1__attr__abs__y"]>0, data["flux_d1_pb4"], (-1.0*((((data["flux_d0_pb2"]) * (np.tanh((data["flux_dif2"]))))))) )) + (data["3__fft_coefficient__coeff_0__attr__abs__x"]))) +
(((np.minimum(((np.minimum(((data["flux_skew"])), ((data["4__fft_coefficient__coeff_1__attr__abs__y"]))))), ((np.where(data["hostgal_photoz"] > -1, data["hostgal_photoz"], ((data["flux_skew"]) * 2.0) ))))) * 2.0)) +
((((((np.where(data["detected_flux_min"]>0, (((((data["flux_std"]) + (data["5__fft_coefficient__coeff_0__attr__abs__y"]))/2.0)) * 2.0), data["4__fft_coefficient__coeff_1__attr__abs__y"] )) + (data["1__skewness_y"]))/2.0)) + (data["4__fft_coefficient__coeff_1__attr__abs__y"]))) +
(((((data["2__skewness_y"]) + (np.where(data["2__skewness_y"] > -1, data["2__skewness_y"], data["2__skewness_y"] )))) + (np.where(data["2__skewness_y"] > -1, data["2__skewness_y"], data["2__skewness_y"] )))) +
(np.where(data["5__skewness_y"]<0, data["flux_err_min"], (((((((((data["5__kurtosis_x"]) * (data["0__skewness_y"]))) - (data["0__skewness_y"]))) + (data["4__skewness_x"]))/2.0)) * 2.0) )) +
(((data["5__fft_coefficient__coeff_1__attr__abs__y"]) - (np.where(data["1__fft_coefficient__coeff_0__attr__abs__y"]>0, np.where(data["1__fft_coefficient__coeff_0__attr__abs__y"] > -1, data["1__fft_coefficient__coeff_0__attr__abs__y"], data["1__fft_coefficient__coeff_0__attr__abs__y"] ), data["detected_flux_ratio_sq_sum"] )))) +
(np.maximum(((data["flux_dif3"])), ((((data["hostgal_photoz_err"]) + (data["1__skewness_y"])))))) +
(np.where(data["detected_flux_median"]<0, np.where(data["4__fft_coefficient__coeff_0__attr__abs__x"]<0, np.where(data["hostgal_photoz"]<0, data["hostgal_photoz"], data["flux_skew"] ), data["hostgal_photoz"] ), data["hostgal_photoz"] )) +
(((((np.maximum(((((data["distmod"]) + (((data["distmod"]) + (data["distmod"])))))), ((data["flux_d0_pb1"])))) + (data["distmod"]))) + (data["distmod"]))) +
(np.where(((data["flux_d0_pb1"]) - (np.tanh((data["0__fft_coefficient__coeff_1__attr__abs__x"]))))<0, data["mwebv"], data["distmod"] )) +
(((data["flux_d0_pb1"]) + (((((((data["flux_d1_pb0"]) + (data["flux_err_std"]))) + (((data["flux_err_std"]) + (data["flux_d0_pb1"]))))) + (data["mjd_diff"]))))) +
((((data["0__skewness_y"]) + (((data["distmod"]) - ((((((data["2__kurtosis_x"]) - (data["flux_d0_pb5"]))) > (((data["4__skewness_y"]) / 2.0)))*1.)))))/2.0)) +
(((np.where(data["0__kurtosis_x"] > -1, data["0__kurtosis_x"], ((data["detected_flux_err_min"]) - (data["detected_flux_std"])) )) - (data["detected_flux_std"]))) +
(((((data["hostgal_photoz_err"]) - (data["detected_mean"]))) + (((((data["flux_d0_pb5"]) + (data["hostgal_photoz_err"]))) - (data["hostgal_photoz"]))))) +
(((((np.where(data["hostgal_photoz"] > -1, data["hostgal_photoz_err"], ((data["hostgal_photoz"]) * 2.0) )) - (np.tanh((data["hostgal_photoz"]))))) * 2.0)) +
(((((data["hostgal_photoz_err"]) * (data["1__kurtosis_y"]))) - (data["detected_flux_max"]))) +
(((np.where(((((data["distmod"]) * 2.0)) + (data["distmod"])) > -1, 1.570796, data["distmod"] )) * 2.0)) +
(((((data["mjd_diff"]) * 2.0)) + (np.where(data["1__kurtosis_y"]>0, ((data["0__skewness_x"]) * 2.0), ((data["detected_flux_min"]) + (((data["1__kurtosis_y"]) * 2.0))) )))) +
(((((((data["1__skewness_x"]) - (data["flux_d1_pb1"]))) + (((data["ddf"]) - (((data["flux_d0_pb1"]) * 2.0)))))) - (data["flux_d0_pb1"]))) +
(np.where((((data["detected_mjd_diff"]) + (data["2__fft_coefficient__coeff_0__attr__abs__y"]))/2.0) > -1, ((((data["detected_mjd_diff"]) + (data["3__kurtosis_y"]))) - (data["2__fft_coefficient__coeff_0__attr__abs__y"])), data["detected_mjd_diff"] )) +
(np.where(data["3__fft_coefficient__coeff_0__attr__abs__y"] > -1, (-1.0*((data["detected_flux_ratio_sq_skew"]))), (-1.0*((((data["flux_err_std"]) + (data["4__kurtosis_x"]))))) )) +
(((((((data["detected_flux_ratio_sq_skew"]) - (data["detected_flux_ratio_sq_skew"]))) - (data["0__fft_coefficient__coeff_0__attr__abs__x"]))) - (data["detected_flux_ratio_sq_skew"]))) +
(((np.where(np.where(data["detected_flux_mean"]<0, data["flux_diff"], data["flux_diff"] ) > -1, data["flux_diff"], np.where(data["flux_dif2"] > -1, data["flux_dif3"], data["flux_diff"] ) )) * 2.0)) +
(((((((((np.where(data["flux_by_flux_ratio_sq_skew"]<0, data["detected_flux_median"], data["flux_diff"] )) - (data["flux_median"]))) * 2.0)) - (data["2__fft_coefficient__coeff_0__attr__abs__y"]))) - (data["flux_median"]))) +
(np.where(((data["distmod"]) - ((-1.0*((data["detected_mjd_diff"]))))) > -1, ((((data["distmod"]) - (data["hostgal_photoz"]))) - (data["hostgal_photoz"])), data["distmod"] )) +
(((np.where(data["distmod"]<0, (((((data["distmod"]) + (data["distmod"]))/2.0)) * 2.0), data["distmod"] )) + (((data["4__fft_coefficient__coeff_1__attr__abs__y"]) + (data["distmod"]))))) +
(((np.maximum(((((((data["flux_d0_pb4"]) / 2.0)) / 2.0))), ((((((data["1__fft_coefficient__coeff_0__attr__abs__x"]) - (data["flux_ratio_sq_skew"]))) / 2.0))))) - (data["detected_mjd_diff"]))) +
(((data["flux_d0_pb5"]) + (((((((data["1__kurtosis_x"]) + (((data["1__kurtosis_x"]) - (data["detected_mean"]))))) - (data["1__kurtosis_x"]))) + (data["flux_w_mean"]))))) +
(((data["hostgal_photoz_err"]) + (((np.where((((data["flux_d1_pb0"]) < (data["detected_flux_ratio_sq_skew"]))*1.) > -1, (-1.0*((data["detected_flux_ratio_sq_skew"]))), data["hostgal_photoz_err"] )) - (data["detected_flux_ratio_sq_skew"]))))) +
((((data["5__kurtosis_x"]) + (np.tanh((data["1__skewness_y"]))))/2.0)) +
(np.where(((data["detected_flux_min"]) + (data["0__skewness_x"])) > -1, ((data["4__fft_coefficient__coeff_1__attr__abs__x"]) + (data["detected_flux_min"])), data["detected_flux_min"] )) +
(((((((data["4__fft_coefficient__coeff_1__attr__abs__x"]) - (data["flux_d0_pb0"]))) + (((data["ddf"]) + (data["3__fft_coefficient__coeff_1__attr__abs__y"]))))) - (data["flux_d0_pb1"]))) +
(((((((((data["flux_skew"]) < (np.tanh((data["3__skewness_x"]))))*1.)) + (((data["detected_flux_skew"]) - (data["flux_std"]))))/2.0)) - (data["detected_flux_max"]))) +
(((((((data["flux_err_min"]) - (data["flux_err_mean"]))) + (data["flux_err_min"]))) + (data["flux_err_min"]))) +
(((((((((data["hostgal_photoz_err"]) + (data["flux_err_min"]))) + ((((data["hostgal_photoz_err"]) + (data["flux_err_min"]))/2.0)))) + (data["flux_err_min"]))) - (data["5__fft_coefficient__coeff_1__attr__abs__y"]))) +
(((np.where(data["flux_d0_pb1"]<0, ((data["4__kurtosis_x"]) - (data["detected_flux_std"])), data["detected_flux_std"] )) - (data["flux_d0_pb1"]))) +
(np.where(data["2__fft_coefficient__coeff_0__attr__abs__y"]<0, np.where(data["detected_mean"] > -1, data["flux_err_max"], data["5__kurtosis_x"] ), np.minimum(((((((data["5__kurtosis_x"]) * 2.0)) * 2.0))), ((data["5__kurtosis_x"]))) )) +
(((data["hostgal_photoz_err"]) - (data["distmod"]))) +
(((np.minimum(((data["1__skewness_y"])), ((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - ((-1.0*((data["4__fft_coefficient__coeff_0__attr__abs__y"]))))))))) - (data["4__fft_coefficient__coeff_1__attr__abs__y"]))) +
(np.where(data["5__fft_coefficient__coeff_1__attr__abs__x"]<0, ((data["flux_d1_pb1"]) * 2.0), ((data["5__fft_coefficient__coeff_0__attr__abs__x"]) - (data["flux_d0_pb3"])) )) +
(((np.where(data["detected_mjd_size"]<0, np.where(data["4__fft_coefficient__coeff_0__attr__abs__x"]<0, data["4__fft_coefficient__coeff_0__attr__abs__x"], data["flux_dif2"] ), data["flux_dif2"] )) * 2.0)) +
(np.where(data["hostgal_photoz"]>0, ((np.where(data["flux_d0_pb5"]>0, ((data["hostgal_photoz_err"]) - (data["distmod"])), data["hostgal_photoz"] )) * 2.0), data["flux_d0_pb5"] )) +
(((((((np.where(data["distmod"]>0, np.where(data["detected_flux_std"]<0, data["flux_mean"], data["hostgal_photoz_err"] ), data["flux_mean"] )) - (data["detected_flux_std"]))) * 2.0)) * 2.0)) +
(((((((((((data["3__fft_coefficient__coeff_1__attr__abs__x"]) + (data["detected_flux_max"]))) * (data["5__kurtosis_x"]))) - (data["detected_flux_ratio_sq_sum"]))) - (data["distmod"]))) - (data["detected_flux_ratio_sq_sum"]))) +
(np.where((((data["1__fft_coefficient__coeff_0__attr__abs__x"]) < (data["flux_max"]))*1.)>0, np.where(data["flux_dif2"]>0, data["flux_max"], np.tanh((data["flux_dif3"])) ), data["flux_dif3"] )) +
(((data["0__fft_coefficient__coeff_1__attr__abs__y"]) * (data["1__kurtosis_y"]))) +
(((((((0.0) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) - (((data["detected_mjd_diff"]) - (data["flux_err_skew"]))))) * 2.0)) +
(((((((((data["flux_err_max"]) * 2.0)) + (data["hostgal_photoz_err"]))) + (data["flux_err_mean"]))) + (((data["flux_err_max"]) - (data["4__fft_coefficient__coeff_0__attr__abs__x"]))))) +
(np.minimum(((np.minimum(((np.minimum(((data["1__skewness_x"])), ((data["flux_d1_pb0"]))))), ((data["flux_max"]))))), ((data["3__fft_coefficient__coeff_1__attr__abs__y"])))) +
(((((data["3__skewness_y"]) - (((data["1__kurtosis_x"]) * 2.0)))) + (((data["1__skewness_y"]) - (data["1__kurtosis_x"]))))) +
(((((((data["distmod"]) > (data["flux_d0_pb3"]))*1.)) < (data["distmod"]))*1.)) +
(((np.where(data["detected_flux_max"]>0, ((((data["detected_flux_max"]) * (data["flux_median"]))) - (data["detected_mjd_diff"])), (-1.0*((data["flux_ratio_sq_skew"]))) )) * 2.0)) +
(((((data["flux_skew"]) - (data["detected_flux_max"]))) + (((data["flux_skew"]) * (((data["flux_err_min"]) + (data["flux_err_min"]))))))) +
(((((data["2__fft_coefficient__coeff_1__attr__abs__x"]) + (data["3__fft_coefficient__coeff_0__attr__abs__y"]))) - (data["flux_skew"]))) +
((((((((data["0__fft_coefficient__coeff_0__attr__abs__y"]) > (data["0__skewness_x"]))*1.)) * (np.minimum(((data["detected_flux_std"])), ((data["flux_err_median"])))))) + (data["5__kurtosis_x"]))) +
(((data["detected_flux_max"]) + (np.where((((data["detected_flux_diff"]) + (data["distmod"]))/2.0)>0, ((((data["detected_flux_diff"]) * 2.0)) - (data["1__fft_coefficient__coeff_1__attr__abs__y"])), data["distmod"] )))) +
((((((data["2__fft_coefficient__coeff_1__attr__abs__x"]) * (data["flux_w_mean"]))) + (((((data["2__fft_coefficient__coeff_1__attr__abs__x"]) * (data["detected_flux_w_mean"]))) - ((((data["flux_w_mean"]) + (data["0__skewness_x"]))/2.0)))))/2.0)) +
(((((data["5__fft_coefficient__coeff_1__attr__abs__y"]) - (data["1__fft_coefficient__coeff_0__attr__abs__x"]))) - (((((data["1__fft_coefficient__coeff_0__attr__abs__x"]) + (((data["1__fft_coefficient__coeff_0__attr__abs__x"]) / 2.0)))) / 2.0)))) +
(((((((((((data["flux_d0_pb0"]) * (((data["ddf"]) - (data["detected_flux_ratio_sq_skew"]))))) - (data["detected_flux_std"]))) * 2.0)) * 2.0)) - (data["detected_flux_std"]))) +
(((data["flux_err_max"]) * (data["detected_flux_by_flux_ratio_sq_skew"]))) +
(np.where(np.tanh((data["flux_ratio_sq_skew"]))>0, ((data["flux_skew"]) * 2.0), data["distmod"] )) +
(((((data["distmod"]) * ((-1.0*((((data["flux_d0_pb4"]) * 2.0))))))) - (data["2__fft_coefficient__coeff_1__attr__abs__x"]))) +
((-1.0*((((((data["distmod"]) * (data["detected_flux_std"]))) + (((((data["detected_mjd_diff"]) + (data["4__fft_coefficient__coeff_1__attr__abs__y"]))) * (data["distmod"])))))))) +
(((((((data["flux_d1_pb5"]) * (0.0))) * 2.0)) * 2.0)) +
(((data["flux_skew"]) * (((data["flux_max"]) + (data["0__skewness_x"]))))) +
(((np.where(data["3__fft_coefficient__coeff_1__attr__abs__x"]>0, data["0__fft_coefficient__coeff_1__attr__abs__y"], ((((((data["2__fft_coefficient__coeff_0__attr__abs__y"]) * 2.0)) - (data["2__fft_coefficient__coeff_0__attr__abs__y"]))) - (data["detected_flux_dif3"])) )) - (data["2__fft_coefficient__coeff_0__attr__abs__y"]))) +
(np.tanh((((data["detected_flux_err_min"]) * 2.0)))) +
(((np.where(data["distmod"]>0, data["flux_ratio_sq_skew"], ((np.where(data["distmod"] > -1, data["0__kurtosis_x"], data["detected_flux_diff"] )) - (data["flux_ratio_sq_skew"])) )) - (data["detected_flux_diff"]))) +
(((np.tanh((data["detected_mjd_diff"]))) - (((data["detected_mjd_diff"]) - (((((data["2__skewness_x"]) / 2.0)) - (data["detected_mjd_diff"]))))))) +
((-1.0*(((((((np.tanh((data["detected_mjd_diff"]))) < (data["flux_median"]))*1.)) * 2.0))))) +
(((np.where(data["hostgal_photoz_err"] > -1, data["hostgal_photoz_err"], data["hostgal_photoz_err"] )) + (data["flux_err_min"]))) +
(((data["1__fft_coefficient__coeff_1__attr__abs__y"]) + (((data["detected_flux_by_flux_ratio_sq_skew"]) + (((((data["detected_flux_by_flux_ratio_sq_skew"]) + (data["1__skewness_y"]))) + (data["detected_flux_by_flux_ratio_sq_skew"]))))))) +
((((data["0__fft_coefficient__coeff_0__attr__abs__x"]) + (data["distmod"]))/2.0)) +
((((((((np.minimum(((data["detected_flux_by_flux_ratio_sq_skew"])), ((data["detected_mjd_size"])))) / 2.0)) + (data["1__fft_coefficient__coeff_1__attr__abs__x"]))/2.0)) / 2.0)) +
(((np.minimum(((data["0__fft_coefficient__coeff_0__attr__abs__y"])), ((np.minimum(((data["2__skewness_x"])), (((((-1.0*((3.141593)))) / 2.0)))))))) - (data["distmod"]))) +
((-1.0*((np.where((((np.tanh((0.636620))) < (data["4__fft_coefficient__coeff_1__attr__abs__x"]))*1.)<0, data["flux_dif2"], (-1.0*((data["5__skewness_x"]))) ))))) +
(np.where(data["hostgal_photoz"] > -1, ((((data["5__fft_coefficient__coeff_1__attr__abs__y"]) - (data["detected_flux_ratio_sq_sum"]))) - (data["hostgal_photoz"])), (-1.0*((data["5__fft_coefficient__coeff_1__attr__abs__y"]))) )) +
(((np.where(data["distmod"] > -1, ((((data["distmod"]) * (((data["hostgal_photoz_err"]) - (data["detected_mjd_diff"]))))) - (data["1__fft_coefficient__coeff_1__attr__abs__y"])), data["detected_mjd_diff"] )) * 2.0)) +
(np.tanh((np.minimum(((data["detected_flux_err_mean"])), ((((data["1__kurtosis_x"]) / 2.0))))))) +
(np.where(np.where(data["flux_d0_pb3"]<0, data["0__kurtosis_x"], data["0__kurtosis_x"] )<0, np.where(data["detected_flux_ratio_sq_sum"]<0, data["flux_d0_pb3"], data["0__kurtosis_x"] ), data["5__kurtosis_x"] )) +
(((((((np.maximum(((data["2__fft_coefficient__coeff_1__attr__abs__x"])), ((((data["flux_median"]) - (data["flux_median"])))))) + (data["flux_median"]))) + (data["flux_median"]))) - (data["flux_w_mean"]))) +
(((((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) > (((((-1.0*((0.0)))) < (((data["flux_d0_pb3"]) + (data["3__fft_coefficient__coeff_0__attr__abs__y"]))))*1.)))*1.)) + (data["flux_ratio_sq_skew"]))/2.0)) +
(((np.where(data["4__skewness_x"]<0, np.tanh((data["1__skewness_x"])), ((data["0__fft_coefficient__coeff_0__attr__abs__y"]) * 2.0) )) - (data["2__fft_coefficient__coeff_1__attr__abs__x"]))) +
(((((np.minimum(((data["1__skewness_y"])), ((data["0__skewness_y"])))) - (((data["3__kurtosis_x"]) * 2.0)))) * 2.0)) +
(np.where(1.0 > -1, data["flux_err_min"], data["flux_d0_pb1"] )) +
(((((((data["detected_flux_err_median"]) - (data["detected_flux_std"]))) - (data["detected_flux_std"]))) - (data["hostgal_photoz"]))) +
((-1.0*(((((data["2__fft_coefficient__coeff_1__attr__abs__y"]) > (data["0__kurtosis_y"]))*1.))))) +
(np.minimum(((np.minimum(((data["4__fft_coefficient__coeff_1__attr__abs__x"])), ((data["4__fft_coefficient__coeff_1__attr__abs__x"]))))), ((np.where(data["0__fft_coefficient__coeff_1__attr__abs__x"]>0, data["5__kurtosis_x"], np.where(data["0__fft_coefficient__coeff_1__attr__abs__x"]>0, data["4__fft_coefficient__coeff_1__attr__abs__x"], data["flux_median"] ) ))))) +
(((np.where(data["detected_flux_err_mean"] > -1, data["1__kurtosis_x"], np.where(data["detected_flux_err_mean"] > -1, data["1__kurtosis_x"], np.where(data["detected_flux_err_mean"] > -1, data["1__kurtosis_x"], data["detected_flux_err_mean"] ) ) )) * 2.0)) +
(((((((((data["detected_flux_min"]) * 2.0)) + (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) * (data["2__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["flux_d0_pb3"]))) +
(((((((data["flux_skew"]) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) + (((((data["flux_skew"]) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) * 2.0)))) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) +
(((((data["0__kurtosis_x"]) - (data["flux_d0_pb0"]))) + (((np.minimum(((((data["0__kurtosis_x"]) + (data["1__kurtosis_y"])))), ((data["3__fft_coefficient__coeff_1__attr__abs__x"])))) - (data["flux_d0_pb0"]))))) +
((((((data["flux_skew"]) / 2.0)) + ((((data["flux_err_skew"]) + (data["flux_err_mean"]))/2.0)))/2.0)) +
(np.where(data["flux_d0_pb5"]>0, np.where(((data["detected_flux_err_median"]) * (data["5__skewness_x"])) > -1, data["detected_flux_skew"], ((data["distmod"]) - (data["flux_d0_pb0"])) ), data["ddf"] )) +
(((((((data["detected_flux_std"]) - (data["detected_flux_min"]))) - ((((data["flux_d0_pb1"]) > (data["detected_flux_std"]))*1.)))) - (data["detected_mjd_diff"]))) +
((-1.0*((((data["hostgal_photoz"]) - (np.tanh((((np.tanh(((-1.0*((((data["detected_flux_max"]) - (data["flux_dif3"])))))))) * 2.0))))))))) +
(((np.where(data["detected_mjd_diff"] > -1, ((data["4__fft_coefficient__coeff_1__attr__abs__x"]) - (data["1__fft_coefficient__coeff_1__attr__abs__y"])), data["1__fft_coefficient__coeff_1__attr__abs__y"] )) * 2.0)) +
(np.where(data["flux_skew"] > -1, ((data["5__kurtosis_x"]) - (data["1__skewness_x"])), ((((data["4__fft_coefficient__coeff_0__attr__abs__y"]) * (data["flux_max"]))) - (((data["4__fft_coefficient__coeff_0__attr__abs__y"]) / 2.0))) )) +
((((((data["detected_mean"]) > (data["1__skewness_y"]))*1.)) * (((data["detected_flux_err_median"]) * 2.0)))) +
(((((((((data["detected_mjd_size"]) * (((data["2__fft_coefficient__coeff_1__attr__abs__x"]) * (data["flux_d1_pb3"]))))) - (data["5__fft_coefficient__coeff_0__attr__abs__y"]))) - (data["detected_mjd_diff"]))) - (data["5__fft_coefficient__coeff_0__attr__abs__y"]))) +
(((((((data["flux_max"]) - (data["flux_d0_pb5"]))) * 2.0)) * 2.0)) +
((((data["detected_flux_err_min"]) + ((((((data["detected_mjd_diff"]) > (((data["flux_by_flux_ratio_sq_sum"]) * (data["detected_flux_err_min"]))))*1.)) * 2.0)))/2.0)) +
(((np.where(data["flux_ratio_sq_skew"] > -1, np.minimum(((data["detected_mjd_diff"])), ((((np.minimum(((data["flux_ratio_sq_skew"])), ((data["flux_ratio_sq_skew"])))) * 2.0)))), data["2__skewness_x"] )) * 2.0)) +
(np.where(data["detected_mjd_diff"]>0, ((data["5__kurtosis_x"]) - (data["0__fft_coefficient__coeff_1__attr__abs__x"])), data["0__fft_coefficient__coeff_1__attr__abs__x"] )) +
(((np.where(((data["detected_flux_dif2"]) - (data["detected_mjd_diff"])) > -1, (((data["detected_flux_dif2"]) < (data["detected_mjd_diff"]))*1.), data["2__kurtosis_y"] )) - (data["detected_flux_dif2"]))) +
(np.where(((data["5__fft_coefficient__coeff_1__attr__abs__y"]) - (data["0__skewness_x"]))<0, ((((data["5__fft_coefficient__coeff_1__attr__abs__y"]) - (data["0__skewness_x"]))) - (data["0__kurtosis_x"])), data["0__skewness_x"] )) +
(((((data["flux_d0_pb0"]) * ((((data["flux_d0_pb0"]) + (data["flux_d0_pb3"]))/2.0)))) - (data["2__fft_coefficient__coeff_1__attr__abs__y"]))) +
(np.where((((data["3__fft_coefficient__coeff_1__attr__abs__y"]) + (((data["1__kurtosis_y"]) * (data["flux_diff"]))))/2.0)<0, ((0.318310) * (data["detected_flux_err_mean"])), data["4__fft_coefficient__coeff_1__attr__abs__y"] )) +
((((-1.0*((np.where(data["flux_diff"] > -1, (-1.0*(((-1.0*((data["detected_flux_min"])))))), (-1.0*(((-1.0*((data["flux_by_flux_ratio_sq_skew"])))))) ))))) + (data["1__fft_coefficient__coeff_1__attr__abs__y"]))) +
(((np.where(data["distmod"] > -1, data["4__kurtosis_x"], data["3__fft_coefficient__coeff_0__attr__abs__y"] )) - (data["flux_d1_pb4"]))) +
(((data["detected_mjd_diff"]) - (np.maximum(((((((((data["detected_mjd_diff"]) < (data["detected_mjd_diff"]))*1.)) < (data["3__skewness_x"]))*1.))), ((((data["3__skewness_x"]) + (data["2__fft_coefficient__coeff_1__attr__abs__x"])))))))) +
(((((((((data["flux_by_flux_ratio_sq_skew"]) - (np.where(data["flux_skew"] > -1, data["4__fft_coefficient__coeff_1__attr__abs__y"], data["4__fft_coefficient__coeff_1__attr__abs__y"] )))) - (data["flux_median"]))) - (data["4__fft_coefficient__coeff_1__attr__abs__y"]))) * 2.0)) +
(((((data["distmod"]) * (np.maximum(((data["hostgal_photoz_err"])), ((np.maximum(((((data["distmod"]) * (data["hostgal_photoz_err"])))), ((data["hostgal_photoz_err"]))))))))) - (data["distmod"]))) +
(((((((((((data["4__kurtosis_x"]) - (data["2__kurtosis_y"]))) * 2.0)) - (data["2__kurtosis_x"]))) - (data["3__kurtosis_x"]))) * 2.0)) +
(((((((((data["hostgal_photoz_err"]) - (data["flux_d0_pb0"]))) - (data["flux_max"]))) + (((data["0__kurtosis_x"]) - (data["flux_d0_pb0"]))))) - (data["flux_max"]))) +
(np.where(data["flux_d0_pb1"]<0, data["1__fft_coefficient__coeff_1__attr__abs__x"], 3.0 )) +
(((data["flux_ratio_sq_skew"]) * (((np.where(((data["2__kurtosis_x"]) / 2.0)>0, np.tanh((data["distmod"])), data["hostgal_photoz_err"] )) / 2.0)))) +
(((((np.where((((data["1__fft_coefficient__coeff_1__attr__abs__x"]) + (data["detected_flux_min"]))/2.0)>0, data["detected_flux_std"], ((data["detected_flux_min"]) * (data["1__fft_coefficient__coeff_1__attr__abs__x"])) )) - (data["flux_d0_pb1"]))) * 2.0)) +
(((np.where(data["detected_mjd_diff"] > -1, ((((data["detected_flux_dif2"]) - (data["detected_mjd_diff"]))) * (data["flux_max"])), data["detected_flux_err_median"] )) - (data["detected_mjd_diff"]))) +
(((((((data["2__skewness_x"]) * (data["flux_d1_pb5"]))) + (((((data["detected_flux_std"]) - (data["2__skewness_x"]))) - (data["2__skewness_x"]))))) + (data["distmod"]))) +
(((np.where(((data["5__fft_coefficient__coeff_0__attr__abs__x"]) * 2.0) > -1, data["2__fft_coefficient__coeff_0__attr__abs__y"], ((((data["0__fft_coefficient__coeff_0__attr__abs__y"]) - (data["1__skewness_x"]))) - (data["2__fft_coefficient__coeff_0__attr__abs__y"])) )) * 2.0)) +
((((data["mjd_diff"]) > (data["hostgal_photoz"]))*1.)) +
(((((data["flux_d0_pb5"]) * (data["flux_d1_pb2"]))) * (((data["flux_max"]) - (data["hostgal_photoz"]))))) +
(np.where(data["flux_max"]>0, data["5__kurtosis_x"], ((data["mwebv"]) - (data["2__fft_coefficient__coeff_0__attr__abs__y"])) )) +
((((((data["detected_flux_mean"]) * (data["detected_flux_mean"]))) > (((data["detected_flux_mean"]) * 2.0)))*1.)))

def GPy(data):
    return ((((((data["flux_by_flux_ratio_sq_skew"]) + (data["flux_by_flux_ratio_sq_skew"]))) + (((((data["flux_by_flux_ratio_sq_skew"]) + (data["4__skewness_x"]))) + (data["flux_skew"]))))) +
(((((data["4__skewness_x"]) + (((data["detected_flux_min"]) + (data["4__skewness_x"]))))) + (((data["detected_flux_min"]) + (((data["4__skewness_x"]) + (data["detected_flux_min"]))))))) +
(((((((data["distmod"]) + (data["distmod"]))) + (data["distmod"]))) + ((((((data["distmod"]) + (data["distmod"]))) + (data["distmod"]))/2.0)))) +
(((((((data["flux_by_flux_ratio_sq_skew"]) + (((((data["flux_by_flux_ratio_sq_skew"]) + (data["detected_flux_min"]))) + (data["detected_flux_min"]))))) + (data["flux_by_flux_ratio_sq_skew"]))) + (data["4__skewness_x"]))) +
((((data["flux_max"]) + (data["flux_err_mean"]))/2.0)) +
(((((data["4__fft_coefficient__coeff_1__attr__abs__y"]) + (((data["distmod"]) + (((data["distmod"]) + (data["1__fft_coefficient__coeff_1__attr__abs__x"]))))))) + (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) +
(((((((data["ddf"]) + (data["5__fft_coefficient__coeff_0__attr__abs__y"]))) + (data["1__fft_coefficient__coeff_1__attr__abs__y"]))) * 2.0)) +
(((((((data["2__skewness_x"]) + (data["flux_by_flux_ratio_sq_skew"]))) + (data["flux_skew"]))) + (data["flux_by_flux_ratio_sq_skew"]))) +
(((np.maximum(((data["flux_d1_pb5"])), ((np.maximum(((data["2__skewness_x"])), ((1.0))))))) * 2.0)) +
((((data["flux_skew"]) + (((((data["flux_skew"]) + (np.minimum(((data["flux_skew"])), ((data["flux_skew"])))))) + (((data["flux_skew"]) + (data["flux_skew"]))))))/2.0)) +
((((((data["0__fft_coefficient__coeff_0__attr__abs__x"]) > (data["mjd_size"]))*1.)) - (data["detected_flux_skew"]))) +
(((((((((data["flux_err_min"]) * 2.0)) + (data["flux_err_min"]))) - (data["flux_err_min"]))) - (data["flux_min"]))) +
(((((((data["distmod"]) + (data["distmod"]))) + (data["flux_d0_pb0"]))) + (data["detected_flux_min"]))) +
(np.minimum(((data["3__fft_coefficient__coeff_0__attr__abs__x"])), (((((data["flux_min"]) + (data["flux_ratio_sq_skew"]))/2.0))))) +
(((data["flux_d0_pb4"]) + (((((np.tanh((((data["detected_flux_min"]) * 2.0)))) + (data["detected_flux_min"]))) + (data["detected_flux_min"]))))) +
(np.maximum((((-1.0*((data["flux_ratio_sq_sum"]))))), ((data["4__kurtosis_x"])))) +
(((0.636620) + ((((((((data["detected_mjd_diff"]) + (data["flux_by_flux_ratio_sq_skew"]))) + ((((0.0) + (data["flux_by_flux_ratio_sq_skew"]))/2.0)))) < (data["detected_flux_w_mean"]))*1.)))) +
(np.where(data["detected_flux_max"]>0, data["detected_flux_err_median"], (((data["2__fft_coefficient__coeff_1__attr__abs__x"]) + ((((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) + (data["detected_flux_err_median"]))) + (data["4__skewness_x"]))/2.0)))/2.0) )) +
(((data["distmod"]) + (((np.minimum(((2.0)), ((data["distmod"])))) + (((2.0) + (((data["distmod"]) + (2.0))))))))) +
(((data["0__fft_coefficient__coeff_1__attr__abs__x"]) + (((((((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) + (data["flux_err_min"]))) * 2.0)) + (data["flux_err_min"]))) + (data["flux_err_min"]))))) +
(np.tanh((np.where(data["flux_d0_pb5"] > -1, np.where(data["detected_mean"] > -1, data["2__kurtosis_x"], ((data["5__fft_coefficient__coeff_1__attr__abs__x"]) + (data["flux_err_min"])) ), data["1__fft_coefficient__coeff_0__attr__abs__x"] )))) +
(((data["flux_std"]) - (np.where(2.0<0, 1.0, 1.570796 )))) +
(((((data["4__skewness_x"]) + (data["flux_skew"]))) + (((data["hostgal_photoz"]) + (data["4__skewness_x"]))))) +
(((data["flux_skew"]) + (((data["1__skewness_x"]) + (data["2__skewness_x"]))))) +
(((data["distmod"]) + (np.maximum(((data["distmod"])), (((((data["distmod"]) + (data["distmod"]))/2.0))))))) +
(((data["flux_err_mean"]) + ((((((data["detected_flux_skew"]) + (data["flux_d1_pb0"]))/2.0)) + (data["flux_err_mean"]))))) +
(((((np.tanh((((data["mjd_diff"]) + (data["2__fft_coefficient__coeff_1__attr__abs__x"]))))) + (data["distmod"]))) + ((((data["detected_flux_median"]) + (data["distmod"]))/2.0)))) +
(((data["distmod"]) + (data["detected_mjd_size"]))) +
(((((np.maximum(((0.636620)), ((data["distmod"])))) + ((((data["detected_mjd_diff"]) + (data["distmod"]))/2.0)))) + (((0.318310) + (data["distmod"]))))) +
(np.minimum(((((np.minimum(((data["4__skewness_x"])), ((np.minimum(((data["detected_flux_by_flux_ratio_sq_skew"])), ((((data["3__fft_coefficient__coeff_1__attr__abs__x"]) + (data["hostgal_photoz"]))))))))) + (data["5__fft_coefficient__coeff_1__attr__abs__x"])))), ((data["4__fft_coefficient__coeff_1__attr__abs__x"])))) +
(np.where(np.minimum(((data["hostgal_photoz"])), ((data["hostgal_photoz"]))) > -1, data["hostgal_photoz"], np.minimum(((data["hostgal_photoz"])), ((np.minimum(((data["hostgal_photoz"])), ((data["hostgal_photoz"])))))) )) +
(((data["2__skewness_x"]) + (((((((data["4__skewness_x"]) + (data["3__skewness_x"]))) + (data["2__skewness_x"]))) + (data["flux_err_mean"]))))) +
(np.minimum(((((((data["flux_skew"]) * 2.0)) + (((data["flux_by_flux_ratio_sq_skew"]) + (data["4__skewness_x"])))))), ((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) + (data["detected_flux_min"])))))) +
(((np.minimum(((np.where(data["distmod"] > -1, (-1.0*((2.0))), data["5__fft_coefficient__coeff_1__attr__abs__x"] ))), ((((data["4__fft_coefficient__coeff_1__attr__abs__y"]) + (data["flux_err_median"])))))) * 2.0)) +
(np.where(data["detected_mjd_diff"]>0, data["detected_mjd_diff"], np.maximum(((data["detected_mjd_diff"])), ((data["detected_mjd_diff"]))) )) +
(((data["3__skewness_x"]) + (((data["4__skewness_x"]) + (data["4__skewness_x"]))))) +
(((((data["flux_err_min"]) - (((2.0) - (data["5__fft_coefficient__coeff_1__attr__abs__x"]))))) - (((((2.0) - (data["flux_err_min"]))) - (2.0))))) +
(((np.minimum(((((data["detected_flux_err_min"]) + (data["detected_flux_err_min"])))), ((data["detected_flux_err_min"])))) * 2.0)) +
(np.minimum(((np.minimum(((data["5__fft_coefficient__coeff_1__attr__abs__y"])), ((np.minimum(((3.141593)), ((((data["5__fft_coefficient__coeff_1__attr__abs__y"]) - (3.141593)))))))))), ((((data["5__fft_coefficient__coeff_1__attr__abs__y"]) - (3.141593)))))) +
(((np.where(((data["detected_mjd_size"]) * 2.0) > -1, data["detected_mjd_size"], data["detected_mjd_size"] )) + (data["detected_mjd_size"]))) +
(((((data["detected_mjd_diff"]) + (data["distmod"]))) + (data["hostgal_photoz"]))) +
((((data["detected_mjd_diff"]) + (data["detected_mjd_diff"]))/2.0)) +
(((np.minimum((((((((-1.0*((1.570796)))) - (data["distmod"]))) - (data["distmod"])))), (((-1.0*((1.570796))))))) - (data["distmod"]))) +
(((np.minimum(((((data["flux_err_std"]) + (data["flux_err_std"])))), ((((data["flux_err_std"]) + (((data["2__skewness_x"]) + (data["flux_err_std"])))))))) + (data["2__skewness_x"]))) +
(np.minimum(((np.minimum(((data["3__skewness_y"])), ((data["flux_err_std"]))))), ((((np.minimum(((data["flux_err_std"])), ((data["flux_err_std"])))) + (data["3__skewness_y"])))))) +
(((((data["3__skewness_x"]) + (np.maximum(((data["3__skewness_x"])), ((data["detected_flux_min"])))))) * 2.0)) +
(np.where(data["distmod"]<0, data["distmod"], data["distmod"] )) +
(np.minimum(((((data["detected_mjd_diff"]) + (((((data["detected_mjd_diff"]) + (data["5__kurtosis_y"]))) + (data["detected_mjd_diff"])))))), ((data["detected_mjd_diff"])))) +
(((((((data["1__skewness_x"]) + (data["flux_err_std"]))) + (((data["flux_err_std"]) + ((((data["1__skewness_x"]) + (data["1__skewness_x"]))/2.0)))))) + (data["1__skewness_x"]))) +
(np.minimum(((data["3__skewness_y"])), ((np.minimum(((np.minimum(((np.minimum(((data["flux_err_std"])), ((data["3__skewness_y"]))))), ((data["detected_mean"]))))), ((data["3__skewness_y"]))))))) +
(((((((data["detected_mean"]) + (data["hostgal_photoz"]))) * 2.0)) + (data["hostgal_photoz"]))) +
(((data["0__fft_coefficient__coeff_1__attr__abs__x"]) + (np.where(data["detected_mjd_diff"] > -1, data["detected_mjd_diff"], np.where(data["detected_mjd_diff"] > -1, data["detected_mjd_diff"], ((data["flux_d0_pb0"]) + (data["detected_mjd_diff"])) ) )))) +
(((np.minimum(((np.where(data["distmod"] > -1, np.where(data["4__fft_coefficient__coeff_1__attr__abs__x"]>0, 2.0, ((data["5__kurtosis_x"]) * 2.0) ), data["5__fft_coefficient__coeff_1__attr__abs__x"] ))), ((data["5__kurtosis_x"])))) * 2.0)) +
(((data["flux_by_flux_ratio_sq_skew"]) + (np.minimum(((((((data["detected_mjd_diff"]) + (data["detected_mjd_diff"]))) * 2.0))), ((((data["flux_by_flux_ratio_sq_skew"]) * 2.0))))))) +
(((((np.minimum(((data["detected_mjd_diff"])), ((np.minimum(((data["detected_mjd_diff"])), ((np.minimum(((data["detected_mjd_diff"])), ((data["flux_by_flux_ratio_sq_skew"])))))))))) * 2.0)) + (data["flux_ratio_sq_skew"]))) +
(((((data["5__kurtosis_x"]) + (((data["flux_d0_pb5"]) - (data["detected_flux_max"]))))) - (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) +
(np.minimum(((data["flux_err_min"])), (((((((np.where(data["flux_std"] > -1, np.minimum(((data["flux_err_min"])), ((data["5__kurtosis_x"]))), data["flux_err_min"] )) * 2.0)) + (data["flux_err_min"]))/2.0))))) +
(((((data["0__fft_coefficient__coeff_1__attr__abs__x"]) + (((((data["distmod"]) + (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) + (data["hostgal_photoz"]))))) + (((data["0__fft_coefficient__coeff_0__attr__abs__y"]) + (data["0__fft_coefficient__coeff_1__attr__abs__x"]))))) +
(((np.tanh((((((data["3__kurtosis_x"]) + (data["3__kurtosis_x"]))) * 2.0)))) + (data["3__kurtosis_x"]))) +
(np.minimum(((data["flux_err_mean"])), ((np.minimum(((data["3__skewness_y"])), ((((data["1__skewness_x"]) + (data["3__skewness_y"]))))))))) +
(((((data["flux_err_min"]) + (((data["detected_flux_min"]) + (data["flux_err_min"]))))) + (((data["detected_flux_min"]) + (data["flux_err_min"]))))) +
(((((data["0__fft_coefficient__coeff_0__attr__abs__y"]) + (data["0__fft_coefficient__coeff_1__attr__abs__y"]))) * 2.0)) +
(((((data["hostgal_photoz"]) * 2.0)) + (np.where(data["detected_mjd_diff"] > -1, np.where(data["hostgal_photoz"] > -1, data["detected_mjd_diff"], data["hostgal_photoz"] ), (10.79039764404296875) )))) +
(np.where(data["distmod"] > -1, ((data["flux_min"]) - (data["hostgal_photoz"])), ((data["distmod"]) + (data["5__fft_coefficient__coeff_0__attr__abs__y"])) )) +
(((((data["flux_err_min"]) - (data["detected_mjd_diff"]))) - (data["detected_mjd_diff"]))) +
(((data["3__skewness_x"]) + (((((((data["flux_max"]) + (data["4__skewness_x"]))) * 2.0)) + (data["flux_skew"]))))) +
(((np.where(data["hostgal_photoz"] > -1, np.where(data["flux_skew"] > -1, np.where(data["flux_skew"] > -1, data["hostgal_photoz"], data["hostgal_photoz"] ), data["flux_skew"] ), data["flux_skew"] )) * 2.0)) +
(np.where(((np.where(data["distmod"] > -1, data["distmod"], data["distmod"] )) / 2.0) > -1, (4.0), data["distmod"] )) +
((((data["detected_flux_min"]) + (data["0__fft_coefficient__coeff_1__attr__abs__y"]))/2.0)) +
(((np.minimum(((((data["0__skewness_x"]) + (data["0__skewness_x"])))), ((data["detected_flux_err_min"])))) + (data["0__skewness_x"]))) +
(((data["3__skewness_x"]) + (np.maximum(((((((((data["flux_skew"]) * 2.0)) / 2.0)) + (data["5__fft_coefficient__coeff_1__attr__abs__x"])))), ((data["flux_skew"])))))) +
(((data["detected_mjd_diff"]) + (((np.where(np.minimum(((data["detected_mjd_diff"])), ((data["detected_mjd_diff"]))) > -1, data["flux_by_flux_ratio_sq_skew"], data["detected_mjd_diff"] )) - (data["flux_diff"]))))) +
((((-1.0*((((data["1__skewness_x"]) - (((data["flux_d0_pb0"]) + (data["1__skewness_x"])))))))) + ((((((data["distmod"]) * 2.0)) + (data["1__skewness_x"]))/2.0)))) +
(np.minimum(((np.tanh((np.minimum(((data["4__skewness_x"])), ((data["flux_err_min"]))))))), ((data["5__kurtosis_x"])))) +
((((((((((data["4__kurtosis_x"]) - (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) < (((data["4__kurtosis_x"]) * 2.0)))*1.)) + (((data["4__kurtosis_x"]) - (data["0__fft_coefficient__coeff_0__attr__abs__y"]))))) * 2.0)) +
(((((data["distmod"]) + (1.570796))) + (data["distmod"]))) +
(((data["5__fft_coefficient__coeff_0__attr__abs__y"]) - (data["detected_mjd_diff"]))) +
(((np.minimum(((data["5__skewness_x"])), ((data["5__fft_coefficient__coeff_1__attr__abs__x"])))) - (((data["1__fft_coefficient__coeff_1__attr__abs__x"]) - (((data["5__skewness_x"]) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))))))) +
(((((((((data["0__skewness_x"]) - (data["4__fft_coefficient__coeff_1__attr__abs__y"]))) + (data["detected_flux_min"]))) + (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["4__fft_coefficient__coeff_1__attr__abs__y"]))) +
(((((((data["hostgal_photoz"]) + (((np.minimum(((data["hostgal_photoz"])), (((((data["hostgal_photoz"]) + (data["flux_d0_pb5"]))/2.0))))) - (data["hostgal_photoz"]))))/2.0)) + (data["hostgal_photoz"]))/2.0)) +
(((((((((data["hostgal_photoz"]) + (np.minimum(((data["hostgal_photoz"])), ((data["4__skewness_x"])))))/2.0)) + (((data["4__fft_coefficient__coeff_0__attr__abs__x"]) * 2.0)))/2.0)) + (data["hostgal_photoz"]))) +
(((data["4__fft_coefficient__coeff_1__attr__abs__x"]) - ((((3.141593) + (((data["mjd_size"]) / 2.0)))/2.0)))) +
(np.minimum(((data["1__kurtosis_x"])), ((data["1__kurtosis_x"])))) +
(np.where(data["hostgal_photoz"] > -1, data["hostgal_photoz"], data["flux_min"] )) +
(((((((data["3__kurtosis_y"]) - (data["0__fft_coefficient__coeff_1__attr__abs__y"]))) - (data["detected_flux_ratio_sq_skew"]))) - (data["0__fft_coefficient__coeff_1__attr__abs__y"]))) +
(((((((((data["2__skewness_x"]) - (data["detected_flux_err_mean"]))) - (data["detected_flux_err_median"]))) - (data["flux_std"]))) - (((data["flux_d0_pb0"]) - (data["detected_flux_err_mean"]))))) +
(((((data["flux_d0_pb5"]) + (((data["distmod"]) - (data["0__fft_coefficient__coeff_0__attr__abs__y"]))))) + (np.where(data["distmod"] > -1, data["flux_d0_pb5"], data["distmod"] )))) +
(((np.minimum(((((data["1__skewness_y"]) + ((((data["detected_flux_max"]) + (data["flux_skew"]))/2.0))))), ((((data["flux_skew"]) + (data["2__skewness_y"])))))) + (data["detected_flux_max"]))) +
(np.minimum(((((((((data["detected_flux_min"]) + (data["detected_flux_min"]))) + (data["5__skewness_x"]))) + (data["detected_flux_min"])))), ((((data["detected_flux_min"]) + (data["hostgal_photoz"])))))) +
((((data["detected_flux_std"]) + (((((data["2__kurtosis_x"]) - (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) / 2.0)))/2.0)) +
(((((((data["3__skewness_y"]) - (np.maximum(((data["0__fft_coefficient__coeff_0__attr__abs__y"])), ((data["flux_ratio_sq_skew"])))))) - (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) - (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) +
(((((data["hostgal_photoz_err"]) - (data["flux_d1_pb0"]))) + (((((data["4__skewness_y"]) - (data["flux_d1_pb0"]))) + (((data["hostgal_photoz_err"]) + (data["3__skewness_y"]))))))) +
(((((((((data["detected_mjd_diff"]) - (data["1__fft_coefficient__coeff_0__attr__abs__x"]))) - (data["flux_max"]))) - (data["hostgal_photoz"]))) + (((data["detected_mjd_diff"]) - (data["detected_flux_err_max"]))))) +
(np.where(data["detected_mjd_diff"] > -1, ((((((data["4__kurtosis_x"]) - (data["detected_mjd_diff"]))) - (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) - (data["detected_mjd_diff"])), data["detected_mjd_diff"] )) +
(((((data["ddf"]) - (((data["flux_d1_pb0"]) * 2.0)))) - (data["detected_flux_std"]))) +
(np.where(np.where(data["2__kurtosis_x"] > -1, data["4__kurtosis_y"], data["4__kurtosis_x"] ) > -1, data["detected_mjd_diff"], np.where(data["detected_mjd_diff"] > -1, data["2__kurtosis_x"], data["4__kurtosis_y"] ) )) +
(((((((data["flux_d0_pb2"]) + (((((data["flux_d0_pb2"]) - (data["flux_d0_pb1"]))) - (data["flux_d0_pb0"]))))) * 2.0)) + (data["flux_d0_pb4"]))) +
(((np.tanh(((((-1.0*((data["distmod"])))) * 2.0)))) - (((((data["detected_flux_std"]) + (data["distmod"]))) * 2.0)))) +
(((((((data["1__skewness_x"]) - (data["detected_flux_std"]))) - (data["hostgal_photoz"]))) * 2.0)) +
(((((data["detected_mjd_diff"]) - ((((data["4__skewness_x"]) > (np.minimum((((((data["0__fft_coefficient__coeff_1__attr__abs__y"]) > (data["4__skewness_x"]))*1.))), ((2.0)))))*1.)))) * 2.0)) +
(np.where(((data["distmod"]) / 2.0) > -1, (((((((data["flux_dif2"]) > (data["0__fft_coefficient__coeff_0__attr__abs__y"]))*1.)) - (data["distmod"]))) - (data["distmod"])), data["distmod"] )) +
(((data["0__skewness_x"]) + (np.where(((data["4__kurtosis_x"]) - ((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) + (data["flux_by_flux_ratio_sq_skew"]))/2.0))) > -1, data["distmod"], data["distmod"] )))) +
((((((((data["detected_flux_dif2"]) > (((data["detected_mjd_diff"]) - ((((data["detected_mjd_diff"]) > (data["detected_mjd_diff"]))*1.)))))*1.)) - (data["detected_mjd_diff"]))) * 2.0)) +
(np.minimum(((np.minimum(((((data["0__fft_coefficient__coeff_1__attr__abs__y"]) - (data["2__kurtosis_y"])))), ((data["1__skewness_y"]))))), ((data["detected_flux_max"])))) +
(np.maximum(((((np.tanh(((((((-1.0*((np.maximum(((data["detected_mjd_diff"])), ((data["hostgal_photoz_err"]))))))) * 2.0)) * (data["3__kurtosis_x"]))))) / 2.0))), ((data["detected_flux_ratio_sq_skew"])))) +
(((((((data["flux_ratio_sq_skew"]) + (data["flux_d0_pb1"]))) - (data["5__fft_coefficient__coeff_1__attr__abs__y"]))) + (((data["flux_d0_pb0"]) - (data["2__fft_coefficient__coeff_1__attr__abs__y"]))))) +
((((((np.tanh((np.where(data["distmod"] > -1, data["4__skewness_x"], data["3__fft_coefficient__coeff_0__attr__abs__x"] )))) + (data["hostgal_photoz_err"]))) + (data["distmod"]))/2.0)) +
((-1.0*((((data["flux_skew"]) + ((((data["detected_flux_skew"]) > (data["detected_flux_skew"]))*1.))))))) +
((((((((-1.0*((np.tanh((data["detected_flux_err_max"])))))) - (((data["flux_max"]) * 2.0)))) * 2.0)) - (data["detected_flux_err_max"]))) +
((((((((data["3__skewness_x"]) - (data["5__fft_coefficient__coeff_0__attr__abs__y"]))) + (((data["3__kurtosis_x"]) + (((data["detected_flux_skew"]) - (data["detected_flux_diff"]))))))/2.0)) - (data["detected_flux_err_mean"]))) +
(np.where(((data["detected_flux_skew"]) - (data["detected_mjd_diff"]))<0, ((data["flux_by_flux_ratio_sq_skew"]) - (data["detected_flux_max"])), data["detected_flux_skew"] )) +
(((((((((data["0__skewness_x"]) - (data["distmod"]))) - (data["4__fft_coefficient__coeff_1__attr__abs__y"]))) - (data["distmod"]))) - (data["3__fft_coefficient__coeff_1__attr__abs__x"]))) +
((((((data["ddf"]) + (0.636620))) + (data["detected_mjd_diff"]))/2.0)) +
(((((((((data["5__fft_coefficient__coeff_1__attr__abs__y"]) < (data["flux_d1_pb1"]))*1.)) < (data["5__skewness_x"]))*1.)) + (data["5__skewness_x"]))) +
((-1.0*((np.where(data["detected_mean"]>0, ((data["4__fft_coefficient__coeff_1__attr__abs__x"]) - (data["mjd_size"])), data["detected_flux_std"] ))))) +
((((((data["0__kurtosis_x"]) + (data["1__kurtosis_x"]))/2.0)) + (data["1__kurtosis_x"]))) +
(((np.where(data["detected_flux_max"] > -1, ((((((data["detected_flux_w_mean"]) - (data["detected_mjd_diff"]))) + (data["detected_flux_skew"]))) - (data["detected_mjd_diff"])), data["detected_mjd_diff"] )) * 2.0)) +
(((((((data["detected_flux_diff"]) - (data["detected_flux_err_median"]))) - (data["flux_d0_pb0"]))) - (((data["detected_flux_err_mean"]) * 2.0)))) +
(((data["4__skewness_x"]) - (data["1__fft_coefficient__coeff_0__attr__abs__y"]))) +
(((((np.where((((np.minimum(((data["hostgal_photoz_err"])), ((data["detected_mjd_diff"])))) + (data["distmod"]))/2.0) > -1, data["flux_dif2"], data["distmod"] )) * 2.0)) * 2.0)) +
(np.where(data["detected_flux_min"] > -1, (((((data["flux_diff"]) + (((np.where(data["hostgal_photoz"] > -1, data["hostgal_photoz"], data["detected_flux_min"] )) * 2.0)))/2.0)) * 2.0), data["detected_flux_min"] )) +
(((((data["5__skewness_y"]) - (((data["detected_mjd_diff"]) * 2.0)))) - (data["flux_d0_pb5"]))) +
(np.where(np.tanh(((-1.0*((np.maximum(((data["1__fft_coefficient__coeff_1__attr__abs__x"])), ((data["detected_flux_err_median"]))))))))>0, data["5__fft_coefficient__coeff_0__attr__abs__x"], np.where(data["3__skewness_x"]>0, data["distmod"], data["5__fft_coefficient__coeff_0__attr__abs__x"] ) )) +
(np.where(data["flux_dif2"]<0, data["flux_dif2"], np.where(((data["distmod"]) * 2.0) > -1, data["flux_dif2"], data["distmod"] ) )) +
(((((((data["flux_d0_pb2"]) - (data["detected_flux_std"]))) + (data["flux_median"]))) + (((((data["flux_d0_pb2"]) - (data["detected_flux_std"]))) + (data["flux_median"]))))) +
(((((data["flux_err_min"]) + (((((data["flux_err_min"]) + (np.minimum(((data["detected_mean"])), ((((data["flux_err_min"]) - (data["flux_err_median"])))))))) * 2.0)))) * 2.0)) +
(np.where(data["hostgal_photoz"] > -1, ((((data["detected_mjd_diff"]) - (data["distmod"]))) * 2.0), data["distmod"] )) +
(np.where(data["flux_skew"] > -1, ((np.where(data["hostgal_photoz_err"]>0, np.tanh((data["3__fft_coefficient__coeff_0__attr__abs__y"])), data["flux_skew"] )) - (data["hostgal_photoz_err"])), data["flux_skew"] )) +
(np.where(((data["distmod"]) / 2.0) > -1, ((((data["hostgal_photoz_err"]) - ((2.19761300086975098)))) * (data["hostgal_photoz"])), data["distmod"] )) +
(((data["detected_mjd_diff"]) - (np.maximum(((data["4__fft_coefficient__coeff_0__attr__abs__y"])), (((((((np.maximum(((data["5__fft_coefficient__coeff_1__attr__abs__x"])), ((data["4__fft_coefficient__coeff_0__attr__abs__y"])))) - (data["detected_mjd_diff"]))) < (data["3__fft_coefficient__coeff_1__attr__abs__x"]))*1.))))))) +
(np.where(data["5__kurtosis_x"] > -1, np.where(data["3__fft_coefficient__coeff_1__attr__abs__x"]>0, data["5__kurtosis_x"], data["5__kurtosis_x"] ), data["hostgal_photoz"] )) +
(((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) + (((data["5__fft_coefficient__coeff_0__attr__abs__y"]) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))))) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) +
(((((data["flux_by_flux_ratio_sq_skew"]) - (data["5__fft_coefficient__coeff_0__attr__abs__y"]))) + (((((data["1__skewness_y"]) - (data["5__fft_coefficient__coeff_0__attr__abs__y"]))) * 2.0)))) +
(((((((((((((((((((data["flux_dif2"]) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +
(((((data["detected_flux_min"]) - (((data["detected_flux_std"]) * 2.0)))) - (((((((data["detected_flux_std"]) + (data["distmod"]))) - (data["4__fft_coefficient__coeff_1__attr__abs__x"]))) * 2.0)))) +
(np.tanh((((data["flux_ratio_sq_skew"]) - (data["flux_d0_pb4"]))))) +
((((((((data["flux_mean"]) + ((((data["5__skewness_y"]) + (data["flux_min"]))/2.0)))/2.0)) - (data["detected_flux_max"]))) * 2.0)) +
((((((data["4__kurtosis_y"]) + (data["2__kurtosis_y"]))/2.0)) + (((((((data["2__kurtosis_y"]) - (data["0__fft_coefficient__coeff_0__attr__abs__y"]))) + (data["distmod"]))) + (data["2__kurtosis_y"]))))) +
(((data["5__skewness_x"]) - (((data["0__fft_coefficient__coeff_0__attr__abs__y"]) - ((((data["3__kurtosis_y"]) > (data["detected_mjd_diff"]))*1.)))))) +
((((((((data["4__kurtosis_x"]) - (data["flux_d1_pb1"]))) + (data["4__kurtosis_x"]))/2.0)) - (data["flux_d1_pb1"]))) +
(((((np.where(data["3__kurtosis_y"] > -1, data["3__kurtosis_y"], np.where((0.98831200599670410)<0, data["3__kurtosis_y"], data["3__kurtosis_y"] ) )) - (data["flux_by_flux_ratio_sq_skew"]))) - (data["0__fft_coefficient__coeff_1__attr__abs__y"]))) +
(((data["detected_flux_skew"]) + (data["3__skewness_y"]))) +
(np.where(np.where(data["5__fft_coefficient__coeff_1__attr__abs__x"] > -1, data["detected_mjd_diff"], ((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (data["5__fft_coefficient__coeff_1__attr__abs__x"])) ) > -1, ((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (data["1__fft_coefficient__coeff_1__attr__abs__y"])), data["detected_mjd_diff"] )) +
(((np.tanh((np.where(np.minimum(((data["5__kurtosis_x"])), ((data["2__kurtosis_x"])))<0, data["flux_dif3"], data["flux_d0_pb3"] )))) * 2.0)) +
(((((data["5__kurtosis_y"]) - (((data["detected_flux_diff"]) - (data["mwebv"]))))) - (data["detected_flux_diff"]))) +
(np.where(data["2__fft_coefficient__coeff_1__attr__abs__x"]>0, np.minimum(((np.minimum(((np.minimum(((data["0__skewness_x"])), ((data["flux_diff"]))))), ((data["flux_skew"]))))), ((data["0__skewness_x"]))), data["4__fft_coefficient__coeff_1__attr__abs__y"] )) +
(((((data["5__kurtosis_y"]) + (((data["flux_median"]) + (((data["3__skewness_y"]) + (data["1__kurtosis_y"]))))))) * 2.0)) +
((((((data["flux_diff"]) + ((((data["flux_skew"]) + (((data["0__fft_coefficient__coeff_0__attr__abs__y"]) - (data["2__fft_coefficient__coeff_1__attr__abs__x"]))))/2.0)))/2.0)) - (data["2__fft_coefficient__coeff_1__attr__abs__x"]))) +
(((((data["flux_ratio_sq_skew"]) - (data["flux_by_flux_ratio_sq_skew"]))) * (((((((data["flux_d1_pb2"]) + (data["flux_err_min"]))/2.0)) + (((data["distmod"]) - (data["flux_err_min"]))))/2.0)))) +
(((((np.tanh((((data["0__skewness_x"]) * (data["distmod"]))))) * 2.0)) * (data["1__fft_coefficient__coeff_0__attr__abs__x"]))) +
(((((data["flux_max"]) * (np.tanh((data["detected_flux_err_mean"]))))) / 2.0)) +
(((((data["3__skewness_x"]) - (((data["detected_flux_ratio_sq_skew"]) - (data["2__skewness_y"]))))) - (((data["detected_flux_ratio_sq_skew"]) - (((data["detected_flux_ratio_sq_skew"]) - (data["detected_flux_ratio_sq_skew"]))))))) +
(np.where(data["distmod"]<0, (((data["4__skewness_x"]) + (((data["distmod"]) + (((data["hostgal_photoz_err"]) * (data["hostgal_photoz_err"]))))))/2.0), data["hostgal_photoz_err"] )) +
(((((((((data["0__kurtosis_x"]) + (data["flux_median"]))) + (data["flux_median"]))) + (((data["flux_median"]) + (data["4__kurtosis_y"]))))) * 2.0)) +
((((-1.0*((np.where(data["distmod"] > -1, ((data["distmod"]) + (data["2__fft_coefficient__coeff_1__attr__abs__y"])), data["flux_by_flux_ratio_sq_skew"] ))))) * 2.0)) +
(np.where(data["detected_mjd_size"] > -1, np.where(data["distmod"] > -1, ((data["flux_d0_pb1"]) - (data["2__fft_coefficient__coeff_1__attr__abs__y"])), data["2__fft_coefficient__coeff_1__attr__abs__y"] ), data["4__fft_coefficient__coeff_1__attr__abs__y"] )) +
(((((data["0__kurtosis_x"]) - (data["detected_flux_std"]))) - (data["flux_d0_pb0"]))) +
(((((((data["2__skewness_x"]) - (data["flux_d0_pb5"]))) + (np.where(data["1__skewness_y"]<0, ((data["2__skewness_x"]) - (data["flux_d0_pb5"])), data["detected_flux_skew"] )))) * 2.0)) +
(((np.where(data["hostgal_photoz"]>0, ((data["hostgal_photoz_err"]) - (((data["hostgal_photoz_err"]) - (data["hostgal_photoz_err"])))), data["flux_d0_pb4"] )) - (data["detected_flux_std"]))) +
(((data["detected_flux_min"]) + (data["detected_mjd_diff"]))) +
(((((data["flux_dif2"]) * 2.0)) / 2.0)) +
(((np.where(np.where(((data["detected_mjd_diff"]) - (data["detected_mjd_diff"])) > -1, data["detected_mjd_diff"], data["flux_err_mean"] )>0, data["detected_flux_skew"], data["flux_err_mean"] )) - (data["detected_mjd_diff"]))) +
(((np.minimum(((data["5__fft_coefficient__coeff_0__attr__abs__x"])), ((((data["flux_std"]) - (data["detected_mean"])))))) + (((data["flux_skew"]) * (((data["hostgal_photoz"]) + (data["hostgal_photoz"]))))))) +
(np.where(data["distmod"]>0, ((((data["distmod"]) * (data["3__kurtosis_x"]))) - (data["4__fft_coefficient__coeff_1__attr__abs__y"])), ((data["distmod"]) * (data["3__kurtosis_x"])) )) +
(np.where(data["flux_d0_pb1"] > -1, ((data["flux_d0_pb0"]) - (data["2__fft_coefficient__coeff_1__attr__abs__y"])), np.where(data["flux_d0_pb0"] > -1, ((data["flux_d0_pb0"]) - (data["flux_d0_pb0"])), data["2__fft_coefficient__coeff_1__attr__abs__y"] ) )) +
((((data["distmod"]) + ((-1.0*((data["2__kurtosis_x"])))))/2.0)) +
(np.where(data["3__skewness_y"]<0, ((data["0__skewness_x"]) / 2.0), np.where(data["4__fft_coefficient__coeff_1__attr__abs__x"] > -1, np.where(data["0__skewness_x"] > -1, data["0__skewness_x"], data["0__skewness_x"] ), data["0__skewness_x"] ) )) +
(((data["detected_flux_diff"]) + (((((data["distmod"]) + (data["distmod"]))) + (data["flux_diff"]))))) +
(((data["1__fft_coefficient__coeff_0__attr__abs__x"]) - (data["3__fft_coefficient__coeff_0__attr__abs__x"]))) +
(np.minimum(((((data["mjd_diff"]) * 2.0))), ((data["mjd_diff"])))) +
(((np.where(np.where(data["1__fft_coefficient__coeff_0__attr__abs__x"]>0, (((data["5__fft_coefficient__coeff_1__attr__abs__x"]) + (data["5__fft_coefficient__coeff_1__attr__abs__y"]))/2.0), data["2__fft_coefficient__coeff_1__attr__abs__x"] )>0, data["5__fft_coefficient__coeff_1__attr__abs__x"], data["5__fft_coefficient__coeff_0__attr__abs__y"] )) - (data["2__fft_coefficient__coeff_1__attr__abs__x"]))) +
(((((data["flux_by_flux_ratio_sq_skew"]) - (np.tanh((np.minimum(((data["2__fft_coefficient__coeff_1__attr__abs__x"])), ((np.tanh((data["hostgal_photoz"])))))))))) * (((data["2__fft_coefficient__coeff_1__attr__abs__x"]) * 2.0)))) +
(np.where(data["hostgal_photoz"] > -1, (-1.0*((data["detected_flux_ratio_sq_sum"]))), np.where(data["4__kurtosis_x"] > -1, data["flux_dif3"], ((data["hostgal_photoz"]) * (data["detected_flux_min"])) ) )) +
(np.where(((data["distmod"]) + (data["detected_flux_diff"]))<0, ((data["distmod"]) - (data["detected_flux_diff"])), data["flux_diff"] )) +
(((((data["2__kurtosis_y"]) - (data["detected_flux_diff"]))) - (data["detected_flux_ratio_sq_sum"]))) +
(((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) - (((np.maximum(((np.maximum(((data["1__fft_coefficient__coeff_1__attr__abs__x"])), ((data["1__fft_coefficient__coeff_1__attr__abs__x"]))))), ((data["flux_max"])))) - (data["5__fft_coefficient__coeff_0__attr__abs__y"]))))) +
(((data["detected_mjd_diff"]) - (np.where(data["distmod"] > -1, data["flux_err_skew"], (((((data["detected_flux_median"]) / 2.0)) + (data["detected_mjd_diff"]))/2.0) )))) +
(((((np.where(data["0__skewness_x"]>0, data["5__skewness_x"], data["1__skewness_x"] )) + (data["1__skewness_x"]))) + (((data["0__skewness_x"]) + (data["1__skewness_x"]))))) +
(((((data["flux_err_min"]) * 2.0)) - (data["detected_flux_max"]))) +
(((((data["flux_dif3"]) + (((((data["2__fft_coefficient__coeff_1__attr__abs__y"]) * (data["3__kurtosis_x"]))) - (data["4__fft_coefficient__coeff_1__attr__abs__y"]))))) + (data["hostgal_photoz_err"]))) +
(np.where(data["distmod"] > -1, ((data["5__fft_coefficient__coeff_1__attr__abs__y"]) - (2.0)), ((((data["2__skewness_x"]) - (data["distmod"]))) - (data["2__fft_coefficient__coeff_0__attr__abs__x"])) )) +
(((((data["ddf"]) - (np.where(data["flux_ratio_sq_sum"] > -1, data["flux_d1_pb0"], data["ddf"] )))) - (np.where(data["hostgal_photoz"] > -1, data["hostgal_photoz"], data["flux_ratio_sq_sum"] )))) +
((((data["flux_err_skew"]) + ((-1.0*(((((-1.0*(((((-1.0*((((data["flux_ratio_sq_skew"]) - (data["flux_median"])))))) / 2.0))))) - (data["detected_mean"])))))))/2.0)) +
(((((data["distmod"]) * (data["distmod"]))) - (data["hostgal_photoz"]))) +
(((((((data["hostgal_photoz"]) + (((data["hostgal_photoz"]) * 2.0)))) + (np.where(data["hostgal_photoz"] > -1, data["flux_std"], data["hostgal_photoz"] )))) * (data["4__skewness_x"]))) +
(((((np.where(((data["5__kurtosis_x"]) / 2.0) > -1, data["5__kurtosis_x"], data["5__kurtosis_x"] )) * (data["4__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["flux_d0_pb0"]))) +
((((((((((data["flux_skew"]) + (data["flux_d1_pb1"]))/2.0)) * (data["2__fft_coefficient__coeff_1__attr__abs__y"]))) - (((data["detected_mjd_diff"]) * (data["detected_mjd_diff"]))))) - (data["flux_d1_pb1"]))) +
(((((np.where(data["distmod"] > -1, data["flux_ratio_sq_skew"], np.where(data["flux_err_skew"] > -1, data["flux_err_min"], data["flux_err_skew"] ) )) - (data["detected_flux_std"]))) - (data["distmod"]))) +
(np.where(((data["detected_mjd_diff"]) / 2.0) > -1, ((data["detected_mjd_diff"]) - (data["3__fft_coefficient__coeff_1__attr__abs__y"])), (((data["3__fft_coefficient__coeff_1__attr__abs__y"]) < (data["detected_mjd_diff"]))*1.) )) +
(((((np.where(data["distmod"] > -1, data["mjd_diff"], data["distmod"] )) + (((data["detected_mjd_diff"]) + (data["0__fft_coefficient__coeff_0__attr__abs__x"]))))) - (data["3__fft_coefficient__coeff_0__attr__abs__y"]))) +
(((np.where(data["flux_dif3"] > -1, data["detected_flux_ratio_sq_skew"], np.tanh((data["3__fft_coefficient__coeff_1__attr__abs__y"])) )) - ((-1.0*(((-1.0*((np.tanh((data["3__fft_coefficient__coeff_1__attr__abs__y"]))))))))))) +
(((((np.where(data["flux_w_mean"]<0, data["flux_w_mean"], data["flux_w_mean"] )) * (data["2__fft_coefficient__coeff_1__attr__abs__x"]))) - (np.where(data["flux_d0_pb0"]<0, data["flux_w_mean"], data["flux_d0_pb0"] )))) +
(((data["mjd_diff"]) + (data["mjd_diff"]))) +
(((np.where(data["flux_ratio_sq_skew"] > -1, data["detected_mjd_diff"], ((data["detected_flux_max"]) - (data["1__kurtosis_x"])) )) - (((data["flux_max"]) - (data["flux_max"]))))) +
(((data["flux_err_max"]) + (((((data["flux_err_max"]) + (data["flux_ratio_sq_skew"]))) + (np.where(data["0__kurtosis_x"]<0, data["flux_err_max"], data["5__kurtosis_x"] )))))) +
(((np.where(np.maximum(((data["flux_err_max"])), ((((data["flux_err_max"]) + (data["flux_err_max"]))))) > -1, ((data["flux_err_max"]) - (data["detected_flux_err_median"])), data["hostgal_photoz_err"] )) * 2.0)) +
(((((np.where(data["flux_d1_pb5"]>0, ((data["3__skewness_y"]) * 2.0), data["3__skewness_y"] )) + (data["flux_err_max"]))) + ((((data["5__skewness_x"]) < (data["3__skewness_y"]))*1.)))) +
(np.where(((data["distmod"]) - (data["hostgal_photoz"])) > -1, ((data["detected_mjd_diff"]) * ((-1.0*((((data["distmod"]) + (data["distmod"]))))))), data["distmod"] )) +
(((data["detected_flux_min"]) + (((((data["detected_flux_min"]) + (((((data["detected_flux_min"]) + (data["detected_mjd_diff"]))) * (data["4__skewness_x"]))))) - (data["4__skewness_x"]))))) +
(((data["4__fft_coefficient__coeff_1__attr__abs__y"]) * ((((-1.0*((data["detected_flux_median"])))) - ((((((-1.0*((data["0__fft_coefficient__coeff_0__attr__abs__y"])))) * (data["flux_w_mean"]))) * 2.0)))))) +
(((np.where(data["distmod"]<0, ((data["2__fft_coefficient__coeff_1__attr__abs__x"]) - (data["detected_flux_std"])), ((((data["flux_dif2"]) * 2.0)) - (data["hostgal_photoz"])) )) * 2.0)) +
((((((((data["2__kurtosis_x"]) - (((((data["2__skewness_x"]) - (data["distmod"]))) * 2.0)))) - (data["detected_flux_ratio_sq_skew"]))) + (data["0__fft_coefficient__coeff_0__attr__abs__x"]))/2.0)) +
(np.minimum(((data["0__fft_coefficient__coeff_0__attr__abs__y"])), ((((((data["flux_err_median"]) + (data["detected_flux_err_max"]))) * 2.0))))) +
(((((data["detected_flux_dif2"]) * (data["detected_flux_max"]))) - (np.where(data["detected_flux_dif2"]>0, data["detected_mjd_diff"], data["detected_mjd_diff"] )))) +
(((np.where(data["flux_d1_pb4"]>0, data["detected_flux_err_skew"], data["4__fft_coefficient__coeff_1__attr__abs__y"] )) + (data["3__skewness_y"]))) +
(np.where((((data["detected_flux_skew"]) + (data["flux_mean"]))/2.0)<0, ((data["detected_flux_skew"]) / 2.0), data["detected_flux_skew"] )) +
(np.where(data["hostgal_photoz"]<0, (-1.0*((data["3__kurtosis_x"]))), np.where(data["3__kurtosis_x"]<0, np.where(data["4__skewness_x"]<0, data["3__kurtosis_x"], data["hostgal_photoz"] ), data["hostgal_photoz"] ) )) +
(np.where(((data["mwebv"]) + ((((data["2__kurtosis_y"]) < (data["detected_flux_dif2"]))*1.)))<0, data["flux_dif2"], data["2__kurtosis_y"] )) +
(np.where(data["4__fft_coefficient__coeff_1__attr__abs__x"]>0, (((data["flux_w_mean"]) < (np.where(((data["flux_d0_pb2"]) * 2.0)>0, data["4__fft_coefficient__coeff_0__attr__abs__x"], data["detected_flux_err_min"] )))*1.), data["distmod"] )) +
(((((((((((data["5__fft_coefficient__coeff_1__attr__abs__y"]) - (data["1__fft_coefficient__coeff_0__attr__abs__x"]))) * 2.0)) * 2.0)) - (data["2__fft_coefficient__coeff_0__attr__abs__x"]))) - (data["1__fft_coefficient__coeff_0__attr__abs__x"]))) +
((((((((data["4__kurtosis_y"]) - (data["2__kurtosis_y"]))) - (((data["flux_min"]) - (data["2__fft_coefficient__coeff_1__attr__abs__x"]))))) + (((data["2__skewness_x"]) - (data["1__skewness_x"]))))/2.0)) +
(((((((np.where(np.where(data["2__fft_coefficient__coeff_1__attr__abs__y"]>0, data["flux_d0_pb3"], data["detected_flux_err_skew"] ) > -1, data["2__fft_coefficient__coeff_1__attr__abs__y"], data["detected_flux_err_skew"] )) - (data["flux_d0_pb0"]))) * 2.0)) * 2.0)) +
(((((data["1__kurtosis_x"]) - (data["3__kurtosis_x"]))) + (((((((data["1__kurtosis_x"]) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["1__fft_coefficient__coeff_1__attr__abs__x"]))))) +
(((((((data["1__skewness_x"]) + (((((data["detected_flux_min"]) * 2.0)) / 2.0)))) + (data["detected_flux_min"]))) + (((data["1__skewness_x"]) / 2.0)))) +
(((data["distmod"]) - (data["hostgal_photoz_err"]))) +
(((((((((data["5__fft_coefficient__coeff_1__attr__abs__x"]) - (data["1__fft_coefficient__coeff_0__attr__abs__x"]))) - (data["detected_mean"]))) + (data["5__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["flux_skew"]))) +
((((data["flux_ratio_sq_skew"]) + (((data["1__fft_coefficient__coeff_1__attr__abs__x"]) - (data["flux_min"]))))/2.0)) +
(((np.where(data["distmod"]>0, data["flux_dif2"], ((data["distmod"]) + (np.maximum(((((data["distmod"]) - (data["flux_diff"])))), ((data["flux_diff"]))))) )) * 2.0)) +
(((((((((data["5__skewness_x"]) - (data["detected_flux_std"]))) - (data["3__skewness_y"]))) - (data["3__skewness_y"]))) - (data["3__skewness_y"]))) +
((((data["flux_max"]) < (np.where(0.636620>0, ((data["1__fft_coefficient__coeff_1__attr__abs__x"]) - (data["1__fft_coefficient__coeff_0__attr__abs__x"])), data["5__kurtosis_x"] )))*1.)) +
(((((data["hostgal_photoz"]) - (((data["hostgal_photoz_err"]) * (data["hostgal_photoz"]))))) + (((data["0__skewness_y"]) + (((data["hostgal_photoz"]) - (data["hostgal_photoz"]))))))) +
(((((-1.0*((data["3__skewness_y"])))) > (data["detected_flux_std"]))*1.)) +
(np.where(data["0__kurtosis_x"]>0, ((data["5__skewness_x"]) - (data["0__fft_coefficient__coeff_1__attr__abs__x"])), ((((data["1__kurtosis_y"]) * (data["flux_err_max"]))) + (data["flux_err_max"])) )) +
(((np.where(data["5__kurtosis_x"]<0, data["flux_err_max"], np.where(data["0__fft_coefficient__coeff_1__attr__abs__x"]<0, np.tanh((((data["flux_median"]) * 2.0))), data["5__kurtosis_x"] ) )) * 2.0)) +
(((data["detected_flux_err_median"]) * (data["4__kurtosis_x"]))) +
(((((data["flux_err_min"]) - (data["flux_d0_pb0"]))) - ((-1.0*((((data["detected_flux_std"]) * (np.where(data["4__skewness_x"] > -1, data["flux_d0_pb4"], data["2__fft_coefficient__coeff_0__attr__abs__x"] ))))))))) +
(((data["flux_err_skew"]) - (np.where(((data["2__skewness_x"]) - (data["2__kurtosis_x"]))<0, np.where(data["1__fft_coefficient__coeff_1__attr__abs__x"]<0, data["2__skewness_x"], data["2__skewness_x"] ), data["1__fft_coefficient__coeff_1__attr__abs__x"] )))) +
(((np.where(data["flux_d0_pb0"]<0, data["flux_d0_pb0"], data["hostgal_photoz_err"] )) + (np.where(data["distmod"]<0, data["5__fft_coefficient__coeff_1__attr__abs__y"], data["hostgal_photoz_err"] )))) +
(((((data["distmod"]) * 2.0)) - (((((data["distmod"]) * 2.0)) * (np.where(data["1__skewness_y"]>0, data["flux_d0_pb0"], data["distmod"] )))))) +
(((((((data["flux_d0_pb5"]) * 2.0)) * (data["flux_d0_pb5"]))) * (np.maximum(((data["flux_d0_pb0"])), (((((data["1__skewness_x"]) < (np.tanh((data["3__skewness_x"]))))*1.))))))) +
(((((data["0__skewness_x"]) - (data["2__fft_coefficient__coeff_1__attr__abs__x"]))) - (data["3__fft_coefficient__coeff_0__attr__abs__y"]))) +
((((data["flux_dif2"]) + (((((data["hostgal_photoz"]) * (data["hostgal_photoz_err"]))) - (np.where(data["hostgal_photoz_err"]<0, data["1__fft_coefficient__coeff_1__attr__abs__y"], data["hostgal_photoz_err"] )))))/2.0)) +
(np.where(data["flux_by_flux_ratio_sq_sum"]<0, (0.98469996452331543), np.minimum(((np.minimum(((data["flux_d0_pb0"])), ((data["5__skewness_x"]))))), ((data["flux_max"]))) )) +
(((data["distmod"]) + (((((data["2__skewness_y"]) + (data["distmod"]))) - (data["hostgal_photoz"]))))) +
(((data["distmod"]) - (((np.tanh((data["1__fft_coefficient__coeff_1__attr__abs__x"]))) * (((data["1__fft_coefficient__coeff_1__attr__abs__x"]) * ((-1.0*((data["3__kurtosis_y"])))))))))) +
(((np.where(data["3__kurtosis_x"] > -1, np.where(data["1__kurtosis_x"] > -1, ((data["flux_median"]) * 2.0), ((data["detected_flux_err_min"]) * 2.0) ), data["detected_flux_err_min"] )) * 2.0)) +
(((((data["distmod"]) / 2.0)) - (data["4__kurtosis_y"]))) +
(((data["1__skewness_y"]) - (data["0__skewness_x"]))) +
(((((((data["hostgal_photoz"]) * ((-1.0*((data["hostgal_photoz_err"])))))) * 2.0)) + (np.where(data["4__kurtosis_x"] > -1, data["hostgal_photoz"], data["hostgal_photoz_err"] )))) +
(((data["4__skewness_x"]) * ((-1.0*((((data["flux_min"]) - ((-1.0*((data["detected_flux_dif2"]))))))))))) +
(np.where(data["detected_flux_ratio_sq_sum"]>0, data["0__kurtosis_x"], np.where(data["flux_d0_pb3"]>0, (-1.0*((data["detected_flux_ratio_sq_sum"]))), np.where(data["flux_d0_pb3"]>0, data["detected_flux_ratio_sq_sum"], data["flux_std"] ) ) )) +
(((data["1__skewness_x"]) + (((np.where(((data["1__skewness_y"]) - (data["flux_skew"]))<0, (-1.0*((data["5__fft_coefficient__coeff_1__attr__abs__y"]))), data["1__skewness_y"] )) * 2.0)))) +
(((((data["flux_w_mean"]) - (((data["flux_d0_pb0"]) * (((data["flux_d0_pb0"]) * 2.0)))))) * 2.0)) +
(((((data["2__fft_coefficient__coeff_0__attr__abs__y"]) - (data["0__fft_coefficient__coeff_1__attr__abs__x"]))) * (data["1__fft_coefficient__coeff_1__attr__abs__y"]))) +
(np.where(data["flux_d0_pb3"]>0, ((data["detected_flux_err_min"]) + (np.where(data["4__fft_coefficient__coeff_1__attr__abs__y"]<0, data["flux_err_max"], data["2__fft_coefficient__coeff_0__attr__abs__x"] ))), data["4__fft_coefficient__coeff_1__attr__abs__x"] )) +
(np.where(data["0__fft_coefficient__coeff_1__attr__abs__x"]>0, ((data["5__skewness_x"]) - (data["0__fft_coefficient__coeff_1__attr__abs__x"])), ((((data["detected_mjd_diff"]) - (data["5__kurtosis_x"]))) - (data["5__kurtosis_x"])) )) +
((-1.0*((((data["flux_err_min"]) + (np.where(np.where(data["flux_ratio_sq_skew"]>0, data["flux_err_min"], data["0__fft_coefficient__coeff_1__attr__abs__y"] )>0, data["detected_flux_ratio_sq_skew"], data["flux_ratio_sq_skew"] ))))))) +
(((((((data["hostgal_photoz_err"]) * (data["distmod"]))) * (data["distmod"]))) - (data["distmod"]))) +
(np.where(data["detected_flux_err_mean"] > -1, np.where(data["flux_err_mean"] > -1, np.where(data["mjd_size"] > -1, data["flux_err_mean"], data["3__fft_coefficient__coeff_1__attr__abs__y"] ), data["flux_err_mean"] ), data["mjd_size"] )) +
(((((data["5__kurtosis_y"]) + (((data["3__skewness_y"]) + (np.minimum(((((data["mjd_size"]) - (data["detected_mjd_diff"])))), ((data["flux_err_median"])))))))) + (data["mjd_size"]))))
cm = plt.cm.get_cmap('RdYlBu')
fig, axes = plt.subplots(1, 1, figsize=(15, 15))
sc = axes.scatter(GPx(full_train_ss),
                  GPy(full_train_ss),
                  alpha=1,
                  c=(y),
                  cmap=cm,
                  s=30)
cbar = fig.colorbar(sc, ax=axes)
cbar.set_label('Target')
_ = axes.set_title("Clustering colored by target")
myfilter = ((y==92)|(y==88))
cm = plt.cm.get_cmap('RdYlBu')
fig, axes = plt.subplots(1, 1, figsize=(15, 15))
sc = axes.scatter(GPx(full_train_ss[myfilter]),
                  GPy(full_train_ss[myfilter]),
                  alpha=1,
                  c=(y[myfilter]),
                  cmap=cm,
                  s=30)
cbar = fig.colorbar(sc, ax=axes)
cbar.set_label('Target')
_ = axes.set_title("Clustering colored by target")
myfilter = ((y==90)|(y==42))
cm = plt.cm.get_cmap('RdYlBu')
fig, axes = plt.subplots(1, 1, figsize=(15, 15))
sc = axes.scatter(GPx(full_train_ss[myfilter]),
                  GPy(full_train_ss[myfilter]),
                  alpha=1,
                  c=(y[myfilter]),
                  cmap=cm,
                  s=30)
cbar = fig.colorbar(sc, ax=axes)
cbar.set_label('Target')
_ = axes.set_title("Clustering colored by target")