%%time
import numpy as np
import pandas as pd
import glob, scipy
from sklearn import *
from catboost import CatBoostRegressor

#https://www.kaggle.com/andrekos/basic-feature-benchmark-with-quantiles
#Added quantiles per andrekos kernel
#Added Mean Absolute Deviation, unbiased kurtosis, unbiased skew per @interneuron suggestion

#https://www.kaggle.com/jsaguiar/baseline-with-abs-and-trend-features
#Added Trend Features from @jsaguiar
def add_trend_feature(arr, abs_values=False):
    idx = np.array(range(len(arr)))
    if abs_values:
        arr = np.abs(arr)
    lr = linear_model.LinearRegression(n_jobs=-1)
    lr.fit(idx.reshape(-1, 1), arr)
    return lr.coef_[0]

size = 10_000
train = pd.read_csv('../input/train.csv', iterator=True, chunksize=size, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
strain = []
group = []
for df in train:
    group.append(df)
    group = group[-int(150_000/size):]
    df2 = pd.concat(group)
    if len(df2)==150_000:
        seismic_chunk = df2['acoustic_data']
        fullmean = seismic_chunk.mean()
        fullstd = seismic_chunk.std()
        fullmax = seismic_chunk.max()
        fullmin = seismic_chunk.min()
        fullmad = seismic_chunk.mad()
        fullkurtosis = seismic_chunk.kurtosis()
        fullskew = seismic_chunk.skew()
        
        #https://www.kaggle.com/artgor/earthquakes-fe-more-features-and-samples
        sc_abs = np.abs(seismic_chunk)
        q01_abs = np.quantile(sc_abs,0.01)
        q05_abs = np.quantile(sc_abs,0.05)
        q95_abs = np.quantile(sc_abs,0.95)
        q99_abs = np.quantile(sc_abs,0.99)
        roll = seismic_chunk.rolling(100).std().dropna().values
        q01_roll = np.quantile(roll,0.01)
        q05_roll = np.quantile(roll,0.05)
        q95_roll = np.quantile(roll,0.95)
        q99_roll = np.quantile(roll,0.99)
        
        seismic_chunk = df2['acoustic_data'].values
        q01 = np.quantile(seismic_chunk,0.01)
        q05 = np.quantile(seismic_chunk,0.05)
        q95 = np.quantile(seismic_chunk,0.95)
        q99 = np.quantile(seismic_chunk,0.99)
        abs_max = np.abs(seismic_chunk).max()
        abs_mean = np.abs(seismic_chunk).mean()
        abs_std = np.abs(seismic_chunk).std()
        trend = add_trend_feature(seismic_chunk)
        abs_trend = add_trend_feature(seismic_chunk, abs_values=True)
        lastTTF = df2['time_to_failure'].values[-1]

        strain.append([fullmean, fullstd, fullmax, fullmin, fullmad, fullkurtosis, fullskew, q01, q05, q95, q99, q01_abs, q05_abs, q95_abs, q99_abs, q01_roll, q05_roll, q95_roll, q99_roll, abs_max, abs_mean, abs_std, trend, abs_trend, lastTTF])
strain = pd.DataFrame(strain, columns=['fullmean', 'fullstd', 'fullmax', 'fullmin', 'fullmad', 'fullkurtosis', 'fullskew', 'q01', 'q05', 'q95', 'q99', 'q01_abs', 'q05_abs', 'q95_abs', 'q99_abs', 'q01_roll', 'q05_roll', 'q95_roll', 'q99_roll', 'abs_max', 'abs_mean', 'abs_std', 'trend', 'abs_trend', 'time_to_failure'])

test = glob.glob('../input/test/**')
stest = []
for path in test:
    df = pd.read_csv(path, dtype={'acoustic_data': np.int16})
    seg_id = path.split('/')[-1].split('.')[0]
    seismic_chunk = df['acoustic_data']
    fullmean = seismic_chunk.mean()
    fullstd = seismic_chunk.std()
    fullmax = seismic_chunk.max()
    fullmin = seismic_chunk.min()
    fullmad = seismic_chunk.mad()
    fullkurtosis = seismic_chunk.kurtosis()
    fullskew = seismic_chunk.skew()

    #https://www.kaggle.com/artgor/earthquakes-fe-more-features-and-samples
    sc_abs = np.abs(seismic_chunk)
    q01_abs = np.quantile(sc_abs,0.01)
    q05_abs = np.quantile(sc_abs,0.05)
    q95_abs = np.quantile(sc_abs,0.95)
    q99_abs = np.quantile(sc_abs,0.99)
    roll = seismic_chunk.rolling(100).std().dropna().values
    q01_roll = np.quantile(roll,0.01)
    q05_roll = np.quantile(roll,0.05)
    q95_roll = np.quantile(roll,0.95)
    q99_roll = np.quantile(roll,0.99)
    
    seismic_chunk = df['acoustic_data'].values
    q01 = np.quantile(seismic_chunk,0.01)
    q05 = np.quantile(seismic_chunk,0.05)
    q95 = np.quantile(seismic_chunk,0.95)
    q99 = np.quantile(seismic_chunk,0.99)
    abs_max = np.abs(seismic_chunk).max()
    abs_mean = np.abs(seismic_chunk).mean()
    abs_std = np.abs(seismic_chunk).std()
    trend = add_trend_feature(seismic_chunk)
    abs_trend = add_trend_feature(seismic_chunk, abs_values=True)
    stest.append([seg_id, fullmean, fullstd, fullmax, fullmin, fullmad, fullkurtosis, fullskew, q01, q05, q95, q99, q01_abs, q05_abs, q95_abs, q99_abs, q01_roll, q05_roll, q95_roll, q99_roll, abs_max, abs_mean, abs_std, trend, abs_trend])
stest = pd.DataFrame(stest, columns=['seg_id', 'fullmean', 'fullstd', 'fullmax', 'fullmin', 'fullmad', 'fullkurtosis', 'fullskew', 'q01', 'q05', 'q95', 'q99', 'q01_abs', 'q05_abs', 'q95_abs', 'q99_abs', 'q01_roll', 'q05_roll', 'q95_roll', 'q99_roll', 'abs_max', 'abs_mean', 'abs_std', 'trend', 'abs_trend'])

sub = pd.read_csv('../input/sample_submission.csv')
print(strain.shape, stest.shape, sub.shape)
%%time
col = [c for c in strain.columns if c not in ['time_to_failure']]

#https://www.kaggle.com/inversion/basic-feature-benchmark
#https://www.kaggle.com/byfone/basic-feature-feat-catboost

scaler = preprocessing.StandardScaler()
scaled_train = scaler.fit_transform(strain[col])
scaled_test = scaler.transform(stest[col])

m = CatBoostRegressor(loss_function='MAE')
m.fit(scaled_train, strain['time_to_failure'], silent=True)
print(metrics.mean_absolute_error(strain['time_to_failure'], m.predict(scaled_train)))
stest['time_to_failure'] = m.predict(scaled_test)

stest[['seg_id','time_to_failure']].to_csv('submission.csv', index=False)
!rm -r catboost_info
