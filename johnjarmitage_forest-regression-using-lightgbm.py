import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm



from scipy.signal import butter, lfilter

from scipy.stats import kurtosis

from scipy.stats import skew



%matplotlib inline
rows = 150_000

n_rows = 100*rows

#train = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64}, nrows=n_rows)

train = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
rows = 150000

rowjump = 10000

njumps = int(15)

segments = int(np.floor(train.shape[0] / rows))

   

X_train = pd.DataFrame(index=range(segments), dtype=np.float64,

                       columns=['ave', 'std', 'max', 'min',

                                'dave', 'dstd', 'dmax', 'dmin',

                                'q01', 'q02', 'q03', 'q04', 'q05', 'q06', 'q07', 'q08', 'q09', 'q10',

                                'q90', 'q91', 'q92', 'q93', 'q94', 'q95', 'q96', 'q97', 'q98', 'q99',

                                'dq01', 'dq02', 'dq03', 'dq04', 'dq05', 'dq06', 'dq07', 'dq08', 'dq09', 'dq10',

                                'dq90', 'dq91', 'dq92', 'dq93', 'dq94', 'dq95', 'dq96', 'dq97', 'dq98', 'dq99',

                                'kurt', 'skew',

                                'dkurt', 'dskew',

                                'ave_roll_std_10', 'std_roll_std_10', 'max_roll_std_10', 'min_roll_std_10',

                                'ave_roll_ave_10', 'std_roll_ave_10', 'max_roll_ave_10', 'min_roll_ave_10',

                                'ave_roll_std_100', 'std_roll_std_100', 'max_roll_std_100', 'min_roll_std_100',

                                'ave_roll_ave_100', 'std_roll_ave_100', 'max_roll_ave_100', 'min_roll_ave_100',

                                'ave_roll_std_1000', 'std_roll_std_1000', 'max_roll_std_1000', 'min_roll_std_1000',

                                'ave_roll_ave_1000', 'std_roll_ave_1000', 'max_roll_ave_1000', 'min_roll_ave_1000',

                                'ave_roll_ave_10', 'std_roll_ave_10', 'max_roll_ave_10', 'min_roll_ave_10',

                                'ave_roll_std_100', 'std_roll_std_100', 'max_roll_std_100', 'min_roll_std_100',

                                'ave_roll_ave_100', 'std_roll_ave_100', 'max_roll_ave_100', 'min_roll_ave_100',

                                'ave_roll_std_1000', 'std_roll_std_1000', 'max_roll_std_1000', 'min_roll_std_1000',

                                'ave_roll_ave_1000', 'std_roll_ave_1000', 'max_roll_ave_1000', 'min_roll_ave_1000',

                                'q01_roll_std_10', 'q02_roll_std_10', 'q03_roll_std_10', 'q04_roll_std_10',

                                'q05_roll_std_10', 'q06_roll_std_10', 'q07_roll_std_10', 'q08_roll_std_10',

                                'q09_roll_std_10', 'q10_roll_std_10',

                                'q90_roll_std_10', 'q91_roll_std_10', 'q92_roll_std_10', 'q93_roll_std_10',

                                'q94_roll_std_10', 'q95_roll_std_10', 'q96_roll_std_10', 'q97_roll_std_10',

                                'q98_roll_std_10', 'q99_roll_std_10',

                                'q01_roll_std_100', 'q02_roll_std_100', 'q03_roll_std_100', 'q04_roll_std_100',

                                'q05_roll_std_100', 'q06_roll_std_100', 'q07_roll_std_100', 'q08_roll_std_100',

                                'q09_roll_std_100', 'q10_roll_std_100',

                                'q90_roll_std_100', 'q91_roll_std_100', 'q92_roll_std_100', 'q93_roll_std_100',

                                'q94_roll_std_100', 'q95_roll_std_100', 'q96_roll_std_100', 'q97_roll_std_100',

                                'q98_roll_std_100', 'q99_roll_std_100',

                                'q01_roll_std_1000', 'q02_roll_std_1000', 'q03_roll_std_1000', 'q04_roll_std_1000',

                                'q05_roll_std_1000', 'q06_roll_std_1000', 'q07_roll_std_1000', 'q08_roll_std_1000',

                                'q09_roll_std_1000', 'q10_roll_std_1000',

                                'q90_roll_std_1000', 'q91_roll_std_1000', 'q92_roll_std_1000', 'q93_roll_std_1000',

                                'q94_roll_std_1000', 'q95_roll_std_1000', 'q96_roll_std_1000', 'q97_roll_std_1000',

                                'q98_roll_std_1000', 'q99_roll_std_1000',

                                'av_change_abs_roll_std_10', 'av_change_rate_roll_std_10',

                                'av_change_abs_roll_ave_10', 'av_change_rate_roll_ave_10',

                                'av_change_abs_roll_std_100', 'av_change_rate_roll_std_100',

                                'av_change_abs_roll_ave_100', 'av_change_rate_roll_ave_100',

                                'av_change_abs_roll_std_1000', 'av_change_rate_roll_std_1000',

                                'av_change_abs_roll_ave_1000', 'av_change_rate_roll_ave_1000',

                                'sum',

                                'tq01', 'tq02', 'tq03', 'tq04', 'tq05', 'tq06', 'tq07', 'tq08', 'tq09', 'tq10',

                                'tq90', 'tq91', 'tq92', 'tq93', 'tq94', 'tq95', 'tq96', 'tq97', 'tq98', 'tq99',

                                'dtq01', 'dtq02', 'dtq03', 'dtq04', 'dtq05', 'dtq06', 'dtq07', 'dtq08', 'dtq09', 'dtq10',

                                'dtq90', 'dtq91', 'dtq92', 'dtq93', 'dtq94', 'dtq95', 'dtq96', 'dtq97', 'dtq98', 'dtq99'])

y_train_raw = pd.DataFrame(index=range(segments), dtype=np.float64,

                       columns=['time_to_failure'])



for njump in range(njumps) :

    for segment in tqdm(range(segments)):

        

        seg = train.iloc[segment*rows+njump*rowjump:segment*rows+rows+njump*rowjump]

        x_raw = seg['acoustic_data']

        x = seg['acoustic_data'].values

        dx = np.diff(x)

        y = seg['time_to_failure'].values[-1]

    

        y_train_raw.loc[segment+njump*segments, 'time_to_failure'] = y

    

        # simple features

    

        X_train.loc[segment+njump*segments, 'ave'] = x.mean()

        X_train.loc[segment+njump*segments, 'std'] = x.std()

        X_train.loc[segment+njump*segments, 'max'] = x.max()

        X_train.loc[segment+njump*segments, 'min'] = x.min()



        X_train.loc[segment+njump*segments, 'dave'] = dx.mean()

        X_train.loc[segment+njump*segments, 'dstd'] = dx.std()

        X_train.loc[segment+njump*segments, 'dmax'] = dx.max()

        X_train.loc[segment+njump*segments, 'dmin'] = dx.min()



        # quartiles



        X_train.loc[segment+njump*segments, 'q01'] = np.quantile(x,0.01)

        X_train.loc[segment+njump*segments, 'q02'] = np.quantile(x,0.02)

        X_train.loc[segment+njump*segments, 'q03'] = np.quantile(x,0.03)

        X_train.loc[segment+njump*segments, 'q04'] = np.quantile(x,0.04)

        X_train.loc[segment+njump*segments, 'q05'] = np.quantile(x,0.05)

        X_train.loc[segment+njump*segments, 'q06'] = np.quantile(x,0.06)

        X_train.loc[segment+njump*segments, 'q07'] = np.quantile(x,0.07)

        X_train.loc[segment+njump*segments, 'q08'] = np.quantile(x,0.08)

        X_train.loc[segment+njump*segments, 'q09'] = np.quantile(x,0.09)

        X_train.loc[segment+njump*segments, 'q10'] = np.quantile(x,0.10)

        X_train.loc[segment+njump*segments, 'q90'] = np.quantile(x,0.90)

        X_train.loc[segment+njump*segments, 'q91'] = np.quantile(x,0.91)

        X_train.loc[segment+njump*segments, 'q92'] = np.quantile(x,0.92)

        X_train.loc[segment+njump*segments, 'q93'] = np.quantile(x,0.93)

        X_train.loc[segment+njump*segments, 'q94'] = np.quantile(x,0.94)

        X_train.loc[segment+njump*segments, 'q95'] = np.quantile(x,0.95)

        X_train.loc[segment+njump*segments, 'q96'] = np.quantile(x,0.96)

        X_train.loc[segment+njump*segments, 'q97'] = np.quantile(x,0.97)

        X_train.loc[segment+njump*segments, 'q98'] = np.quantile(x,0.98)

        X_train.loc[segment+njump*segments, 'q99'] = np.quantile(x,0.99)



        X_train.loc[segment+njump*segments, 'dq01'] = np.quantile(dx,0.01)

        X_train.loc[segment+njump*segments, 'dq02'] = np.quantile(dx,0.02)

        X_train.loc[segment+njump*segments, 'dq03'] = np.quantile(dx,0.03)

        X_train.loc[segment+njump*segments, 'dq04'] = np.quantile(dx,0.04)

        X_train.loc[segment+njump*segments, 'dq05'] = np.quantile(dx,0.05)

        X_train.loc[segment+njump*segments, 'dq06'] = np.quantile(dx,0.06)

        X_train.loc[segment+njump*segments, 'dq07'] = np.quantile(dx,0.07)

        X_train.loc[segment+njump*segments, 'dq08'] = np.quantile(dx,0.08)

        X_train.loc[segment+njump*segments, 'dq09'] = np.quantile(dx,0.09)

        X_train.loc[segment+njump*segments, 'dq10'] = np.quantile(dx,0.10)

        X_train.loc[segment+njump*segments, 'dq90'] = np.quantile(dx,0.90)

        X_train.loc[segment+njump*segments, 'dq91'] = np.quantile(dx,0.91)

        X_train.loc[segment+njump*segments, 'dq92'] = np.quantile(dx,0.92)

        X_train.loc[segment+njump*segments, 'dq93'] = np.quantile(dx,0.93)

        X_train.loc[segment+njump*segments, 'dq94'] = np.quantile(dx,0.94)

        X_train.loc[segment+njump*segments, 'dq95'] = np.quantile(dx,0.95)

        X_train.loc[segment+njump*segments, 'dq96'] = np.quantile(dx,0.96)

        X_train.loc[segment+njump*segments, 'dq97'] = np.quantile(dx,0.97)

        X_train.loc[segment+njump*segments, 'dq98'] = np.quantile(dx,0.98)

        X_train.loc[segment+njump*segments, 'dq99'] = np.quantile(dx,0.99)

        # extras



        X_train.loc[segment+njump*segments, 'kurt'] = kurtosis(x)

        X_train.loc[segment+njump*segments, 'skew'] = skew(x)



        X_train.loc[segment+njump*segments, 'dkurt'] = kurtosis(dx)

        X_train.loc[segment+njump*segments, 'dskew'] = skew(dx)



        # rolling mean, std, quantiles



        for windows in [10,100,1000]:

            x_roll_std = x_raw.rolling(windows).std().dropna().values

            x_roll_mean = x_raw.rolling(windows).mean().dropna().values



            X_train.loc[segment+njump*segments, 'ave_roll_std_' + str(windows)] = x_roll_std.mean()

            X_train.loc[segment+njump*segments, 'std_roll_std_' + str(windows)] = x_roll_std.std()

            X_train.loc[segment+njump*segments, 'max_roll_std_' + str(windows)] = x_roll_std.max()

            X_train.loc[segment+njump*segments, 'min_roll_std_' + str(windows)] = x_roll_std.min()



            X_train.loc[segment+njump*segments, 'ave_roll_ave_' + str(windows)] = x_roll_mean.mean()

            X_train.loc[segment+njump*segments, 'std_roll_ave_' + str(windows)] = x_roll_mean.std()

            X_train.loc[segment+njump*segments, 'max_roll_ave_' + str(windows)] = x_roll_mean.max()

            X_train.loc[segment+njump*segments, 'min_roll_ave_' + str(windows)] = x_roll_mean.min()



            X_train.loc[segment+njump*segments, 'q01_roll_std_' + str(windows)] = np.quantile(x_roll_std,0.01)

            X_train.loc[segment+njump*segments, 'q02_roll_std_' + str(windows)] = np.quantile(x_roll_std,0.02)

            X_train.loc[segment+njump*segments, 'q03_roll_std_' + str(windows)] = np.quantile(x_roll_std,0.03)

            X_train.loc[segment+njump*segments, 'q04_roll_std_' + str(windows)] = np.quantile(x_roll_std,0.04)

            X_train.loc[segment+njump*segments, 'q05_roll_std_' + str(windows)] = np.quantile(x_roll_std,0.05)

            X_train.loc[segment+njump*segments, 'q06_roll_std_' + str(windows)] = np.quantile(x_roll_std,0.06)

            X_train.loc[segment+njump*segments, 'q07_roll_std_' + str(windows)] = np.quantile(x_roll_std,0.07)

            X_train.loc[segment+njump*segments, 'q08_roll_std_' + str(windows)] = np.quantile(x_roll_std,0.08)

            X_train.loc[segment+njump*segments, 'q09_roll_std_' + str(windows)] = np.quantile(x_roll_std,0.09)

            X_train.loc[segment+njump*segments, 'q10_roll_std_' + str(windows)] = np.quantile(x_roll_std,0.10)

            X_train.loc[segment+njump*segments, 'q90_roll_std_' + str(windows)] = np.quantile(x_roll_std,0.90)

            X_train.loc[segment+njump*segments, 'q91_roll_std_' + str(windows)] = np.quantile(x_roll_std,0.91)

            X_train.loc[segment+njump*segments, 'q92_roll_std_' + str(windows)] = np.quantile(x_roll_std,0.92)

            X_train.loc[segment+njump*segments, 'q93_roll_std_' + str(windows)] = np.quantile(x_roll_std,0.93)

            X_train.loc[segment+njump*segments, 'q94_roll_std_' + str(windows)] = np.quantile(x_roll_std,0.94)

            X_train.loc[segment+njump*segments, 'q95_roll_std_' + str(windows)] = np.quantile(x_roll_std,0.95)

            X_train.loc[segment+njump*segments, 'q96_roll_std_' + str(windows)] = np.quantile(x_roll_std,0.96)

            X_train.loc[segment+njump*segments, 'q97_roll_std_' + str(windows)] = np.quantile(x_roll_std,0.97)

            X_train.loc[segment+njump*segments, 'q98_roll_std_' + str(windows)] = np.quantile(x_roll_std,0.98)

            X_train.loc[segment+njump*segments, 'q99_roll_std_' + str(windows)] = np.quantile(x_roll_std,0.99)



            X_train.loc[segment+njump*segments, 'q01_roll_ave_' + str(windows)] = np.quantile(x_roll_mean,0.01)

            X_train.loc[segment+njump*segments, 'q02_roll_ave_' + str(windows)] = np.quantile(x_roll_mean,0.02)

            X_train.loc[segment+njump*segments, 'q03_roll_ave_' + str(windows)] = np.quantile(x_roll_mean,0.03)

            X_train.loc[segment+njump*segments, 'q04_roll_ave_' + str(windows)] = np.quantile(x_roll_mean,0.04)

            X_train.loc[segment+njump*segments, 'q05_roll_ave_' + str(windows)] = np.quantile(x_roll_mean,0.05)

            X_train.loc[segment+njump*segments, 'q06_roll_ave_' + str(windows)] = np.quantile(x_roll_mean,0.06)

            X_train.loc[segment+njump*segments, 'q07_roll_ave_' + str(windows)] = np.quantile(x_roll_mean,0.07)

            X_train.loc[segment+njump*segments, 'q08_roll_ave_' + str(windows)] = np.quantile(x_roll_mean,0.08)

            X_train.loc[segment+njump*segments, 'q09_roll_ave_' + str(windows)] = np.quantile(x_roll_mean,0.09)

            X_train.loc[segment+njump*segments, 'q10_roll_ave_' + str(windows)] = np.quantile(x_roll_mean,0.10)

            X_train.loc[segment+njump*segments, 'q90_roll_ave_' + str(windows)] = np.quantile(x_roll_mean,0.90)

            X_train.loc[segment+njump*segments, 'q91_roll_ave_' + str(windows)] = np.quantile(x_roll_mean,0.91)

            X_train.loc[segment+njump*segments, 'q92_roll_ave_' + str(windows)] = np.quantile(x_roll_mean,0.92)

            X_train.loc[segment+njump*segments, 'q93_roll_ave_' + str(windows)] = np.quantile(x_roll_mean,0.93)

            X_train.loc[segment+njump*segments, 'q94_roll_ave_' + str(windows)] = np.quantile(x_roll_mean,0.94)

            X_train.loc[segment+njump*segments, 'q95_roll_ave_' + str(windows)] = np.quantile(x_roll_mean,0.95)

            X_train.loc[segment+njump*segments, 'q96_roll_ave_' + str(windows)] = np.quantile(x_roll_mean,0.96)

            X_train.loc[segment+njump*segments, 'q97_roll_ave_' + str(windows)] = np.quantile(x_roll_mean,0.97)

            X_train.loc[segment+njump*segments, 'q98_roll_ave_' + str(windows)] = np.quantile(x_roll_mean,0.98)

            X_train.loc[segment+njump*segments, 'q99_roll_ave_' + str(windows)] = np.quantile(x_roll_mean,0.99)



            X_train.loc[segment+njump*segments, 'av_change_abs_roll_std_' + str(windows)] = np.mean(np.diff(x_roll_std))

            X_train.loc[segment+njump*segments, 'av_change_rate_roll_std_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])



            X_train.loc[segment+njump*segments, 'av_change_abs_roll_ave_' + str(windows)] = np.mean(np.diff(x_roll_mean))

            X_train.loc[segment+njump*segments, 'av_change_rate_roll_ave_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])



        # accumulated energy



        X_train.loc[segment+njump*segments, 'sum'] = sum(abs(x))



        # threshold



        X_train.loc[segment+njump*segments, 'tq01'] = np.count_nonzero(x < np.quantile(x,0.01))

        X_train.loc[segment+njump*segments, 'tq02'] = np.count_nonzero(x < np.quantile(x,0.02))

        X_train.loc[segment+njump*segments, 'tq03'] = np.count_nonzero(x > np.quantile(x,0.03))

        X_train.loc[segment+njump*segments, 'tq04'] = np.count_nonzero(x > np.quantile(x,0.04))

        X_train.loc[segment+njump*segments, 'tq05'] = np.count_nonzero(x < np.quantile(x,0.05))

        X_train.loc[segment+njump*segments, 'tq06'] = np.count_nonzero(x < np.quantile(x,0.06))

        X_train.loc[segment+njump*segments, 'tq07'] = np.count_nonzero(x > np.quantile(x,0.07))

        X_train.loc[segment+njump*segments, 'tq08'] = np.count_nonzero(x > np.quantile(x,0.08))

        X_train.loc[segment+njump*segments, 'tq09'] = np.count_nonzero(x < np.quantile(x,0.09))

        X_train.loc[segment+njump*segments, 'tq10'] = np.count_nonzero(x < np.quantile(x,0.10))

        X_train.loc[segment+njump*segments, 'tq90'] = np.count_nonzero(x < np.quantile(x,0.90))

        X_train.loc[segment+njump*segments, 'tq91'] = np.count_nonzero(x > np.quantile(x,0.91))

        X_train.loc[segment+njump*segments, 'tq92'] = np.count_nonzero(x > np.quantile(x,0.92))

        X_train.loc[segment+njump*segments, 'tq93'] = np.count_nonzero(x < np.quantile(x,0.93))

        X_train.loc[segment+njump*segments, 'tq94'] = np.count_nonzero(x < np.quantile(x,0.94))

        X_train.loc[segment+njump*segments, 'tq95'] = np.count_nonzero(x > np.quantile(x,0.95))

        X_train.loc[segment+njump*segments, 'tq96'] = np.count_nonzero(x > np.quantile(x,0.96))

        X_train.loc[segment+njump*segments, 'tq97'] = np.count_nonzero(x < np.quantile(x,0.97))

        X_train.loc[segment+njump*segments, 'tq98'] = np.count_nonzero(x < np.quantile(x,0.98))

        X_train.loc[segment+njump*segments, 'tq99'] = np.count_nonzero(x > np.quantile(x,0.99))



        X_train.loc[segment+njump*segments, 'dtq01'] = np.count_nonzero(x < np.quantile(dx,0.01))

        X_train.loc[segment+njump*segments, 'dtq02'] = np.count_nonzero(x < np.quantile(dx,0.02))

        X_train.loc[segment+njump*segments, 'dtq03'] = np.count_nonzero(x > np.quantile(dx,0.03))

        X_train.loc[segment+njump*segments, 'dtq04'] = np.count_nonzero(x > np.quantile(dx,0.04))

        X_train.loc[segment+njump*segments, 'dtq05'] = np.count_nonzero(x < np.quantile(dx,0.05))

        X_train.loc[segment+njump*segments, 'dtq06'] = np.count_nonzero(x < np.quantile(dx,0.06))

        X_train.loc[segment+njump*segments, 'dtq07'] = np.count_nonzero(x > np.quantile(dx,0.07))

        X_train.loc[segment+njump*segments, 'dtq08'] = np.count_nonzero(x > np.quantile(dx,0.08))

        X_train.loc[segment+njump*segments, 'dtq09'] = np.count_nonzero(x < np.quantile(dx,0.09))

        X_train.loc[segment+njump*segments, 'dtq10'] = np.count_nonzero(x < np.quantile(dx,0.10))

        X_train.loc[segment+njump*segments, 'dtq90'] = np.count_nonzero(x < np.quantile(dx,0.90))

        X_train.loc[segment+njump*segments, 'dtq91'] = np.count_nonzero(x > np.quantile(dx,0.91))

        X_train.loc[segment+njump*segments, 'dtq92'] = np.count_nonzero(x > np.quantile(dx,0.92))

        X_train.loc[segment+njump*segments, 'dtq93'] = np.count_nonzero(x < np.quantile(dx,0.93))

        X_train.loc[segment+njump*segments, 'dtq94'] = np.count_nonzero(x < np.quantile(dx,0.94))

        X_train.loc[segment+njump*segments, 'dtq95'] = np.count_nonzero(x > np.quantile(dx,0.95))

        X_train.loc[segment+njump*segments, 'dtq96'] = np.count_nonzero(x > np.quantile(dx,0.96))

        X_train.loc[segment+njump*segments, 'dtq97'] = np.count_nonzero(x < np.quantile(dx,0.97))

        X_train.loc[segment+njump*segments, 'dtq98'] = np.count_nonzero(x < np.quantile(dx,0.98))

        X_train.loc[segment+njump*segments, 'dtq99'] = np.count_nonzero(x > np.quantile(dx,0.99))

    
X_train.describe()
submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')

X_submit = pd.DataFrame(index=submission.index, dtype=np.float64, columns=X_train.columns)



for i, seg_id in enumerate(tqdm(X_submit.index)):

    seg = pd.read_csv('../input/test/' + seg_id + '.csv')

    x_raw = seg['acoustic_data']

    x = seg['acoustic_data'].values

    dx = np.diff(x)

    

    # simple features

    

    X_submit.loc[seg_id, 'ave'] = x.mean()

    X_submit.loc[seg_id, 'std'] = x.std()

    X_submit.loc[seg_id, 'max'] = x.max()

    X_submit.loc[seg_id, 'min'] = x.min()

        

    X_submit.loc[seg_id, 'dave'] = dx.mean()

    X_submit.loc[seg_id, 'dstd'] = dx.std()

    X_submit.loc[seg_id, 'dmax'] = dx.max()

    X_submit.loc[seg_id, 'dmin'] = dx.min()

     

    # quartiles

    

    X_submit.loc[seg_id, 'q01'] = np.quantile(x,0.01)

    X_submit.loc[seg_id, 'q02'] = np.quantile(x,0.02)

    X_submit.loc[seg_id, 'q03'] = np.quantile(x,0.03)

    X_submit.loc[seg_id, 'q04'] = np.quantile(x,0.04)

    X_submit.loc[seg_id, 'q05'] = np.quantile(x,0.05)

    X_submit.loc[seg_id, 'q06'] = np.quantile(x,0.06)

    X_submit.loc[seg_id, 'q07'] = np.quantile(x,0.07)

    X_submit.loc[seg_id, 'q08'] = np.quantile(x,0.08)

    X_submit.loc[seg_id, 'q09'] = np.quantile(x,0.09)

    X_submit.loc[seg_id, 'q10'] = np.quantile(x,0.10)

    X_submit.loc[seg_id, 'q90'] = np.quantile(x,0.90)

    X_submit.loc[seg_id, 'q91'] = np.quantile(x,0.91)

    X_submit.loc[seg_id, 'q92'] = np.quantile(x,0.92)

    X_submit.loc[seg_id, 'q93'] = np.quantile(x,0.93)

    X_submit.loc[seg_id, 'q94'] = np.quantile(x,0.94)

    X_submit.loc[seg_id, 'q95'] = np.quantile(x,0.95)

    X_submit.loc[seg_id, 'q96'] = np.quantile(x,0.96)

    X_submit.loc[seg_id, 'q97'] = np.quantile(x,0.97)

    X_submit.loc[seg_id, 'q98'] = np.quantile(x,0.98)

    X_submit.loc[seg_id, 'q99'] = np.quantile(x,0.99)

        

    X_submit.loc[seg_id, 'dq01'] = np.quantile(dx,0.01)

    X_submit.loc[seg_id, 'dq02'] = np.quantile(dx,0.02)

    X_submit.loc[seg_id, 'dq03'] = np.quantile(dx,0.03)

    X_submit.loc[seg_id, 'dq04'] = np.quantile(dx,0.04)

    X_submit.loc[seg_id, 'dq05'] = np.quantile(dx,0.05)

    X_submit.loc[seg_id, 'dq06'] = np.quantile(dx,0.06)

    X_submit.loc[seg_id, 'dq07'] = np.quantile(dx,0.07)

    X_submit.loc[seg_id, 'dq08'] = np.quantile(dx,0.08)

    X_submit.loc[seg_id, 'dq09'] = np.quantile(dx,0.09)

    X_submit.loc[seg_id, 'dq10'] = np.quantile(dx,0.10)

    X_submit.loc[seg_id, 'dq90'] = np.quantile(dx,0.90)

    X_submit.loc[seg_id, 'dq91'] = np.quantile(dx,0.91)

    X_submit.loc[seg_id, 'dq92'] = np.quantile(dx,0.92)

    X_submit.loc[seg_id, 'dq93'] = np.quantile(dx,0.93)

    X_submit.loc[seg_id, 'dq94'] = np.quantile(dx,0.94)

    X_submit.loc[seg_id, 'dq95'] = np.quantile(dx,0.95)

    X_submit.loc[seg_id, 'dq96'] = np.quantile(dx,0.96)

    X_submit.loc[seg_id, 'dq97'] = np.quantile(dx,0.97)

    X_submit.loc[seg_id, 'dq98'] = np.quantile(dx,0.98)

    X_submit.loc[seg_id, 'dq99'] = np.quantile(dx,0.99)

    

    # extras

    

    X_submit.loc[seg_id, 'kurt'] = kurtosis(x)

    X_submit.loc[seg_id, 'skew'] = skew(x)

    

    X_submit.loc[seg_id, 'dkurt'] = kurtosis(dx)

    X_submit.loc[seg_id, 'dskew'] = skew(dx)

    

    # rolling mean and std

    

    for windows in [10,100,1000]:

        x_roll_std = x_raw.rolling(windows).std().dropna().values

        x_roll_mean = x_raw.rolling(windows).mean().dropna().values

        

        X_submit.loc[seg_id, 'ave_roll_std_' + str(windows)] = x_roll_std.mean()

        X_submit.loc[seg_id, 'std_roll_std_' + str(windows)] = x_roll_std.std()

        X_submit.loc[seg_id, 'max_roll_std_' + str(windows)] = x_roll_std.max()

        X_submit.loc[seg_id, 'min_roll_std_' + str(windows)] = x_roll_std.min()

        

        X_submit.loc[seg_id, 'ave_roll_ave_' + str(windows)] = x_roll_mean.mean()

        X_submit.loc[seg_id, 'std_roll_ave_' + str(windows)] = x_roll_mean.std()

        X_submit.loc[seg_id, 'max_roll_ave_' + str(windows)] = x_roll_mean.max()

        X_submit.loc[seg_id, 'min_roll_ave_' + str(windows)] = x_roll_mean.min()

        

        X_submit.loc[seg_id, 'q01_roll_std_' + str(windows)] = np.quantile(x_roll_std,0.01)

        X_submit.loc[seg_id, 'q02_roll_std_' + str(windows)] = np.quantile(x_roll_std,0.02)

        X_submit.loc[seg_id, 'q03_roll_std_' + str(windows)] = np.quantile(x_roll_std,0.03)

        X_submit.loc[seg_id, 'q04_roll_std_' + str(windows)] = np.quantile(x_roll_std,0.04)

        X_submit.loc[seg_id, 'q05_roll_std_' + str(windows)] = np.quantile(x_roll_std,0.05)

        X_submit.loc[seg_id, 'q06_roll_std_' + str(windows)] = np.quantile(x_roll_std,0.06)

        X_submit.loc[seg_id, 'q07_roll_std_' + str(windows)] = np.quantile(x_roll_std,0.07)

        X_submit.loc[seg_id, 'q08_roll_std_' + str(windows)] = np.quantile(x_roll_std,0.08)

        X_submit.loc[seg_id, 'q09_roll_std_' + str(windows)] = np.quantile(x_roll_std,0.09)

        X_submit.loc[seg_id, 'q10_roll_std_' + str(windows)] = np.quantile(x_roll_std,0.10)

        X_submit.loc[seg_id, 'q90_roll_std_' + str(windows)] = np.quantile(x_roll_std,0.90)

        X_submit.loc[seg_id, 'q91_roll_std_' + str(windows)] = np.quantile(x_roll_std,0.91)

        X_submit.loc[seg_id, 'q92_roll_std_' + str(windows)] = np.quantile(x_roll_std,0.92)

        X_submit.loc[seg_id, 'q93_roll_std_' + str(windows)] = np.quantile(x_roll_std,0.93)

        X_submit.loc[seg_id, 'q94_roll_std_' + str(windows)] = np.quantile(x_roll_std,0.94)

        X_submit.loc[seg_id, 'q95_roll_std_' + str(windows)] = np.quantile(x_roll_std,0.95)

        X_submit.loc[seg_id, 'q96_roll_std_' + str(windows)] = np.quantile(x_roll_std,0.96)

        X_submit.loc[seg_id, 'q97_roll_std_' + str(windows)] = np.quantile(x_roll_std,0.97)

        X_submit.loc[seg_id, 'q98_roll_std_' + str(windows)] = np.quantile(x_roll_std,0.98)

        X_submit.loc[seg_id, 'q99_roll_std_' + str(windows)] = np.quantile(x_roll_std,0.99)

        

        X_submit.loc[seg_id, 'q01_roll_ave_' + str(windows)] = np.quantile(x_roll_mean,0.01)

        X_submit.loc[seg_id, 'q02_roll_ave_' + str(windows)] = np.quantile(x_roll_mean,0.02)

        X_submit.loc[seg_id, 'q03_roll_ave_' + str(windows)] = np.quantile(x_roll_mean,0.03)

        X_submit.loc[seg_id, 'q04_roll_ave_' + str(windows)] = np.quantile(x_roll_mean,0.04)

        X_submit.loc[seg_id, 'q05_roll_ave_' + str(windows)] = np.quantile(x_roll_mean,0.05)

        X_submit.loc[seg_id, 'q06_roll_ave_' + str(windows)] = np.quantile(x_roll_mean,0.06)

        X_submit.loc[seg_id, 'q07_roll_ave_' + str(windows)] = np.quantile(x_roll_mean,0.07)

        X_submit.loc[seg_id, 'q08_roll_ave_' + str(windows)] = np.quantile(x_roll_mean,0.08)

        X_submit.loc[seg_id, 'q09_roll_ave_' + str(windows)] = np.quantile(x_roll_mean,0.09)

        X_submit.loc[seg_id, 'q10_roll_ave_' + str(windows)] = np.quantile(x_roll_mean,0.10)

        X_submit.loc[seg_id, 'q90_roll_ave_' + str(windows)] = np.quantile(x_roll_mean,0.90)

        X_submit.loc[seg_id, 'q91_roll_ave_' + str(windows)] = np.quantile(x_roll_mean,0.91)

        X_submit.loc[seg_id, 'q92_roll_ave_' + str(windows)] = np.quantile(x_roll_mean,0.92)

        X_submit.loc[seg_id, 'q93_roll_ave_' + str(windows)] = np.quantile(x_roll_mean,0.93)

        X_submit.loc[seg_id, 'q94_roll_ave_' + str(windows)] = np.quantile(x_roll_mean,0.94)

        X_submit.loc[seg_id, 'q95_roll_ave_' + str(windows)] = np.quantile(x_roll_mean,0.95)

        X_submit.loc[seg_id, 'q96_roll_ave_' + str(windows)] = np.quantile(x_roll_mean,0.96)

        X_submit.loc[seg_id, 'q97_roll_ave_' + str(windows)] = np.quantile(x_roll_mean,0.97)

        X_submit.loc[seg_id, 'q98_roll_ave_' + str(windows)] = np.quantile(x_roll_mean,0.98)

        X_submit.loc[seg_id, 'q99_roll_ave_' + str(windows)] = np.quantile(x_roll_mean,0.99)

        

        X_submit.loc[seg_id, 'av_change_abs_roll_std_' + str(windows)] = np.mean(np.diff(x_roll_std))

        X_submit.loc[seg_id, 'av_change_rate_roll_std_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])

        

        X_submit.loc[seg_id, 'av_change_abs_roll_ave_' + str(windows)] = np.mean(np.diff(x_roll_mean))

        X_submit.loc[seg_id, 'av_change_rate_roll_ave_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])

                

    # accumulated energy

    

    X_submit.loc[seg_id, 'sum'] = sum(abs(x))

    

    # threshold

    

    X_submit.loc[seg_id, 'tq01'] = np.count_nonzero(x < np.quantile(x,0.01))

    X_submit.loc[seg_id, 'tq02'] = np.count_nonzero(x < np.quantile(x,0.02))

    X_submit.loc[seg_id, 'tq03'] = np.count_nonzero(x > np.quantile(x,0.03))

    X_submit.loc[seg_id, 'tq04'] = np.count_nonzero(x > np.quantile(x,0.04))

    X_submit.loc[seg_id, 'tq05'] = np.count_nonzero(x < np.quantile(x,0.05))

    X_submit.loc[seg_id, 'tq06'] = np.count_nonzero(x < np.quantile(x,0.06))

    X_submit.loc[seg_id, 'tq07'] = np.count_nonzero(x > np.quantile(x,0.07))

    X_submit.loc[seg_id, 'tq08'] = np.count_nonzero(x > np.quantile(x,0.08))

    X_submit.loc[seg_id, 'tq09'] = np.count_nonzero(x < np.quantile(x,0.09))

    X_submit.loc[seg_id, 'tq10'] = np.count_nonzero(x < np.quantile(x,0.10))

    X_submit.loc[seg_id, 'tq90'] = np.count_nonzero(x < np.quantile(x,0.90))

    X_submit.loc[seg_id, 'tq91'] = np.count_nonzero(x > np.quantile(x,0.91))

    X_submit.loc[seg_id, 'tq92'] = np.count_nonzero(x > np.quantile(x,0.92))

    X_submit.loc[seg_id, 'tq93'] = np.count_nonzero(x < np.quantile(x,0.93))

    X_submit.loc[seg_id, 'tq94'] = np.count_nonzero(x < np.quantile(x,0.94))

    X_submit.loc[seg_id, 'tq95'] = np.count_nonzero(x > np.quantile(x,0.95))

    X_submit.loc[seg_id, 'tq96'] = np.count_nonzero(x > np.quantile(x,0.96))

    X_submit.loc[seg_id, 'tq97'] = np.count_nonzero(x < np.quantile(x,0.97))

    X_submit.loc[seg_id, 'tq98'] = np.count_nonzero(x < np.quantile(x,0.98))

    X_submit.loc[seg_id, 'tq99'] = np.count_nonzero(x > np.quantile(x,0.99))

    

    X_submit.loc[seg_id, 'dtq01'] = np.count_nonzero(x < np.quantile(dx,0.01))

    X_submit.loc[seg_id, 'dtq02'] = np.count_nonzero(x < np.quantile(dx,0.02))

    X_submit.loc[seg_id, 'dtq03'] = np.count_nonzero(x > np.quantile(dx,0.03))

    X_submit.loc[seg_id, 'dtq04'] = np.count_nonzero(x > np.quantile(dx,0.04))

    X_submit.loc[seg_id, 'dtq05'] = np.count_nonzero(x < np.quantile(dx,0.05))

    X_submit.loc[seg_id, 'dtq06'] = np.count_nonzero(x < np.quantile(dx,0.06))

    X_submit.loc[seg_id, 'dtq07'] = np.count_nonzero(x > np.quantile(dx,0.07))

    X_submit.loc[seg_id, 'dtq08'] = np.count_nonzero(x > np.quantile(dx,0.08))

    X_submit.loc[seg_id, 'dtq09'] = np.count_nonzero(x < np.quantile(dx,0.09))

    X_submit.loc[seg_id, 'dtq10'] = np.count_nonzero(x < np.quantile(dx,0.10))

    X_submit.loc[seg_id, 'dtq90'] = np.count_nonzero(x < np.quantile(dx,0.90))

    X_submit.loc[seg_id, 'dtq91'] = np.count_nonzero(x > np.quantile(dx,0.91))

    X_submit.loc[seg_id, 'dtq92'] = np.count_nonzero(x > np.quantile(dx,0.92))

    X_submit.loc[seg_id, 'dtq93'] = np.count_nonzero(x < np.quantile(dx,0.93))

    X_submit.loc[seg_id, 'dtq94'] = np.count_nonzero(x < np.quantile(dx,0.94))

    X_submit.loc[seg_id, 'dtq95'] = np.count_nonzero(x > np.quantile(dx,0.95))

    X_submit.loc[seg_id, 'dtq96'] = np.count_nonzero(x > np.quantile(dx,0.96))

    X_submit.loc[seg_id, 'dtq97'] = np.count_nonzero(x < np.quantile(dx,0.97))

    X_submit.loc[seg_id, 'dtq98'] = np.count_nonzero(x < np.quantile(dx,0.98))

    X_submit.loc[seg_id, 'dtq99'] = np.count_nonzero(x > np.quantile(dx,0.99))

    
import multiprocessing

multiprocessing.cpu_count()
import lightgbm as lgb

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV



# fixed parameters

params = {'boosting_type': 'gbdt',

          'objective': 'regression_l1',

          'nthread': 4,

          'num_leaves': 64,

          'learning_rate': 0.05,

          'max_bin': 512,

          'colsample_bytree': 1,

          'reg_alpha': 0.1,

          'reg_lambda': 0.1,

          'min_split_gain': 0.1,

          'min_child_weight': 0.1,

          'min_child_samples': 20,

          'metric' : 'mae'}



#          'subsample_for_bin': 200,

#          'subsample': 1,

#          'subsample_freq': 1,



gbm = lgb.LGBMRegressor(boosting_type= 'gbdt',

                        objective = 'regression_l1',

                        n_jobs = 4,

                        silent = True,

                        max_bin = params['max_bin'],

                        min_split_gain = params['min_split_gain'],

                        min_child_weight = params['min_child_weight'],

                        min_child_samples = params['min_child_samples'])



# To view the default model params:

gbm.get_params().keys()
print('Starting training...')



folds = KFold(n_splits=5, shuffle=True, random_state=11)



for fold_n, (train_index, valid_index) in enumerate(folds.split(X_train)):

    print('Fold', fold_n)

    X_train_f, X_valid = X_train.iloc[train_index], X_train.iloc[valid_index]

    y_train_f, y_valid = y_train_raw.iloc[train_index], y_train_raw.iloc[valid_index]



    # train

#    gbm = lgb.LGBMRegressor(num_leaves=50,

#                            min_data_in_leaf=50,

#                            learning_rate=0.01,

#                            n_estimators=500,

#                            verbosity=-1,

#                            boosting='gbdt',

#                            n_jobs = 4)



    gbm.fit(X_train_f, y_train_f.values.flatten(),

            eval_set=[(X_valid, y_valid.values.flatten())],

            eval_metric='mae',

            early_stopping_rounds=200)



    print('Starting predicting...')

    # predict

    y_pred = gbm.predict(X_valid, num_iteration=gbm.best_iteration_)

    # eval

    print('The mae of prediction is:', mean_absolute_error(y_valid.values.flatten(), y_pred))



# feature importances

print('Feature importances:', list(gbm.feature_importances_))
import matplotlib.pyplot as plt

import seaborn as sns



feature_imp = pd.DataFrame(sorted(zip(gbm.feature_importances_,X_train.columns)), columns=['Value','Feature'])



plt.figure(figsize=(15, 30))

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))

plt.title('LightGBM Features (avg over folds)')

plt.tight_layout()

plt.show()
plt.figure(figsize=(6, 6))

plt.scatter(y_valid.values.flatten(), y_pred)

plt.xlim(0, 20)

plt.ylim(0, 20)

plt.xlabel('actual', fontsize=12)

plt.ylabel('predicted', fontsize=12)

plt.plot([(0, 0), (20, 20)], [(0, 0), (20, 20)])

plt.show()
y_submit = gbm.predict(X_submit)
submission['time_to_failure'] = y_submit

print(submission.head())

submission.to_csv('submission.csv')