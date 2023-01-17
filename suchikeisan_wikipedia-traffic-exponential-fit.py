import sys
import time
import random
from scipy import optimize
import statsmodels.api as sm
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
%matplotlib inline
# %matplotlib notebook
# %matplotlib inline
is_debug = True

#I will run this notebook tonight on condition that this is True.
is_night_run = False

import scipy
scipy.__version__
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 18, 6

class Plotter(object):
    def __init__(self):
        self.xs = [i for i in range(2000)]
    
    def plot(
            self, array1, low_x_array1=0, array2=None, low_x_array2=0,
            yscale='linear'):
        plt.figure(figsize=(18, 6))
        plt.yscale(yscale)
        plt.plot(
            self.xs[low_x_array1:low_x_array1 + len(array1)],
            array1, label='array1')
        if array2 is not None:
            plt.plot(
                self.xs[low_x_array2:low_x_array2 + len(array2)], array2,
                label='array2')
#         plt.title(train_row[0])
        plt.xlabel('Days')
        plt.ylabel('Views')
        plt.legend()

plotter = Plotter()

def calc_correlation(y1, y2, lag):
    n_elements = min(len(y1), len(y2))
    norm_diff1 = (y1 - y1.mean()) / y1.std()
    norm_diff2 = (y2 - y2.mean()) / y2.std()
    return (norm_diff1[lag:n_elements] * norm_diff2[:n_elements-lag]).mean()

def calc_autocorrelation(y, lag):
    return calc_correlation(y, y, lag)

x = np.arange(0,10000,0.1)
y = np.sin(x)
lag = 3
print(sm.tsa.acf(y,lag))
print(calc_autocorrelation(y, lag))
data_dir = r'../input/'
train = pd.read_csv(
    data_dir + r'train_2.csv',
    skiprows=(lambda x: x % 100 != 0) if False else None
).fillna(0)
train.head()
page_id_etc = pd.read_csv(data_dir + r'key_2.csv').fillna(0)
print(page_id_etc.shape)

for i in range(65):
    print(page_id_etc.iloc[i])
train.info()
import math

def smape_fast(y_true, y_pred):
    out = 0.0
    for i in range(y_true.shape[0]):
        a = y_true[i]
        b = y_pred[i]
        c = a+b
        if -1 < c < 1:
            continue
        out += math.fabs(a - b) / c
    out *= (200.0 / y_true.shape[0])
    return out
def calc_median_of_day_of_week(data_all_days):
    """
    Following equations are assumed:
        data_all_days = rolling_week * week_dependency + residue
        week_dependency[i] = median_of_day_of_week[i%7]
    This function returns median_of_day_of_week.
    """
    pds_original = pd.Series(data_all_days)
    rolling_week = pds_original.rolling(
        7, min_periods=1, center=True).mean()
#     if is_debug:
#         plotter.plot(
#             pds_original[3:-3], 1000, rolling_week[3:-3], 1000, yscale='log')
#         plt.show()

    # +0.1 is for zero-division
    original_over_rolling_week = pds_original / (rolling_week + 0.1)
    number_of_days_of_week = len(data_all_days) // 7 + 1
    data_each_day = []  # np.zeros((7, number_of_days_of_week))
    for i in range(7):
        data_each_day.append(original_over_rolling_week.iloc[i::7])
#     print(data_each_day)
    median_of_day_of_week = np.zeros(7)
    for i in range(7):
        median_of_day_of_week[i] = data_each_day[i].median()
    return median_of_day_of_week



import scipy

def gaussian_like(x_data, y_scale, mu, width, exponent_x, bias):
    x_mod = (x_data - mu) / width
    return bias + y_scale * np.exp(-np.abs(x_mod)**exponent_x)

def calc_residuals(params, x_data, y_data):
    y_model = gaussian_like(x_data, *params)
    return (y_model - y_data)**2

def reduce_week_dependency(actual_train_data, mean_days1=11, mean_days2=3):
    """
    @param actual_train_data         np.array
    @param mean_days1
    @param mean_days2
    @return without_week_dependency  pd.Series
    @return median_of_day_of_week    np.array or None
                                     None means low weekly dependence.
    """

    median_of_day_of_week = calc_median_of_day_of_week(actual_train_data)
    if is_debug:
        print('median of weekday', median_of_day_of_week)
        original_data = actual_train_data.copy()
    without_week_dependency = actual_train_data
    is_week_dependent = (
        np.count_nonzero(median_of_day_of_week) == len(median_of_day_of_week))
    if is_week_dependent:
        for i in range(len(without_week_dependency)):
            without_week_dependency[i] /= median_of_day_of_week[i % 7]
    else:
        median_of_day_of_week = None
    
#     print(without_week_dependency)
    rolling_median = pd.Series(without_week_dependency).rolling(
        3, min_periods=1, center=True).median()
    
    rolling_mean = rolling_median.mean()
#     print('mean_days1', mean_days1, '   rolling_mean', rolling_mean)
#     if rolling_mean > 1e4:#1e4:
#         number_of_mean_days = 3
#     elif rolling_mean > 2e3: #2000:
#         number_of_mean_days = 7
#     else:
#         number_of_mean_days = 14
    number_of_mean_days = int(
       np.floor(1000 * mean_days1 / (rolling_mean + 0.1) + mean_days2))
#     if not is_night_run:
#         print('number_of_mean_days', number_of_mean_days)
    
    rolling_median = rolling_median.rolling(
        number_of_mean_days, min_periods=1, center=True).mean()
#     plotter.plot(
#         without_week_dependency, 1000, rolling_median.values, 1000,
#         yscale='log')
#     plt.show()
    without_week_dependency = rolling_median
    
    if is_debug:
        print('actual_train_data, without_week_dependency')
        plotter.plot(original_data, 1000, without_week_dependency, 1000,
                     yscale='log')
        for i in range(1000, 1000 + len(without_week_dependency), 7):
            plt.axvline(x=i, color='red',alpha=0.3)
        plt.show()
    
    return without_week_dependency, median_of_day_of_week

class IncreasingTrend(RuntimeError):
    pass

def train_and_pred_gaussian_like(
        without_week_dependency, median_of_day_of_week,
        low_idx_pred, up_idx_pred, page, min_bias=0.):
    """
    without_week_dependency[0] is data of 1st day,
    without_week_dependency[1] is data of 2nd day, ....
    This predict (low_idx_pred+1)th day, (low_idx_pred+2)th day,
    ..., (up_idx_pred)th day.

    @param without_week_dependency   pd.Series
    @param median_of_day_of_week     np.array
    @param low_idx_pred              int
    @param up_idx_pred               int
    @param page                      str
    @param min_bias                  int or float
    @return predicted value          np.array
    @return low_idx_fit              int
    """

    # +1 is for 0-division avoiding. e.g. norm = 1e+5 + 1
    norm = without_week_dependency.max() + 1
#     norm = np.median(without_week_dependency) + 1

    pred_data = np.zeros(up_idx_pred - low_idx_pred)

    normalized_train_data = without_week_dependency / norm
    normalized_min_bias = min_bias / norm

    # Training

    min_fit_points = 6
    low_idx_fit = 3 #len(actual_train_data) - 10
    sigmas = np.sqrt(without_week_dependency)
    diff_state = 0
    mean_1day = without_week_dependency.iloc[-1]
    for i in reversed(without_week_dependency.index[3:]):
#         print('without_week_dependency', i, without_week_dependency[i])
        if without_week_dependency[i - 1] \
                > mean_1day + sigmas[i]:
#             if diff_state == -1:
#                 low_idx_fit = min(i, len(actual_train_data) - min_fit_points)
#                 break
            diff_state = 1
            mean_1day = without_week_dependency[i - 1]
        elif without_week_dependency[i - 1] \
                < mean_1day - sigmas[i]:
            if diff_state == 1:
                low_idx_fit = min(i, len(actual_train_data) - min_fit_points)
                break
#             diff_state = -1
            if is_debug:
                print('diff < 0')
            raise IncreasingTrend()
    if is_debug:
        print('norm', norm, '  lower index for fit:', low_idx_fit)
    
    fitting_data = normalized_train_data.iloc[low_idx_fit:]
    init_mu = fitting_data.idxmax()
    init_y_scale = fitting_data[init_mu]
    init_param = (init_y_scale, init_mu, 1.0, 0.5, normalized_min_bias)
    
    x_data = fitting_data.index #np.arange(low_idx_fit, len(actual_train_data))
    
    # (min, max) of [y_scale, mu, width, exponent_x, bias]
    bounds=([0., x_data[0] - 1000., 1e-10, 0., normalized_min_bias],   
            [1e20, x_data[0] + 1000., 10000., 10.,1e20])
#     if is_debug:
#         print('x:', x_data, 'fitting_data:\n', fitting_data,
#               'min_bias', normalized_min_bias,
#               'p0', init_param)
#         plotter.plot(fitting_data, 1000+low_idx_fit)
#     popt, pcov = scipy.optimize.curve_fit(
#         gaussian_like, x_data, fitting_data,
#         bounds=bounds,
#         p0=init_param, method='trf')

    optimize_result = optimize.least_squares(
        calc_residuals, init_param, bounds=bounds, args=(x_data, fitting_data))
    if is_debug:
        print('opt param', optimize_result.x)
    popt = optimize_result.x
    normalized_pred = gaussian_like(x_data, *popt)
    
    if is_debug:
        plotter.plot(
            fitting_data, 1000+low_idx_fit, 
            normalized_pred, 1000+low_idx_fit)
        plt.title('data and pred')
        plt.show()

#   Prediction
    if median_of_day_of_week is not None:
        for i in range(low_idx_pred, up_idx_pred):
            pred_data[i - low_idx_pred] = norm * gaussian_like(
                i, *popt) * median_of_day_of_week[i % 7]
    else:
        for i in range(low_idx_pred, up_idx_pred):
            pred_data[i - low_idx_pred] = norm * gaussian_like(i, *popt)
    
#     plotter.plot(
#         actual_train_data[low_idx_fit:], 1000+low_idx_fit, 
#         pred_data[low_idx_fit:len(actual_train_data)], 1000+low_idx_fit)
    
    return pred_data, low_idx_fit


# test
x_data = np.array([2, 3, 4, 5, 6])
y_model = gaussian_like(x_data, 10, 3, 1, 0.5, 100)
# print(y_model)

actual_train_data = pd.Series(
    [200-x**2 for x in range(-5, 10)],
    index=pd.date_range('2011/01/01', periods=15, freq='D'))
print('Test train data\n', actual_train_data)
without_week_dependency, median_of_day_of_week = reduce_week_dependency(
    actual_train_data.values)
print('without_week_dependency', without_week_dependency)
pred, idx = train_and_pred_gaussian_like(
    without_week_dependency, median_of_day_of_week, 2, 17, 'Test', 0.1)
plotter.plot(actual_train_data, 1000, pred, 1000)

date_char_counts = len("YYYY-MM-DD")
page_id_etc['Date'] = [
    page[-date_char_counts:] for page in tqdm(page_id_etc.Page)]
page_id_etc['Page'] = [
    page[:-date_char_counts-1] for page in tqdm(page_id_etc.Page)]
submit_work = page_id_etc.copy()
submit_work['Visits'] = np.NaN
print(submit_work.head())
submit_work = submit_work.pivot(
    index='Page', columns='Date', values='Visits'
).astype('float32').reset_index()
print(submit_work.head())
def parse_iso_format(iso_format):
    """
    eg. If iso_format=='2017-09-10' then this return (2017, 9, 10) 
    """
    return tuple(int(x) for x in iso_format.split('-'))
# date1 = datetime.date(*parse_iso_format('2017-09-10'))
# date2 = datetime.date(*parse_iso_format('2017-09-13'))
# (date2 - date1).days

train_file_first_day = datetime.date(*parse_iso_format(train.columns[1]))
submission_last_day = datetime.date(*parse_iso_format(submit_work.columns[-1]))
up_idx_submission = (submission_last_day - train_file_first_day).days + 1

print(up_idx_submission)

submit_work = submit_work.iloc[:, 1:]
columns_name = submit_work.columns.name
submit_work = train[['Page']].join(submit_work)
submit_work.columns.name = columns_name
print(submit_work.head(2))
submit_work.tail()
if train.columns[-1] == 'lang':
    date_end = len(train.columns) - 1
else:
    date_end = len(train.columns)
    
cols = train.columns[1:date_end]
low_idx_submission = up_idx_submission - (submit_work.shape[1] - 1)

# training days
number_of_actual_train = 90
smape_days = 21 #up_idx_train - (low_idx_train + low_idx_fit)  #21
if is_night_run:  
#     low_idx_train = (date_end - 1) - number_of_actual_train
    original_up_idx_train = date_end - 1
#     low_idx_pred = original_up_idx_train - smape_days
    up_idx_pred = up_idx_submission
else:
#     low_idx_train = 740 - number_of_actual_train
    original_up_idx_train = 740
#     low_idx_pred = low_idx_train + 10
    up_idx_pred = up_idx_submission

train_row_count = train.shape[0]

def train_and_predict(
        hyper_params, train_row_start, row_step, is_variable_up_idx):
    smape_sum = 0.0
    smape_count = 0
    if is_variable_up_idx or (not is_night_run):
        low_idx_train = 740 - number_of_actual_train
        low_idx_pred = low_idx_train + 10
    else:
        low_idx_train = (date_end - 1) - number_of_actual_train
        low_idx_pred = original_up_idx_train - smape_days
#     print('hyper_params', hyper_params)
    for row_num in tqdm(range(train_row_start, train_row_count, row_step)):
#     for row_num in [7001, 70002, 70003, 100001]:
        if is_variable_up_idx or (not is_night_run):
            up_idx_train = 740 - 50 + ((row_num // 1000) % 80)
    #         up_idx_train = date_end - 1
#             print('up_idx_train', up_idx_train)
        else:
            up_idx_train = original_up_idx_train

        train_row = train.iloc[row_num, :]
    #     print(train_row[0], train_row[-3:])
    #     data_row = np.array(train_row[1:date_end],'f')
    #     actual_train_data = data_row[low_idx_train:up_idx_train]

        data_row = train_row[1:date_end].astype('float64')
        data_row.index = pd.to_datetime(data_row.index)
        try:
            first_nonzero = data_row.nonzero()[0][0]
    #         idx_min_data = data_row[
    #             max(first_nonzero, up_idx_train - 400):up_idx_train].values.argmin()
            col_for_min_bias = data_row[
                max(first_nonzero, up_idx_train - 400):up_idx_train].idxmin()
            idx_min_data = data_row[:col_for_min_bias].shape[0] - 1
            if is_debug:
                print('idx_min_data', idx_min_data,
    #                   '  data_row[col_for_min_bias]', data_row[col_for_min_bias],
    #                   '  data_row[:col_for_min_bias].shape', data_row[:col_for_min_bias].shape,
    #                   '  data_row[idx_min_data-2:idx_min_data+2]', data_row.iloc[idx_min_data-2:idx_min_data+2],
    #                   '  col_for_min_bias', col_for_min_bias,
                      '  min data', data_row.iloc[idx_min_data])
        except:
            if is_debug:
                print('idx_min is not found.')
            idx_min_data = None

    #     print(
    #         '1st nonzero index:', first_nonzero, '   min_data_row:', min_data_row)
#         print('low_idx_train', low_idx_train)
        actual_train_data = data_row.iloc[low_idx_train:up_idx_train]

        train_last_part = actual_train_data.iloc[-30:]
    #     print('train_last_part', train_last_part)
        train_last_part_median = train_last_part.median()
        if train_last_part_median < 20.0: #30.0: #10.0:
            last_mean = train_last_part.mean()
            if is_debug:
                print('last_part_median:', train_last_part_median,
                      '  last_part_mean:', last_mean)
            visit = last_mean if (
                last_mean < train_last_part_median + 1.0
                ) else train_last_part_median
            pred = np.full((up_idx_pred - low_idx_pred), visit)
            pred_no_trend = pred_with_trend = pred
        else:
    #         if is_debug: continue
            (without_week_dependency, median_of_day_of_week
                ) = reduce_week_dependency(
                actual_train_data.values, *hyper_params)
            pred_no_trend = np.full(
                (up_idx_pred - low_idx_pred),
                without_week_dependency.iloc[-30:].median())
            if median_of_day_of_week is not None:
                for i in range(low_idx_pred - low_idx_train, up_idx_pred - low_idx_train):
                    pred_no_trend[i - low_idx_pred + low_idx_train] \
                        *= median_of_day_of_week[i % 7]
            if idx_min_data is not None:
                min_data_row = data_row[idx_min_data]
    #             if median_of_day_of_week is not None:
    #                  min_data_row /= median_of_day_of_week[
    #                     (idx_min_data - low_idx_train) % 7]
            else:
                min_data_row = 0.0

            rolling_week = data_row[:up_idx_train].rolling(
                7, min_periods=1, center=True).mean()
            rolling_drop0229 = rolling_week.drop(labels=pd.to_datetime(['2016-02-29']))
            ac = calc_autocorrelation(rolling_drop0229.values, 365)
            if is_debug:
                print('autocorr:', ac)
#             print('rolling_drop0229', rolling_drop0229)
                plotter.plot(rolling_drop0229)
                plt.show()
#             train_diff = rolling_drop0229.diff().dropna()
#             plotter.plot(train_diff[300:])
#             plt.show()
            
    
            try:
                if ac > 0.6:
#                     diff_1year_ago = rolling_drop0229[low_idx_pred - 100:low_idx_pred - 1
#                                     ].values - rolling_drop0229[low_idx_pred - 100 - 365:low_idx_pred - 1 - 365].values
#                     diff_1year_ago = np.median(diff_1year_ago)
#                     if is_debug: print(diff_1year_ago)
                    pred_with_trend = rolling_drop0229[
                        low_idx_pred-365:up_idx_pred-365].values

#                     pred_with_trend[0] = rolling_week[low_idx_pred - 1] + rolling_drop0229[low_idx_pred - 365]
#                     for i in range(1, up_idx_pred - low_idx_pred):
#                         pred_with_trend[i] = pred_with_trend[i - 1] \
#                            + rolling_drop0229[i + low_idx_pred - 1 - 365]
                    if median_of_day_of_week is not None:
                        for i in range(low_idx_pred - low_idx_train, up_idx_pred - low_idx_train):
                            pred_with_trend[i - low_idx_pred + low_idx_train] \
                                *= median_of_day_of_week[i % 7]
                    pred = pred_with_trend
                else:
                    pred_with_trend, low_idx_fit = train_and_pred_gaussian_like(
                        without_week_dependency,
                        median_of_day_of_week,
                        low_idx_pred - low_idx_train,
                        up_idx_pred - low_idx_train, 
                        train_row[0], min_data_row)
                
            
                    smape_no_trend = smape_fast(
                        data_row[up_idx_train - smape_days:up_idx_train],
                        pred_no_trend[
                            up_idx_train - smape_days - low_idx_pred
                            :up_idx_train - low_idx_pred])
                    smape_with_trend = smape_fast(
                        data_row[up_idx_train - smape_days:up_idx_train],
                        pred_with_trend[
                            up_idx_train - smape_days - low_idx_pred
                            :up_idx_train - low_idx_pred])
                    if is_debug:
                        print(
                            'SMAPE no_trend, with_trend:',
                            smape_no_trend, smape_with_trend)
                    pred = pred_no_trend if smape_no_trend < smape_with_trend \
                        else pred_with_trend
            except:
                if is_debug:
                    print("Curve fit failed at row#", row_num, ":", sys.exc_info()[0])
    #             low_idx_fit = len(actual_train_data) - 1
    #             pred_with_trend = train_and_pred_nn2_week(
    #                 actual_train_data,
    #                 low_idx_pred - low_idx_train,
    #                 up_idx_pred - low_idx_train, 
    #                 train_row[0])
                pred = pred_with_trend = pred_no_trend        
        
#         print('pred', pred[low_idx_submission - low_idx_pred:])
        submit_work.iloc[row_num, 1:] = pred[
            low_idx_submission - low_idx_pred:]

        up_idx_smape = min(len(data_row), up_idx_pred)
        if up_idx_train < up_idx_smape:
            smape_row = smape_fast(
                data_row[up_idx_train:up_idx_smape],
                pred[up_idx_train - low_idx_pred:up_idx_smape - low_idx_pred])
            smape_sum += smape_row
            smape_count += 1
        else:
            smape_row = None

        if is_debug: #and (not is_night_run):
    #         plotter.plot(
    #             data_row[low_idx_train:], low_idx_train, pred, low_idx_pred,
    #             'linear')
    #         plt.axvline(x=up_idx_train, color='red',alpha=0.3)
    #         plt.show()
            low_idx_plot = 0
            plotter.plot(
                data_row[low_idx_plot:], low_idx_plot, 
                pred_no_trend[:data_row.shape[0] - low_idx_pred], low_idx_pred, 'log')
            plt.axvline(x=up_idx_train, color='red',alpha=0.3)
            plt.title('pred_no_trend')
            plt.show()
    #         plotter.plot(
    #             data_row[low_idx_plot:], low_idx_plot, 
    #             pred[:data_row.shape[0] - low_idx_pred], low_idx_pred, 'log')
            plotter.plot(
                data_row[low_idx_plot:], low_idx_plot, 
                pred, low_idx_pred, 'log')
            plt.axvline(x=up_idx_train, color='red',alpha=0.3)
            plt.title('pred')
            plt.show()

            print("SMAPE:", smape_row, "Row:", row_num, train_row[0])
            print("\n\n")

    #     if is_debug: break
    if smape_count > 0:
        mean_smape = smape_sum/float(smape_count)
        print("mean SMAPE:", mean_smape)
        return mean_smape
    else:
        return np.nan
#Optimize hyper parameters
t_start = time.time()

optimized_params = {}
row_step_for_optimize = 10000
maxiter = 1 if is_debug else 100
for i in range(0, row_step_for_optimize, row_step_for_optimize // 5):
    res = optimize.minimize(
        train_and_predict, np.array([11, 3]), method='Nelder-Mead',
        args=(i, row_step_for_optimize, True), options={'maxiter':maxiter, 'xatol':0.1})
    print('Elapsed time(s):',time.time() - t_start,
          '  Optimized params:', res.x)
    optimized_params[i] = res.x
    if is_debug: break

optimized_params = pd.DataFrame(optimized_params)
print('Optimized params')
print(optimized_params)

using_params = optimized_params.T.mean().values

print('Mean of params:', using_params)
# %%capture output

if is_debug: using_params = [11.17037109, 2.93941406]
print('Mean of params:', using_params)

#Main part

train_row_start = 0
if is_debug: train_row_start = 1000
row_step = 1 if (is_night_run and (not is_debug)) else 10000
print('row step:', row_step)

t_start = time.time()
train_and_predict(using_params, train_row_start, row_step, False)  
print('Elapsed time(s):', time.time() - t_start)
# print(type(output))
# print(submit_work.iloc[train_row_start, :])
submit_save = pd.melt(
    submit_work, id_vars=['Page'], var_name='Date', value_name='Visits')
submit_save = submit_save.merge(page_id_etc, on=['Page','Date'])
print(submit_save.head())
submit_visits = submit_save['Visits']
large_visits = submit_save.loc[submit_visits > 10000, 'Visits']
# print(large_visits.iloc[0:5])
print('Max visits:', large_visits.max())
submit_save.loc[np.isinf(submit_visits), 'Visits'] = 1e9
submit_save.loc[submit_visits < 0.0, 'Visits']  = 0.0
submit_save.head()
if not is_debug:
    submit_save[['Id', 'Visits']].to_csv(
        'submission.csv', index=False)
large_visits = submit_save.loc[submit_visits > 10000, 'Visits']
# print(large_visits.iloc[0:5])
print('Max visits:', large_visits.max())
print('Number of Ids of large visits:', len(large_visits))
print('Numver of Ids of which value < 0:',
      len(submit_save.loc[submit_visits < 0.0, 'Visits']))
print('Numver of Ids of which value is finite:',
      len(submit_save.loc[np.isfinite(submit_visits), 'Visits']))
print('Numver of Ids of which value is NaN:',
      len(submit_save.loc[np.isnan(submit_save['Visits']), 'Visits']))

# print(submit_save.loc[np.isnan(submit_save['Visits'])])

# inf_rows = submit_work.loc[submit_work[np.isinf(submit_work)]]
inf_rows = submit_work.loc[np.isinf(submit_work['2017-11-13'])]
if not inf_rows.empty:
    print('Infinity rows:', inf_rows[0:5])
    
nan_rows = submit_work.loc[np.isnan(submit_work['2017-11-13'])]
if not nan_rows.empty:
    print('NaN rows:', nan_rows[0:5])

large_rows = []
for date in submit_work.columns[1:]:
    large_rows.append(submit_work.loc[submit_work[date] > 1e12])
# if not large_rows:
for rows in large_rows:
    if not rows.empty:
        print('Large rows:', rows.head())

# submit_save_mod = submit_save.copy()
# submit_save_mod.loc[np.isnan(submit_visits), 'Visits']  = 0.0
# if not is_debug:
#     submit_save_mod[['Id', 'Visits']].to_csv(
#         data_dir + 'submission.csv', index=False)