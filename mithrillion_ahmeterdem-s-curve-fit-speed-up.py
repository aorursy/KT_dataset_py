#### import argparse
import multiprocessing
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import time
import warnings
import numba
import dask.dataframe as dd
import pdb
from dask.distributed import LocalCluster, Client

NUM_PARTITIONS = 100
LOW_PASSBAND_LIMIT = 3
FEATURES = ["A", "B", "t0", "tfall", "trise", "cc", "fit_error", "status", "t0_shift"]
# bazin, errorfunc and fit_scipy are developed using:
# https://github.com/COINtoolbox/ActSNClass/blob/master/examples/1_fit_LC/fit_lc_parametric.py
def bazin(time, low_passband, A, B, t0, tfall, trise, cc):
    X = np.exp(-(time - t0) / tfall) / (1 + np.exp((time - t0) / trise))
    return (A * X + B) * (1 - cc * low_passband)


def errfunc(params, time, low_passband, flux, weights):
    return abs(flux - bazin(time, low_passband, *params)) * weights


def fit_scipy(time, low_passband, flux, flux_err):
    time -= time[0]
    sn = np.power(flux / flux_err, 2)
    start_point = (sn * flux).argmax()

    t0_init = time[start_point] - time[0]
    amp_init = flux[start_point]
    weights = 1 / (1 + flux_err)
    weights = weights / weights.sum()
    guess = [0, amp_init, t0_init, 40, -5, 0.5]

    result = least_squares(errfunc, guess, args=(time, low_passband, flux, weights), method='lm')
    result.t_shift = t0_init - result.x[2]

    return result


def yield_data(meta_df, lc_df):
    cols = ["object_id", "mjd", "flux", "flux_err", "low_passband"]
    for i in range(NUM_PARTITIONS):
        yield meta_df[(meta_df["object_id"] % NUM_PARTITIONS) == i]["object_id"].values, \
              lc_df[(lc_df["object_id"] % NUM_PARTITIONS) == i][cols]


def get_params(object_id_list, lc_df, result_queue):
    results = {}
    for object_id in object_id_list:
        light_df = lc_df[lc_df["object_id"] == object_id]
        try:
            result = fit_scipy(light_df["mjd"].values, light_df["low_passband"].values,
                               light_df["flux"].values, light_df["flux_err"].values)
            results[object_id] = np.append(result.x, [result.cost, result.status, result.t_shift])
        except Exception as e:
            print(e)
            results[object_id] = None
    result_queue.put(results)


def parallelize(meta_df, df):
    pool_size = multiprocessing.cpu_count() * 2
    pool = multiprocessing.Pool(processes=pool_size)

    manager = multiprocessing.Manager()
    result_queue = manager.Queue()

    for m, d in yield_data(meta_df, df):
        pool.apply_async(get_params, (m, d, result_queue))

    pool.close()
    pool.join()

    return [result_queue.get() for _ in range(NUM_PARTITIONS)]
meta_df = pd.read_csv('../input/training_set_metadata.csv')
lc_df = pd.read_csv('../input/training_set.csv')
lc_df["low_passband"] = (lc_df["passband"] < LOW_PASSBAND_LIMIT).astype(int)
warnings.simplefilter('ignore', RuntimeWarning)
start = time.time()
result_list = parallelize(meta_df, lc_df)
calc_fin = time.time()
print(f'calculation time: {calc_fin - start}')
# optimise 1 - using numba
@numba.jit(nopython=True)  # added
def bazin(time, low_passband, A, B, t0, tfall, trise, cc):
    X = np.exp(-(time - t0) / tfall) / (1. + np.exp((time - t0) / trise))
    return (A * X + B) * (1. - cc * low_passband)

@numba.jit(nopython=True)  # added
def errfunc(params, time, low_passband, flux, weights):
    A, B, t0, tfall, trise, cc = params
    return np.abs(flux - bazin(time, low_passband, A, B, t0, tfall, trise, cc)) * weights
warnings.simplefilter('ignore', RuntimeWarning)
start = time.time()
result_list = parallelize(meta_df, lc_df)
calc_fin = time.time()
print(f'calculation time: {calc_fin - start}')
start = time.time()
final_result = {}
for res in result_list:
    final_result.update(res)

for index, col in enumerate(FEATURES):
    meta_df[col] = meta_df["object_id"].apply(lambda x: final_result[x][index])
end = time.time()
print(f'feature collection time: {end - start}')
# optimise 2 - using numba, dask
def collect_results(chunk):
    result = fit_scipy(time=chunk['mjd'].values,
                       low_passband=chunk['low_passband'].values, 
                       flux=chunk['flux'].values, 
                       flux_err=chunk['flux_err'].values)
    return pd.Series(np.append(result.x, [result.cost, result.status, result.t_shift]), index=FEATURES)
cluster = LocalCluster(n_workers=4, processes=True, scheduler_port=0,
                           diagnostics_port=8787)
client = Client(cluster)
lc_df_dask = dd.from_pandas(lc_df, npartitions=1).set_index('object_id').repartition(npartitions=100)
warnings.simplefilter('ignore', RuntimeWarning)
start = time.time()
# result_list = lc_df.head(2000).groupby('object_id').apply(collect_results)
result_list = lc_df_dask.groupby('object_id').apply(
    collect_results,
    meta=dict(A='f8', B='f8', t0='f8', tfall='f8', trise='f8', 
             cc='f8', fit_error='f8', status='f8', t0_shift='f8')).compute()
end = time.time()
print(end - start)
result_list.head()