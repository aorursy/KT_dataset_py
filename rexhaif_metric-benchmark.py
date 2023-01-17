import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from sklearn import metrics

from tqdm import tqdm



import numba

from numba import jit, float32



import os

print(os.listdir("../input"))
import warnings

warnings.filterwarnings("ignore", category=numba.NumbaWarning)
def comp_score (y_true, y_pred, jtype):

    df = pd.DataFrame()

    df['y_true'] , df['y_pred'], df['jtype'] = y_true , y_pred, jtype

    score = 0 

    for t in np.unique(jtype):

        score_jtype = np.log(metrics.mean_absolute_error(df[df.jtype==t]['y_true'],df[df.jtype==t]['y_pred']))

        score += score_jtype

        #print(f'{t} : {score_jtype}')

    score /= len(np.unique(jtype))

    return score
def metric(df, preds, verbose=False):

    

    if verbose:

        iterator = lambda x: tqdm(x)

    else:

        iterator = list

        

    df["prediction"] = list(preds)

    maes = []

    for t in iterator(df.type.unique()):

        y_true = df[df.type==t].scalar_coupling_constant.values

        y_pred = df[df.type==t].prediction.values

        mae = np.log(metrics.mean_absolute_error(y_true, y_pred))

        maes.append(mae)

    return np.mean(maes)
def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):

    """

    Fast metric computation for this competition: https://www.kaggle.com/c/champs-scalar-coupling

    Code is from this kernel: https://www.kaggle.com/uberkinder/efficient-metric

    """

    maes = (y_true-y_pred).abs().groupby(types).mean()

    return np.log(maes.map(lambda x: max(x, floor))).mean()
# Basic version

def mean_log_mae(y_true, y_pred, types, verbose=False):

    if verbose:

        iterator = lambda x: tqdm(x)

    else:

        iterator = list

    

    per_type_data = {

        t : {

            'true': [],

            'pred': []

        } 

        for t in list(set(types))

    }

    for true, pred, t in iterator(zip(y_true, y_pred, types)):

        per_type_data[t]['true'].append(true)

        per_type_data[t]['pred'].append(pred)

        

    maes = []

    for t in iterator(set(types)):

        maes.append(np.log(metrics.mean_absolute_error(per_type_data[t]['true'], per_type_data[t]['pred'])))

        

    return np.mean(maes)
# Compiling efficent log(mae) implementation

@jit(float32(float32[:], float32[:]))

def jit_log_mae(y_true: np.ndarray, y_pred: np.ndarray):

    n = y_true.shape[0]

    return np.log(np.sum(np.absolute(y_true - y_pred))/n)





def speedup_mean_log_mae(y_true: np.ndarray, y_pred: np.ndarray, types: np.ndarray, verbose=False) -> np.float64:

    if verbose:

        iterator = lambda x: tqdm(x)

    else:

        iterator = list

    

    per_type_data = {

        t : {

            'true': [],

            'pred': []

        } 

        for t in list(set(types))

    }

    for true, pred, t in iterator(zip(y_true, y_pred, types)):

        per_type_data[t]['true'].append(true)

        per_type_data[t]['pred'].append(pred)

        

    maes = []

    for t in iterator(set(types)):

        maes.append(

            jit_log_mae( ## that's the speedup

                np.array(per_type_data[t]['true'], dtype=np.float32),

                np.array(per_type_data[t]['pred'], dtype=np.float32)

            )

        )

        

    return np.mean(maes)
# Trying to jit-compile all

@jit

def jit_mean_log_mae(y_true: np.ndarray, y_pred: np.ndarray, types: np.ndarray) -> np.float64:

    

    uniq_types: np.ndarray = np.unique(types)

    

    per_type_data = dict()

    for t in uniq_types:

        per_type_data[t] = {

            'true': [],

            'pred': []

        }

    

    for i in np.arange(len(types)):

        per_type_data[types[i]]['true'].append(y_true[i])

        per_type_data[types[i]]['pred'].append(y_pred[i])

        

    maes = []

    for t in uniq_types:

        maes.append(jit_log_mae(np.array(per_type_data[t]['true'], dtype=np.float32), np.array(per_type_data[t]['pred'], dtype=np.float32)))

        

    return np.mean(maes)

        
from timeit import Timer
general_train = pd.read_csv("../input/train.csv")
## Pre-Compiling jit functions

jit_mean_log_mae(general_train.scalar_coupling_constant.values, np.zeros(len(general_train)), general_train.type.values)

speedup_mean_log_mae(general_train.scalar_coupling_constant.values, np.zeros(len(general_train)), general_train.type.values)
benchmarking_code = {

    'lgb_plus': "comp_score(train.scalar_coupling_constant.values, zeros, train.type.values)",

    'competition_metric': "metric(train, zeros, verbose=False)",

    'efficent_metric': "group_mean_log_mae(train.scalar_coupling_constant, zeros, train.type)",

    "basic": "mean_log_mae(train.scalar_coupling_constant.values, zeros, train.type.values, verbose=False)",

    "partial_jit": "speedup_mean_log_mae(train.scalar_coupling_constant.values, zeros, train.type.values, verbose=False)",

    "full_jit": "jit_mean_log_mae(train.scalar_coupling_constant.values, zeros, train.type.values)"

}



quality_code = {

    'lgb_plus': lambda train, zero_arr: comp_score(train.scalar_coupling_constant.values, zero_arr, train.type.values),

    'competition_metric': lambda train, zero_arr: metric(train, zero_arr, verbose=False),

    'efficent_metric': lambda train, zero_arr: group_mean_log_mae(train.scalar_coupling_constant, zero_arr, train.type),

    "basic": lambda train, zero_arr: mean_log_mae(train.scalar_coupling_constant.values, zero_arr, train.type.values, verbose=False),

    "partial_jit": lambda train, zero_arr: speedup_mean_log_mae(train.scalar_coupling_constant.values, zero_arr, train.type.values, verbose=False),

    "full_jit": lambda train, zero_arr: jit_mean_log_mae(train.scalar_coupling_constant.values, zero_arr, train.type.values)

}
## ensure that results are the same

def measure_quality(implementations, n_samples=1000, n_different_seeds=5):

    np.random.seed(0) # for reproducible random seed generation

    results = {key:[] for key in implementations.keys()}

    for seed in np.random.randint(0, 100, size=n_different_seeds):

        train = general_train.sample(n=n_samples, random_state=seed).reset_index(drop=True)

        zeros = np.zeros(n_samples)

        

        for impl_name, impl in implementations.items():

            value = impl(train, zeros)

            results[impl_name].append(value)

            

    result = pd.DataFrame(results)

    result['std'] = result.std(axis=1)

    

    mean_std = result['std'].mean()

    return result, mean_std
def measure_performance(implementations, n_samples=1000, n_different_seeds=5, n_iterations=1000, n_repeats=10):

    np.random.seed(0) # for reproducible random seed generation

    results = {key:[] for key in implementations.keys()}

    for seed in np.random.randint(0, 100, size=n_different_seeds):

        train = general_train.sample(n=n_samples, random_state=seed).reset_index(drop=True)

        zeros = np.zeros(n_samples)

        scope = dict(globals(), **locals())

        for impl_name, impl_code in implementations.items():

            eval_speed = Timer(impl_code, globals=scope).repeat(repeat=n_repeats, number=n_iterations)

            results[impl_name] += eval_speed

            

    return pd.DataFrame(results)
%%time

qdf, mstd = measure_quality(quality_code, 100000, 20)

print(f"Mean standard deviation of metric values across different implementations: {mstd:.10f}")
%%time

df = measure_performance(benchmarking_code, 1000, 10, 1000, 10)
df.describe()