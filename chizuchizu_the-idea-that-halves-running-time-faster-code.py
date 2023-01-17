from typing import Any, Dict

import numpy as np

import itertools as it

import gc

from tqdm import tqdm

from numba import jit

import pandas as pd
def read_data(n_rows: int) -> pd.DataFrame:

    data = pd.read_pickle("../input/walmartbasedata/data2.pickle").loc[:n_rows]

    data = data[data["part"] != "evaluation"]

    return data

data = read_data(1_000)

DAYS_PRED = 28
data
%%time

df = data.copy()

memo = df.groupby(["id"])["demand"]

for diff in [0, 1, 2]:

    shift = DAYS_PRED + diff

    df[f"shift_t{shift}"] = memo.transform(

        lambda x: x.shift(shift)

    )



for size in [7, 30, 60, 90, 180]:

    df[f"rolling_std_t{size}"] = memo.transform(

        lambda x: x.shift(DAYS_PRED).rolling(size).std()

    )



for size in [7, 30, 60, 90, 180]:

    df[f"rolling_mean_t{size}"] = memo.transform(

        lambda x: x.shift(DAYS_PRED).rolling(size).mean()

    )



df["rolling_skew_t30"] = memo.transform(

    lambda x: x.shift(DAYS_PRED).rolling(30).skew()

)

df["rolling_kurt_t30"] = memo.transform(

    lambda x: x.shift(DAYS_PRED).rolling(30).kurt()

)

df
from scipy.stats import skew as skew2

from scipy.stats import kurtosis as kurt2

def shift2(arr, num):

    result = np.empty(arr.shape[0])

    if num > 0:

        result[:num] = np.nan

        result[num:] = arr[:-num]

    elif num < 0:

        result[num:] = fill_value

        result[:num] = arr[-num:]

    else:

        result[:] = arr

    return result



def rolling2(arr, num):

    v = np.ones(num) / num

    y = np.convolve(arr, v, mode="same")

    return y
%%time

df = data.copy()

memo = df.groupby(["id"])["demand"]

for diff in [0, 1, 2]:

    shift = DAYS_PRED + diff

    df[f"shift_t{shift}"] = memo.transform(

        lambda x: shift2(x.values, int(shift))

    )

for size in [7, 30, 60, 90, 180]:

    df[f"rolling_std_t{size}"] = memo.transform(

        lambda x: np.std(rolling2(shift2(x.values, DAYS_PRED), size))

    )

for size in [7, 30, 60, 90, 180]:

    df[f"rolling_mean_t{size}"] = memo.transform(

        lambda x: np.mean(rolling2(shift2(x.values, DAYS_PRED), size))

    )

df["rolling_kurt_t30"] = memo.transform(

    lambda x: kurt2(rolling2(shift2(x.values, DAYS_PRED), size))

)

df["rolling_skew_t30"] = memo.transform(

    lambda x: skew2(rolling2(shift2(x.values, DAYS_PRED), size))

)

df