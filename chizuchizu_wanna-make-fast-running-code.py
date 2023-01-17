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

del memo

df
%%time

master_id = []

memo = data.groupby("id")["demand"]

# id_list = sorted(data["id"].unique().tolist())

flag = False

DAYS_PRED = 28

for id in tqdm(memo):

    x = id[1]

    id = pd.DataFrame(id[1])

    for diff in [0, 1, 2]:

        shift = DAYS_PRED + diff

        id[f"shift_t{shift}"] = x.shift(shift)

    for size in [7, 30, 60, 90, 180]:

        id[f"rolling_std_t{size}"] = id["shift_t28"].rolling(size).std()

        id[f"rolling_mean_t{size}"] = id["shift_t28"].rolling(size).mean()

    id["rolling_skew_t30"] = id["shift_t28"].rolling(30).skew()

    id["rolling_kurt_t30"] = id["shift_t28"].rolling(30).kurt()

    master_id.append(id)

pd.concat(master_id)