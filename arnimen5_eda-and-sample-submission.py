import pandas as pd

import numpy as np
PATH = '../input/conways-reverse-game-of-life-2020/'
train_df = pd.read_csv(PATH + 'train.csv')

sub = pd.read_csv(PATH + 'test.csv')
train_df.head()
INIT_COLS = []

STOP_COLS = []

REN_DICT = {}

for i in range(625):

    INIT_COLS.append(f"start_{i}")

    STOP_COLS.append(f"stop_{i}")

    REN_DICT[f"stop_{i}"] = f"start_{i}"

#=====

sub = sub.rename(columns=REN_DICT) # swapping stopping columns as starting ones

sub.drop("delta", axis=1, inplace=True)

sub.head()

sub.to_csv("submission.csv", index=False)