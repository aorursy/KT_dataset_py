import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
dataset = pd.read_csv("../input/suicide-rates-overview-1985-to-2016/master.csv")
dataset.iloc[0:11, :]
rows_with_missing_val = dataset[dataset.isnull().any(axis=1)]

rows_with_missing_val.iloc[0, :]
dataset.info()