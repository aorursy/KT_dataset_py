import random

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns



%matplotlib inline

plt.rcParams['figure.figsize']=(10, 6)
def read_csv_random_sample(path, nrows):

    total_rows_in_file_minus_header = sum(1 for line in open(path)) - 1

    skip_mask = random.sample(

        population=range(1, total_rows_in_file_minus_header + 1),

        k=total_rows_in_file_minus_header - nrows

    )

    return pd.read_csv(path, skiprows=skip_mask)
train = read_csv_random_sample(path="../input/train_ver2.csv", nrows=5000000)
PRODUCT_COLUMNS = [column for column in train.columns if column.endswith('ult1')]
train = train.assign(sum_of_products_owned = lambda df: df[PRODUCT_COLUMNS].sum(axis=1))
train.groupby('fecha_dato')['sum_of_products_owned'].mean()
for product in PRODUCT_COLUMNS:

    train.groupby('fecha_dato')[product].mean().plot()
for product in PRODUCT_COLUMNS:

    if product != 'ind_cco_fin_ult1':

        train.groupby('fecha_dato')[product].mean().plot()
train.groupby('fecha_dato')['sum_of_products_owned'].sum().plot(kind='bar')