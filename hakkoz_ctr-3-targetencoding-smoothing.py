# import packages

import csv

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

import seaborn as sns

import pickle

import gzip

with open('../input/ctr-read-data/merged_df.pkl', 'rb') as inputfile:

    merged_df = pickle.load(inputfile)
def mem_usage(pandas_obj):

    if isinstance(pandas_obj,pd.DataFrame):

        usage_b = pandas_obj.memory_usage(deep=True).sum()

    else: # we assume if not a df it's a series

        usage_b = pandas_obj.memory_usage(deep=True)

    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes

    return "{:03.2f} MB".format(usage_mb)
def calc_smooth_mean(df, col, label, mean_label, weight):

    # Compute the number of values and the mean of each group

    agg = df.groupby(col)[label].agg(['count', 'mean'])

    counts = agg['count']

    means = agg['mean']



    # Compute the "smoothed" means

    smooth = (counts * means + weight * mean_label) / (counts + weight)



    # Replace each value by the according smoothed mean

    return df[col].map(smooth)
# Compute the global mean

mean_label = merged_df["label"].mean()



for col in merged_df.columns.tolist():

    encoded = calc_smooth_mean(merged_df, col, "label", mean_label, 300)

    downcasted = pd.to_numeric(encoded , downcast='float')

    merged_df[col] = downcasted
mem_usage(merged_df)
merged_df.info()
with gzip.open('merged_df_target_smooth.pkl', 'wb') as output:

    pickle.dump(merged_df, output,protocol=-1)