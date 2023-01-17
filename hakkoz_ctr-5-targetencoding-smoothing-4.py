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
fourth_df=merged_df.loc[30000001:, :]
fourth_df.to_csv("fourth_df.csv")