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
train_df = merged_df.loc[:40000000, :]

test_df = merged_df.loc[40000001:, :]
def calc_smooth_mean(df, col, label, mean_label, weight):

    # Compute the number of values and the mean of each group

    agg = df.groupby(col)[label].agg(['count', 'mean'])

    counts = agg['count']

    means = agg['mean']



    # Compute the "smoothed" means

    smooth = (counts * means + weight * mean_label) / (counts + weight)



    mapping_dict = {key:value for key,value in zip(smooth.index.tolist(),smooth.values.tolist())}

    # Replace each value by the according smoothed mean and return mapping dict

    return df[col].map(smooth), mapping_dict
# Compute the global mean

mean_label = train_df["label"].mean()

list_mapping_dict = []



for col in train_df.columns.tolist():

    encoded, mapping_dict = calc_smooth_mean(train_df, col, "label", mean_label, 300)

    downcasted = pd.to_numeric(encoded , downcast='float')

    train_df[col+'_tenc'] = downcasted

    list_mapping_dict.append(mapping_dict)
train_df4=train_df[30000001:35000000, :]
train_df4.to_csv("train_df4")