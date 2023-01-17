# import packages

import csv

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

import seaborn as sns

import pickle

import gzip

import json
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
for col in train_df.columns:

    print(col, train_df[col].isnull().sum())
for i, col in enumerate(test_df.columns.tolist()):

    encoded=test_df[col].map(list_mapping_dict[i])

    downcasted = pd.to_numeric(encoded , downcast='float')

    test_df[col+'_tenc']=downcasted
for col in test_df.columns:

    print(col, test_df[col].isnull().sum(), test_df[col].nunique())
test_df.to_csv("test_df.csv")
test_df.info()
# export mapping dict for datatypes

test_df.dtypes.to_csv("dtype_mapping.csv")



# export mapping dict for target encoding

with open('list_mapping_dict.json', 'w') as f:

    json.dump(list_mapping_dict, f)
with gzip.open('train_df.pkl', 'wb') as outputfile:

    pickle.dump(train_df, outputfile, protocol=-1)

    

with gzip.open('test_df.pkl', 'wb') as outputfile2:

    pickle.dump(test_df, outputfile2, protocol=-1)