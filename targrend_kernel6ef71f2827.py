# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. wpd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from skopt import gp_minimize

classifier_output = pd.read_csv('/kaggle/input/tractable_ds_excercise_data/classifier_output.csv')

print(f'Classifier output data shape:  {classifier_output.shape}')

classifier_output.head(10)
# Rows with no urr_score aren't needed for this analysis.  We're missing a lot for some reason

classifier_output.dropna(inplace=True)

print(f'Classifier output data shape without nans:  {classifier_output.shape}')
# Get line data



metadata_files = []

for dirname, _, filenames in os.walk('/kaggle/input/tractable_ds_excercise_data/metadata'):

    for filename in filenames:

        metadata_files.append(os.path.join(dirname, filename))



line_data = pd.concat([pd.read_csv(filepath) for filepath in metadata_files])

print(f'Line data shape: {line_data.shape}')

line_data.head(10)
# We seem to be missing about 5000 of the promised claims - perhaps ones where no repairs or replacements were made



print(f'Unique claims: {len(line_data["claim_id"].unique())}')
# Merge the claim-level data first, and then the line-level data

claim_merged = classifier_output.merge(line_data[['claim_id', 'make', 'model', 'year','poi']].drop_duplicates(subset=['claim_id'], keep='first'),

                                       how='left', on='claim_id')



print(f'Classifier outputs not associated with a claim: {claim_merged["make"].isna().sum()}')

# Remove any classifier outputs that can't be associated with a claim

claim_merged.dropna(subset=['make'], inplace=True)



data = pd.merge(claim_merged, line_data[['claim_id', 'line_num', 'part', 'operation', 'part_price', 'labour_amt']],

                how='left', on=['claim_id', 'part'])



data['operation'].fillna('undamaged', inplace=True)

print(f'Merge data shape: {data.shape}')

data.head(10)
# Visualise the effectiveness of the classifier on the test set



data['rounded_urr_score'] = data['urr_score'].apply(lambda x: round(x, 2))



bucket_counts = (data[(data['set']==2)][['rounded_urr_score', 'operation', 'urr_score']]

                 .groupby(['rounded_urr_score', 'operation'])

                 .count()

                 .reset_index()

                 .rename(columns={'urr_score': 'count'})

                 .set_index('rounded_urr_score')

                 .pivot(columns='operation', values='count')

                 .fillna(0)

                )



bucket_counts = bucket_counts[['undamaged', 'repair', 'replace']]



bucket_counts.head(10)
bucket_counts.sum(axis=1).plot.bar()
bucket_counts_divided = bucket_counts.divide(bucket_counts.sum(axis=1), axis=0)



bucket_counts_divided.plot.area()
operation_ranks = {'undamaged': 0,

                   'repair': 1,

                   'replace': 2}



data['operation_rank'] = data['operation'].apply(lambda x: operation_ranks[x])



def mae_single_point(urr_score, operation_rank, repair_threshold, replace_threshold):

    classified_outcome_rank = int(urr_score > repair_threshold) + int(urr_score > replace_threshold)



    return abs(classified_outcome_rank - operation_rank)



assert(mae_single_point(0.9, 0, 0.4, 0.7) == 2)

assert(mae_single_point(0.5, 1, 0.4, 0.7) == 0)

assert(mae_single_point(0.5, 2, 0.4, 0.7) == 1)

    
def mae_dataset(data, repair_threshold, replace_threshold):

    class_maes =[]

    for i in range(2):

        class_data = data[(data['operation_rank']==i)]

        class_mae = sum(class_data

                        .apply(lambda row: mae_single_point(row['urr_score'], row['operation_rank'], repair_threshold, replace_threshold), axis=1))/len(class_data)

        class_maes.append(class_mae)

    total_mae = sum(class_maes)/3

    return total_mae
# Use only the test set to evaluate the best thresholds



test_set = data[(data['set']==2)][['urr_score', 'operation_rank']]



def mae(thresholds):

    return mae_dataset(test_set, thresholds[0], thresholds[1])



# Calculating mse is somewhat expensive at a couple of seconds a time, so use an optimizer and small number of iterations

# Takes about 2.5 minutes

opt = gp_minimize(mae, dimensions=[(0.0, 1.0, 'uniform'), (0.0, 1.0, 'uniform')], n_calls=50, verbose=True)



print(f'Best thresholds: {opt.x}')

print(f'Best average mse: {opt.fun}')