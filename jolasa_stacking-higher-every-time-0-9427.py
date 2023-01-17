# This Python 3 environment comes with many helpful analytics libraries installed

import os

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
sub_path = "../input/ieeesubmissions"

all_files = os.listdir(sub_path)

all_files
outs = [pd.read_csv(os.path.join(sub_path, f), index_col=0) for f in all_files]

concat_sub = pd.concat(outs, axis=1)

cols = list(map(lambda x: "ieee" + str(x), range(len(concat_sub.columns))))

concat_sub.columns = cols

concat_sub.reset_index(inplace=True)

concat_sub.head()

ncol = concat_sub.shape[1]
# check correlation

concat_sub.iloc[:,1:ncol].corr()
corr = concat_sub.iloc[:,1:7].corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
# get the data fields ready for stacking

concat_sub['ieee_max'] = concat_sub.iloc[:, 1:ncol].max(axis=1)

concat_sub['ieee_min'] = concat_sub.iloc[:, 1:ncol].min(axis=1)

concat_sub['ieee_mean'] = concat_sub.iloc[:, 1:ncol].mean(axis=1)

concat_sub['ieee_median'] = concat_sub.iloc[:, 1:ncol].median(axis=1)
cutoff_lo = 0.8

cutoff_hi = 0.2
concat_sub['isFraud'] = concat_sub['ieee_mean']

concat_sub[['TransactionID', 'isFraud']].to_csv('stack_mean.csv', 

                                        index=False, float_format='%.6f')
concat_sub['isFraud'] = concat_sub['ieee_median']

concat_sub[['TransactionID', 'isFraud']].to_csv('stack_median.csv', 

                                        index=False, float_format='%.6f')
concat_sub['isFraud'] = np.where(np.all(concat_sub.iloc[:,1:ncol] > cutoff_lo, axis=1), 1, 

                                    np.where(np.all(concat_sub.iloc[:,1:ncol] < cutoff_hi, axis=1),

                                             0, concat_sub['ieee_median']))

concat_sub[['TransactionID', 'isFraud']].to_csv('stack_pushout_median.csv', 

                                        index=False, float_format='%.6f')
concat_sub['isFraud'] = np.where(np.all(concat_sub.iloc[:,1:ncol] > cutoff_lo, axis=1), 

                                    concat_sub['ieee_max'], 

                                    np.where(np.all(concat_sub.iloc[:,1:ncol] < cutoff_hi, axis=1),

                                             concat_sub['ieee_min'], 

                                             concat_sub['ieee_mean']))

concat_sub[['TransactionID', 'isFraud']].to_csv('stack_minmax_mean.csv', 

                                        index=False, float_format='%.6f')
concat_sub['isFraud'] = np.where(np.all(concat_sub.iloc[:,1:ncol] > cutoff_lo, axis=1), 

                                    concat_sub['ieee_max'], 

                                    np.where(np.all(concat_sub.iloc[:,1:ncol] < cutoff_hi, axis=1),

                                             concat_sub['ieee_min'], 

                                             concat_sub['ieee_median']))

concat_sub[['TransactionID', 'isFraud']].to_csv('stack_minmax_median.csv', 

                                        index=False, float_format='%.6f')
concat_sub['isFraud'] = concat_sub['ieee0'].rank(method ='min') + concat_sub['ieee1'].rank(method ='min') + concat_sub['ieee2'].rank(method ='min') +concat_sub['ieee3'].rank(method ='min')

concat_sub['isFraud'] = (concat_sub['isFraud']-concat_sub['isFraud'].min())/(concat_sub['isFraud'].max() - concat_sub['isFraud'].min())

concat_sub.describe()

concat_sub[['TransactionID', 'isFraud']].to_csv('stack_rank.csv', index=False, float_format='%.8f')