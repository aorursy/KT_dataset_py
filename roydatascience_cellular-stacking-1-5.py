import os

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
from scipy.stats import rankdata

import glob

LABELS = ["sirna"]

all_files = glob.glob("../input/cellstack/*.csv")

all_files
outs = [pd.read_csv(f, index_col=0) for f in all_files]

concat_sub = pd.concat(outs, axis=1)

cols = list(map(lambda x: "m" + str(x), range(len(concat_sub.columns))))

concat_sub.columns = cols

concat_sub.reset_index(inplace=True)
rank = np.tril(concat_sub.iloc[:,1:].corr().values,-1)

m = (rank>0).sum()

m_gmean, s = 0, 0

for n in range(min(rank.shape[0],m)):

    mx = np.unravel_index(rank.argmin(), rank.shape)

    w = (m-n)/(m+n/10)

    print(w)

    m_gmean += w*(np.log(concat_sub.iloc[:,mx[0]+1])+np.log(concat_sub.iloc[:,mx[1]+1]))/2

    s += w

    rank[mx] = 1

m_gmean = np.exp(m_gmean/s).clip(0.0,1.0)
predict_list = []

predict_list.append(pd.read_csv("../input/cellstack/submission-174.csv")[LABELS].values)

predict_list.append(pd.read_csv("../input/cellstack/submission-201.csv")[LABELS].values)

predict_list.append(pd.read_csv("../input/cellstack/submission-231.csv")[LABELS].values)
import warnings

warnings.filterwarnings("ignore")

print("Rank averaging on ", len(predict_list), " files")

predictions = np.zeros_like(predict_list[0])

for predict in predict_list:

    for i in range(1):

        predictions[:, i] = np.add(predictions[:, i], rankdata(predict[:, i])/predictions.shape[0])  



predictions = predictions /len(predict_list)



submission = pd.read_csv('../input/recursion-cellular-image-classification/sample_submission.csv')

submission[LABELS] = predictions

submission.to_csv('AggStacker.csv', index=False)
sub_path = "../input/cellstack"

all_files = os.listdir(sub_path)

all_files
import warnings

warnings.filterwarnings("ignore")

outs = [pd.read_csv(os.path.join(sub_path, f), index_col=0) for f in all_files]

concat_sub = pd.concat(outs, axis=1)

cols = list(map(lambda x: "var" + str(x), range(len(concat_sub.columns))))

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

concat_sub['m_max'] = concat_sub.iloc[:, 1:ncol].max(axis=1)

concat_sub['m_min'] = concat_sub.iloc[:, 1:ncol].min(axis=1)

concat_sub['m_median'] = concat_sub.iloc[:, 1:ncol].median(axis=1)
concat_sub.describe()
cutoff_lo = 0.8

cutoff_hi = 0.2
concat_sub['sirna'] = m_gmean.astype(int)

concat_sub[['id_code','sirna']].to_csv('stack_mean.csv', 

                                        index=False, float_format='%.6f')
concat_sub['sirna']  = concat_sub['m_median'].astype(int)

concat_sub[['id_code','sirna']].to_csv('stack_median.csv', 

                                        index=False, float_format='%.6f')
concat_sub['sirna']  = np.where(np.all(concat_sub.iloc[:,1:ncol] > cutoff_lo, axis=1), 1, 

                                    np.where(np.all(concat_sub.iloc[:,1:ncol] < cutoff_hi, axis=1),

                                             0, concat_sub['m_median']))

concat_sub[['id_code','sirna']].to_csv('stack_pushout_median.csv', 

                                        index=False, float_format='%.6f')
concat_sub['m_mean'] = m_gmean.astype(int)

concat_sub['sirna']  = np.where(np.all(concat_sub.iloc[:,1:ncol] > cutoff_lo, axis=1), 

                                    concat_sub['m_max'], 

                                    np.where(np.all(concat_sub.iloc[:,1:ncol] < cutoff_hi, axis=1),

                                             concat_sub['m_min'], 

                                             concat_sub['m_mean'])).astype(int)

concat_sub[['id_code','sirna']].to_csv('stack_minmax_mean.csv', 

                                        index=False, float_format='%.6f')
concat_sub['sirna'] = np.where(np.all(concat_sub.iloc[:,1:ncol] > cutoff_lo, axis=1), 

                                    concat_sub['m_max'], 

                                    np.where(np.all(concat_sub.iloc[:,1:ncol] < cutoff_hi, axis=1),

                                             concat_sub['m_min'], 

                                             concat_sub['m_median'])).astype(int)

concat_sub[['id_code','sirna']].to_csv('stack_minmax_median.csv', 

                                        index=False, float_format='%.6f')