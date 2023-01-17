from scipy.stats import rankdata

from scipy.stats.mstats import gmean



LABELS = ["isFraud"]
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import glob

%matplotlib inline

from subprocess import check_output

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

all_files = glob.glob("../input/lgmodels/*.csv")

all_files



# Any results you write to the current directory are saved as output.
predict_list = []

predict_list.append(pd.read_csv('../input/lgmodels/Submission-.9433.csv')[LABELS].values)

predict_list.append(pd.read_csv('../input/lgmodels/submission-.9451.csv')[LABELS].values)

predict_list.append(pd.read_csv('../input/lgmodels/submission-.9459.csv')[LABELS].values)

predict_list.append(pd.read_csv('../input/lgmodels/submission-.9463.csv')[LABELS].values)

predict_list.append(pd.read_csv('../input/lgmodels/submission-0.9467.csv')[LABELS].values)

predict_list.append(pd.read_csv('../input/lgmodels/Submission-.9440.csv')[LABELS].values)

predict_list.append(pd.read_csv('../input/lgmodels/submission-.9454.csv')[LABELS].values)

predict_list.append(pd.read_csv('../input/lgmodels/submission-0.9466.csv')[LABELS].values)

predict_list.append(pd.read_csv('../input/lgmodels/submission-0.9475.csv')[LABELS].values)

predict_list.append(pd.read_csv('../input/lgmodels/submission-0.9433.csv')[LABELS].values)

predict_list.append(pd.read_csv('../input/lgmodels/submission-0.9468.csv')[LABELS].values)

predict_list.append(pd.read_csv('../input/lgmodels/submission-.9452.csv')[LABELS].values)

predict_list.append(pd.read_csv('../input/lgmodels/Submission-.9429.csv')[LABELS].values)

predict_list.append(pd.read_csv('../input/lgmodels/submission-.9449.csv')[LABELS].values)

predict_list.append(pd.read_csv('../input/lgmodels/submission-.9457.csv')[LABELS].values)

predict_list.append(pd.read_csv('../input/lgmodels/Submission-.9438.csv')[LABELS].values)

predict_list.append(pd.read_csv('../input/lgmodels/Submission-.9442.csv')[LABELS].values)

predict_list.append(pd.read_csv('../input/lgmodels/submission-.9469.csv')[LABELS].values)

import warnings

warnings.filterwarnings("ignore")

print("Rank averaging on ", len(predict_list), " files")

predictions = np.zeros_like(predict_list[0])

for predict in predict_list:

    for i in range(1):

        predictions[:, i] = np.add(predictions[:, i], rankdata(predict[:, i])/predictions.shape[0])  

predictions /= len(predict_list)



submission = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv')

submission[LABELS] = predictions

submission.to_csv('AggStacker.csv', index=False)
submission.head()
sub_path = "../input/lgmodels/"

all_files = os.listdir(sub_path)

all_files
import warnings

warnings.filterwarnings("ignore")

outs = [pd.read_csv(os.path.join(sub_path, f), index_col=0) for f in all_files]

concat_sub = pd.concat(outs, axis=1)

cols = list(map(lambda x: "mol" + str(x), range(len(concat_sub.columns))))

concat_sub.columns = cols

concat_sub.reset_index(inplace=True)

concat_sub.head()

ncol = concat_sub.shape[1]
# check correlation

concat_sub.iloc[:,1:ncol].corr()
concat_sub.describe()
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
rank = np.tril(concat_sub.iloc[:,1:].corr().values,-1)

m = (rank>0).sum()

m_gmean, s = 0, 0

for n in range(min(rank.shape[0],m)):

    mx = np.unravel_index(rank.argmin(), rank.shape)

    w = (m-n)/(m+n)

    print(w)

    m_gmean += w*(np.log(concat_sub.iloc[:,mx[0]+1])+np.log(concat_sub.iloc[:,mx[1]+1]))/2

    s += w

    rank[mx] = 1

m_gmean = np.exp(m_gmean/s)
# get the data fields ready for stacking

concat_sub['isFraud'] = m_gmean

concat_sub[['TransactionID','isFraud']].to_csv('stack_gmean.csv', 

                                        index=False, float_format='%.4g')
concat_sub['m_median'] = concat_sub.iloc[:, 1:ncol].median(axis=1)

concat_sub['isFraud'] = concat_sub['m_median']

concat_sub[['TransactionID','isFraud']].to_csv('stack_median.csv', 

                                        index=False, float_format='%.6f')