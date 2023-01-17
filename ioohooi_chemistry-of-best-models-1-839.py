import os

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

from scipy.stats.mstats import gmean

import seaborn as sns

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
%matplotlib inline
sub_path = "../input/stackingoutputs"

all_files = os.listdir(sub_path)

all_files
import warnings

warnings.filterwarnings("ignore")

outs = [pd.read_csv(os.path.join(sub_path, f), index_col=0) for f in ['stack_median_1_801.csv', 'stack_median_1_822.csv']]

concat_sub = pd.concat(outs, axis=1)

cols = list(map(lambda x: "mol" + str(x), range(len(concat_sub.columns))))

concat_sub.columns = cols

concat_sub.reset_index(inplace=True)

concat_sub.head()

ncol = concat_sub.shape[1]
concat_sub.head()
concat_sub['m_median'] = concat_sub.iloc[:, 1:].median(axis=1)
concat_sub['scalar_coupling_constant'] = concat_sub['m_median']
concat_sub['m_median_1_835'] = pd.read_csv("../input/stackingoutputs/stack_median_1_835.csv")["scalar_coupling_constant"]
concat_sub['scalar_coupling_constant'] = concat_sub[["m_median", "m_median_1_835"]].median(axis=1)
concat_sub.head()
train = pd.read_csv("../input/champs-scalar-coupling/train.csv")
train.head()
test = pd.read_csv("../input/champs-scalar-coupling/test.csv")
test.head()
concat_sub.info(verbose=True, null_counts=True)
test.info(verbose=True, null_counts=True)
concat_sub = pd.merge(concat_sub, test[['id', 'type']], on='id', how='left')
concat_sub.info(verbose=True, null_counts=True)
concat_sub.head()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 10))

sns.boxplot(x="type", y="scalar_coupling_constant", data=train, order=train["type"].unique(), ax=ax1)

sns.boxplot(x="type", y="scalar_coupling_constant", data=concat_sub, order=train["type"].unique(), ax=ax2)

ax1.grid(True)

ax2.grid(True)

plt.tight_layout()

plt.show()
concat_sub[['id', 'scalar_coupling_constant']].to_csv('stack_median.csv', index=False, float_format='%.6f')