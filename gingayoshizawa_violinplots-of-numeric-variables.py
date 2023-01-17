import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
sns.set_style("whitegrid")

# get input files
train = pd.read_csv("../input/train.csv", index_col=0)
test = pd.read_csv("../input/test.csv", index_col=0)

# concat train/test and make combi
combi = pd.concat([train, test], axis=0, ignore_index=True)
combi.drop('target', axis=1, inplace=True)

# get numeric columns
combi_numeric = []
for c in combi.columns:
    if combi[c].dtype == np.float64 or combi[c].dtype == np.int64:
        combi_numeric.append(c)

# get violin plot (only for 1st variable since its too many to show here)
sns.violinplot(x=None, y=combi_numeric[0], data=combi)
        
### (alternative1) get all plots
# for cn in combi_numeric:
#    sns.violinplot(x=None, y=cn, data=combi)
#    plt.show()

### (alternative2) tiled view (should work in python 2.7 but gets error with python 3)
# plt.figure(figsize=(20,50))
# for i, cn in enumerate(combi_numeric):
    # plt.subplot(len(combi_numeric)/6+1,6,i)
    # sns.violinplot(x=None, y=cn, data=combi)
# plt.tight_layout()

