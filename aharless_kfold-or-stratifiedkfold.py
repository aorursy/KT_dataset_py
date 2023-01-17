import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import KFold, StratifiedKFold
trn = pd.read_csv("../input/train.csv")

target = trn.target.copy()

target.sort_values(inplace=True)
import matplotlib.pyplot as plt

%matplotlib inline

folds = KFold(n_splits=3, shuffle=True, random_state = 45)

for trn_idx, val_idx in folds.split(target, target):

    plt.plot(np.hstack((target.iloc[trn_idx].values, target.iloc[val_idx].values)))

    plt.title("KFold Shuffle=True ?")

    plt.show()
folds = StratifiedKFold(n_splits=3, shuffle=True, random_state = 5)

for trn_idx, val_idx in folds.split(target, target):

    plt.plot(np.hstack((target.iloc[trn_idx].values, target.iloc[val_idx].values)))

    plt.title("StratifiedKFold Shuffle=True ?")

    plt.show()