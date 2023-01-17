import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline
df_train = pd.read_csv('../input/train.csv', usecols=['ID', 'target'], index_col='ID')
df_train.head()
logy = np.log1p(df_train['target'].values)
def stratitied_folds(n_folds, y, random_state=42):
    sorted_y = np.argsort(y)

    folds = np.empty(len(y), dtype=np.int32)

    np.random.seed(random_state)

    for i in range(0, sorted_y.shape[0], n_folds):
        fold_idx = sorted_y[i:i+n_folds]
        np.random.shuffle(fold_idx)

        folds[fold_idx] = np.arange(len(fold_idx))
                    
    return folds
folds = stratitied_folds(5, logy)

df_folds = pd.DataFrame({'FOLD': folds}, index=df_train.index)
df_folds.to_csv('folds-5.csv')

plt.hist([logy[folds == f] for f in range(5)]);
folds = stratitied_folds(10, logy)

df_folds = pd.DataFrame({'FOLD': folds}, index=df_train.index)
df_folds.to_csv('folds-10.csv')

plt.hist([logy[folds == f] for f in range(10)]);