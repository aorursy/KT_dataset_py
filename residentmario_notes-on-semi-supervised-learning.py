import seaborn as sns
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=5000, n_features=2, n_informative=2,
                           n_redundant=0, n_repeated=0, n_classes=3,
                           n_clusters_per_class=1,
                           weights=[0.01, 0.05, 0.94],
                           class_sep=0.8, random_state=0)

import matplotlib.pyplot as plt
colors = ['#ef8a62' if v == 0 else '#f7f7f7' if v == 1 else '#67a9cf' for v in y]
kwarg_params = {'linewidth': 1, 'edgecolor': 'black'}
fig = plt.Figure(figsize=(12,6))
plt.scatter(X[:, 0], X[:, 1], c=colors, **kwarg_params)
# from sklearn.semi_supervised import LabelPropagation

# trans = LabelPropagation()
# trans.fit(X_labeled, y_labeled)

# np.random.seed(42)
# y_unlabeled_pred = trans.predict(X_unlabeled)
from sklearn.semi_supervised import LabelPropagation
from sklearn.datasets import make_classification


def label(df, class_sep, trans=LabelPropagation()):
    """Label a point cloud with the given class separation """
    X, y = make_classification(n_samples=5000, n_features=2, n_informative=2,
                               n_redundant=0, n_repeated=0, n_classes=3,
                               n_clusters_per_class=1,
                               weights=[0.05, 0.10, 0.85],
                               class_sep=class_sep, random_state=0)
    
    X_l = X.shape[0]
    np.random.seed(42)
    unl_idx = np.random.randint(0, len(X), size=X_l - 500)
    l_idx = list(set(range(X_l)).difference(set(unl_idx)))
    X_unlabeled, X_labeled = X[unl_idx], X[l_idx]
    y_unlabeled, y_labeled = y[unl_idx], y[l_idx]
    
    trans = LabelPropagation()
    trans.fit(X_labeled, y_labeled)

    y_unlabeled_pred = trans.predict(X_unlabeled)
    
    r = (pd.DataFrame({'y': y_unlabeled, 'y_pred': y_unlabeled_pred}))
    
    return X_unlabeled, X_labeled, y_unlabeled, y_labeled, r
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

cmap = {0: '#ef8a62',1: '#f7f7f7', 2: '#67a9cf'}
fig, axarr = plt.subplots(2, 4, figsize=(12, 6))

for plot_n, sep in enumerate([0.1, 0.5, 1, 2]):
    X_unlabeled, X_labeled, y_unlabeled, y_labeled, r = label(X, sep, trans=LabelPropagation())

    for classnum, color in cmap.items():
        k_idx_labeled = np.where(y_labeled == classnum)[0]
        X_v, y_v = X_labeled[k_idx_labeled], y_labeled[k_idx_labeled]
        axarr[0][plot_n].plot(X_v[:, 0], X_v[:, 1], marker='o', linewidth=0, alpha=0.25, c=color)
        
        k_idx_unlabeled_misclassified = np.where(r.query("y == @classnum & y != y_pred").values)[0]
        X_v, y_v = X_unlabeled[k_idx_unlabeled_misclassified], y_unlabeled[k_idx_unlabeled_misclassified]
        axarr[0][plot_n].plot(X_v[:, 0], X_v[:, 1], marker='o', 
                              markeredgewidth=1, markeredgecolor='black', linewidth=0, c='None', zorder=10)
        
        result = r.rename_axis(None).groupby('y').apply(lambda df: (df.y == df.y_pred).sum() / len(df))
        result.plot.bar(color=cmap.values(), ax=axarr[1][plot_n])
        
plt.suptitle("$LabelPropagation$ Performance Benchmark, $Sep \in [0.1, 0.5, 1, 2]$")
from sklearn.semi_supervised import LabelSpreading

fig, axarr = plt.subplots(2, 4, figsize=(12, 6))

for plot_n, sep in enumerate([0.1, 0.5, 1, 2]):
    X_unlabeled, X_labeled, y_unlabeled, y_labeled, r = label(X, sep, trans=LabelSpreading())

    for classnum, color in cmap.items():
        k_idx_labeled = np.where(y_labeled == classnum)[0]
        X_v, y_v = X_labeled[k_idx_labeled], y_labeled[k_idx_labeled]
        axarr[0][plot_n].plot(X_v[:, 0], X_v[:, 1], marker='o', linewidth=0, alpha=0.25, c=color)
        
        k_idx_unlabeled_misclassified = np.where(r.query("y == @classnum & y != y_pred").values)[0]
        X_v, y_v = X_unlabeled[k_idx_unlabeled_misclassified], y_unlabeled[k_idx_unlabeled_misclassified]
        axarr[0][plot_n].plot(X_v[:, 0], X_v[:, 1], marker='o', 
                              markeredgewidth=1, markeredgecolor='black', linewidth=0, c='None', zorder=10)
        
        result = r.rename_axis(None).groupby('y').apply(lambda df: (df.y == df.y_pred).sum() / len(df))
        result.plot.bar(color=cmap.values(), ax=axarr[1][plot_n])
        
plt.suptitle("$LabelSpreading$ Performance Benchmark, $Sep \in [0.1, 0.5, 1, 2]$")
import pandas as pd
df = pd.read_csv("../input/openpowerlifting.csv")

# import missingno as msno
# msno.matrix(df.sample(1000))

import numpy as np
np.random.seed(42)

cols = ['Sex', 'Equipment', 'Age', 'BodyweightKg', 'BestSquatKg', 'BestBenchKg', 'BestDeadliftKg', 'TotalKg', 'Wilks', 'Division']

df_nonempty = (df
   .dropna(subset=['Division'])
   .drop(['MeetID', 'Name', 'Place', 'WeightClassKg'], axis='columns')
   .pipe(lambda df: df.assign(
         Sex=df.Sex.astype('category').cat.codes,
         Division=df.Division.astype('category').cat.codes,
         Equipment=df.Equipment.astype('category').cat.codes
   ))
   .pipe(lambda df: df[df.Division.isin(df.Division.value_counts().head(10).index.values)])
   .loc[:, cols]
   .dropna()
   .reset_index(drop=True)               
)
unl_idx = np.random.randint(0, len(df_nonempty), size=len(df_nonempty) - 500)
l_idx = list(set(range(len(df_nonempty))).difference(set(unl_idx)))

cols = [c for c in cols if c != 'Division']

X_unlabeled, X_labeled = df_nonempty.loc[unl_idx, cols].head(5000), df_nonempty.loc[l_idx, cols].head(5000)
y_unlabeled, y_labeled = df_nonempty.loc[unl_idx, 'Division'].head(5000), df_nonempty.loc[l_idx, 'Division'].head(5000)
from sklearn.semi_supervised import LabelPropagation

trans = LabelPropagation()
trans.fit(X_labeled, y_labeled)

np.random.seed(42)
y_unlabeled_pred = trans.predict(X_unlabeled)

print("Performance on division imputation: {0}% Accuracy".format((y_unlabeled_pred == y_unlabeled)\
                                                                 .pipe(lambda df: df.sum() / len(df))))