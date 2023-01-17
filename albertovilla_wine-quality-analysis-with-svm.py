# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
PATH_RED_WINE = '../input/wineQualityReds.csv'
PATH_WHITE_WINE = '../input/wineQualityWhites.csv'
reds = pd.read_csv(PATH_RED_WINE, index_col=0)
# Check distribution of quality classes
sns.catplot(x="quality", kind="count", palette="ch:.25", data=reds);
# Plot features
g = sns.PairGrid(reds)
g.map_diag(sns.kdeplot)
g.map_offdiag(plt.scatter);
# Split between features and label
y = reds['quality'].values
X = reds.drop(axis=1, labels=['quality'])
# Normalize features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X[X.columns] = scaler.fit_transform(X[X.columns])
sns.distplot(X['fixed.acidity']);
sns.distplot(X['alcohol']);
# Compute correlation matrix
corr = X.corr()

# Generate matrix for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

sns.set(style="white")

# Setup matplotlib figure
f, ax = plt.subplots(figsize=(11,9))

# Generate diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, mask=mask, square=True, cmap=cmap, center=0, linewidths=0.5, cbar_kws={"shrink": .5});
# Test and training validation sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
from sklearn.model_selection import GridSearchCV

parameters = {
    'kernel': ['linear', 'poly', 'rbf'],
    'C': [0.001, 0.01, 0.1, 1, 2, 5, 10, 20, 50, 100],
    'gamma': [0.01, 0.1, 0.5, 1]
}

svc = SVC(gamma='scale', random_state=8)
clf = GridSearchCV(svc, param_grid=parameters, cv=3)
clf.fit(X_train, y_train)
# Best estimator
clf.best_estimator_
print("Train: ", clf.score(X_train, y_train))
print("Test:  ", clf.score(X_test, y_test))
# Score versus C value
ax = sns.lineplot(x=clf.cv_results_['param_C'] ,y=clf.cv_results_['mean_test_score']);
ax.set(xlabel='C value', ylabel='Score')
plt.show()
# Score vs selected kernel
ax = sns.lineplot(x=clf.cv_results_['param_kernel'] ,y=clf.cv_results_['mean_test_score']);
ax.set(xlabel='Kernel', ylabel='Score')
plt.show()
# Score vs Gamma value
ax = sns.lineplot(x=clf.cv_results_['param_gamma'] ,y=clf.cv_results_['mean_test_score']);
ax.set(xlabel='Gamma', ylabel='Score')
plt.show()