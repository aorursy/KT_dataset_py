import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt



data = pd.read_csv("../input/aps_failure_training_set.csv", na_values="na")

data.head()
missing = data.isna().sum().div(data.shape[0]).mul(100).to_frame().sort_values(by=0, ascending = False)

missing.plot.bar(figsize=(50,10))

plt.show()
cols_missing = missing[missing[0]>75]

cols_missing
cols_to_drop = list(cols_missing.index) # list with columns to drop

cols_to_drop.append('class')

cols_to_drop
X = data.drop(cols_to_drop, axis=1)

y = data.loc[:,"class"]

y = pd.get_dummies(y).drop("neg",axis=1)
X.fillna(X.mean(), inplace=True)
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

scaler.fit(X)

X_scaled = scaler.transform(X)
from sklearn.metrics.scorer import make_scorer

from sklearn.metrics import confusion_matrix



def my_scorer(y_true,y_pred):

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    cost = 10*fp+500*fn

    return cost



my_func = make_scorer(my_scorer, greater_is_better=False)
from sklearn.svm import SVC

from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV



clf = SVC(probability = False, class_weight="balanced", gamma="auto") # initialising SVC classifier

pca = PCA() # initialising PCA component



pipe = Pipeline(steps=[("pca",pca), ("clf",clf)]) # creating pipeline



param_grid = {

    'pca__n_components': range(10,24),

    'clf__C': np.arange(0.2,0.5,0.05),

}



search = GridSearchCV(pipe, param_grid, iid=False, cv=3, return_train_score=False, scoring=my_func, n_jobs=-1, verbose=3) # Grid Search with 3-fold CV

search.fit(X_scaled, np.ravel(y))



# Plotting best classificator

print("Best parameters (CV score: {:0.3f}):".format(search.best_score_))

print(search.best_params_)
pca.fit(X_scaled)



fig, ax0 = plt.subplots(nrows=1, sharex=True, figsize=(12, 6))

ax0.plot(pca.explained_variance_ratio_, linewidth=2)

ax0.set_ylabel('PCA explained variance')

ax0.axvline(search.best_estimator_.named_steps['pca'].n_components, linestyle=':', label='n_components chosen')

ax0.legend(prop=dict(size=12))

plt.show()
search.best_estimator_
fig, ax1 = plt.subplots(nrows=1, sharex=True, figsize=(12, 6))



results = pd.DataFrame(search.cv_results_)

components_col = 'param_pca__n_components'

best_clfs = results.groupby(components_col).apply(lambda g: g.nlargest(1, 'mean_test_score'))



best_clfs.plot(x=components_col, y='mean_test_score', yerr='std_test_score',legend=False, ax=ax1)



ax1.set_ylabel('Classification accuracy (val)')

ax1.set_xlabel('n_components')



plt.tight_layout()

plt.show()