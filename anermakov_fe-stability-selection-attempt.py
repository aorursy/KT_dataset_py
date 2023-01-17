import pandas as pd

import numpy as np
df_train = pd.read_csv('../input/train.csv', index_col='id')

df_test = pd.read_csv('../input/test.csv', index_col='id')

df_train.shape, df_test.shape
target = 'target'



X_train = df_train.drop([target], axis=1).values

y_train = df_train[target]
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from stability_selection import StabilitySelection
Cs = np.logspace(-5, 5, 21)



pipe = Pipeline(steps=[

    ('scaler', StandardScaler()),

    ('clf', LogisticRegression(solver='lbfgs', verbose=1, n_jobs=1, random_state=42))

])



ss = StabilitySelection(base_estimator=pipe, 

                        lambda_name='clf__C', lambda_grid=Cs, 

                        n_jobs=-1, verbose=1, random_state=42)
%%time

ss.fit(X_train, y_train)
ss.stability_scores_[0]
ss
from stability_selection import plot_stability_path
fig, ax = plot_stability_path(ss, figsize=(12,8))

ax.set_xscale('log')
ss_indices = np.where(ss.get_support() == True)[0]

ss_indices
len(ss_indices)
X_train = ss.transform(X_train)

X_train.shape
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import RepeatedStratifiedKFold

from mlxtend.feature_selection import SequentialFeatureSelector
pipe = Pipeline(steps=[

    ('scaler', StandardScaler()),

    ('clf', LogisticRegression(solver='lbfgs', verbose=1, n_jobs=1, random_state=42))

])



cv = RepeatedStratifiedKFold(2, 50, random_state=42)



sfs = SequentialFeatureSelector(pipe, k_features=(5,10), floating=True,

                                scoring='roc_auc', cv=cv, verbose=1, n_jobs=-1)
%%time

sfs.fit(X_train, y_train)
for idx in sfs.k_feature_idx_:

    print(ss_indices[idx])
len(sfs.k_feature_names_)
import matplotlib.pyplot as plt
scores = [v['avg_score'] for k, v in sfs.subsets_.items()]

plt.plot(scores)
X_train = sfs.transform(X_train)

X_train.shape
from sklearn.model_selection import GridSearchCV
Cs = np.logspace(-5, 5, 21)



params = {

    'clf__C': Cs

}



pipe = Pipeline(steps=[

    ('scaler', StandardScaler()),

    ('clf', LogisticRegression(solver='lbfgs', verbose=1, n_jobs=1, random_state=42))

])



cv = RepeatedStratifiedKFold(2, 100, random_state=42)



grid = GridSearchCV(estimator=pipe, param_grid=params, cv=cv,

                    return_train_score=True,

                    scoring='roc_auc', verbose=1, n_jobs=-1)
%%time

grid.fit(X_train, y_train)
np.max(grid.cv_results_['mean_test_score'])
pipe = grid.best_estimator_

pipe
import matplotlib.pyplot as plt
plt.semilogx(Cs, grid.cv_results_['mean_train_score'], label='train')

plt.semilogx(Cs, grid.cv_results_['mean_test_score'], label='test')

plt.xlabel('C')

plt.ylabel('ROC-AUC')

plt.grid()

plt.legend()
X_test = df_test.values

for t in [ss, sfs]:

    X_test = t.transform(X_test)
pipe.fit(X_train, y_train)
y_pred = pipe.predict_proba(X_test)[:, 1]
plt.hist(y_pred, bins=50)

pass
df_subm = pd.read_csv('../input/sample_submission.csv', index_col='id')
df_subm[target] = y_pred
df_subm.head()
df_subm.to_csv('submission.csv')