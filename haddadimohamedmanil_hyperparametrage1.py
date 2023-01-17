import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import make_scorer

from sklearn.metrics import r2_score

import seaborn as sns

from sklearn.model_selection import GridSearchCV

from sklearn import tree

from sklearn import metrics

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import VarianceThreshold

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from tqdm import tnrange, tqdm_notebook
def best_params(tab_score, gcv_tab):

    i = tab_score.index(max(tab_score))

    return gcv_tab[i].best_params_    
data = pd.read_hdf('../input/cleandata/clean.h5')

data.reset_index(inplace=True, drop=True)

# data.to_csv('clean.csv', index=False, sep=' ')

lc = data.columns.tolist()

del lc[14]

lc.append('sulfate_dose')

data = data[lc]



Y_traite = data.loc[:, 'ph_s1':'cl_s1'] # les target de ph_s1 à cl_s1

X_traite = data.drop(['ph_s1', 't_s1', 'cond_s1', 'turb_s1', 'cl_s1'], axis=1) # les variables data sans y_traite avec drop
X_train, X_test, y_train, y_test = train_test_split(X_traite, Y_traite.loc[:, 'cl_s1'], test_size=0.2, random_state=0)

scoring = make_scorer(r2_score)

tuned_parameters = {'max_depth':range(2,30), 'min_samples_split':range(2,30)}

cv3 = [2, 9]

tab_gcv3 = []

scores_cv3 = []

for i in cv3:

    g_cv3 = GridSearchCV(RandomForestRegressor(random_state=0), param_grid=tuned_parameters, scoring=scoring, cv=i, refit=True, n_jobs=-1, verbose=1)

    g_cv3.fit(X_train, y_train)

    tab_gcv3.append(g_cv3)

    scores_cv3.append(g_cv3.best_score_)

    

print('Le meilleur score est',best_params(scores_cv3, tab_gcv3))

print('Le fold est', cv3)

print('Les scores sont', scores_cv3)

print('cl_s1')