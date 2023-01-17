# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from time import time

from typing import List, Tuple, Union

from functools import partial



from scipy.optimize import linprog

import scipy.stats as stats

from sklearn.datasets import load_breast_cancer

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import make_scorer

from sklearn.metrics import accuracy_score, f1_score

import seaborn as sns

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import holoviews as hv



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.hv.extension('bokeh')

hv.extension('bokeh')
def _unit_shadow_prices(

    model_metrics: pd.Series, peer_metrics: pd.DataFrame, greater_is_better: List[bool], compute_primal: bool = False

) -> np.ndarray:

    peer_metrics = (peer_metrics.where(lambda x: x.ne(model_metrics, 0), np.nan)

    .dropna())



    greater_is_better_weight = np.where(greater_is_better, 1, -1)

    inputs_outputs = greater_is_better_weight * np.ones_like(peer_metrics)



    # outputs - inputs

    A_ub = inputs_outputs * peer_metrics

    b_ub = np.zeros(A_ub.shape[0])



    # \sum chosen model inputs = 1

    A_eq = np.where(greater_is_better_weight < 0.0, model_metrics, 0).reshape(1, -1)

    b_eq = np.array(1.0).reshape(1, -1)



    # max outputs == min -outputs

    c = np.where(greater_is_better_weight >= 0.0, model_metrics, 0.0).reshape(1, -1)



    # compute dual

    dual_A_ub = np.vstack((A_ub, A_eq)).T

    dual_c = np.hstack((b_ub, b_eq.reshape(-1,))).T

    dual_b_ub = c.T



    dual_result = linprog(

        dual_c,

        A_ub=-dual_A_ub,

        b_ub=-dual_b_ub,

        bounds=[(0, None) for _ in range(dual_A_ub.shape[1] -1 )] + [(None, None)],

    )



    return dual_result.fun





def data_envelopment_analysis(

    validation_metrics: Union[pd.DataFrame, np.ndarray], greater_is_better: List = []

) -> pd.DataFrame:

    """

    :param validation_metrics: Metrics produced by __SearchCV

    :param greater_is_better: Whether that metric are to be considered inputs to decrease or outputs to increase

    :return: Shadow prices for comparing a model to is peers & Hypothetical Comparison Units to compare units

    """

    partialed_unit_shadow_scores = partial(

        _unit_shadow_prices,

        peer_metrics=validation_metrics,

        greater_is_better=greater_is_better,

    )

    efficiency_scores = pd.DataFrame(validation_metrics).apply(

        partialed_unit_shadow_scores, axis=1

    )



    return efficiency_scores

# get some data

X, y = load_breast_cancer(return_X_y=True)

# build a classifier

clf = MLPClassifier()

# specify parameters and distributions to sample from

param_dist = {

    "hidden_layer_sizes": [(10, 5, ), (10, ), (10, 5, 3, )],

    "learning_rate_init": stats.uniform(0, 1),

    "alpha": stats.uniform(1e-4, 1e0),

}

# run randomized search

n_iter_search = 100

scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score),  'F1': make_scorer(f1_score)}

random_search = RandomizedSearchCV(

    clf, param_distributions=param_dist, 

    scoring = scoring,

    n_iter=n_iter_search, 

    n_jobs=-1,

    refit = 'AUC',

    return_train_score=True

)

random_search.fit(X, y)

cv_results_df = pd.DataFrame(random_search.cv_results_)





# %%

metrics = [

    "mean_fit_time",

    "mean_score_time",

    "mean_test_Accuracy",

    "mean_test_AUC",

    "mean_test_F1",

]

metrics_greater_is_better = [False, False, True, True, True]

efficiency_scores = data_envelopment_analysis(

    validation_metrics=cv_results_df[metrics],

    greater_is_better=metrics_greater_is_better,

)
table = hv.Dataset(cv_results_df.loc[:, metrics])

matrix = hv.operation.gridmatrix(table)



top = hv.Dataset(cv_results_df.loc[:, metrics]

                   .assign(efficiency_scores=efficiency_scores)

                   .where(lambda df: df.efficiency_scores >= df.efficiency_scores.quantile(0.75))

                   .drop(columns=['efficiency_scores']))

best = hv.operation.gridmatrix(top)

(matrix * best).opts(title='Top 25% in Red', width=800, height=600)
(cv_results_df.loc[:, ['param_alpha', 'param_hidden_layer_sizes', 'param_learning_rate_init'] + metrics]

 .assign(efficiency_scores=efficiency_scores)

 .nlargest(3, 'efficiency_scores'))
(cv_results_df.loc[:, ['param_alpha', 'param_hidden_layer_sizes', 'param_learning_rate_init'] + metrics]

 .assign(efficiency_scores=efficiency_scores)

 .nsmallest(3, 'efficiency_scores'))
columns = ['RANK', 'METHOD', 'TOP 1 ACCURACY', 'TOP 5 ACCURACY', 'NUMBER OF PARAMS', 'PAPER TITLE', 'YEAR']

imagenet = pd.read_csv('/kaggle/input/papers-with-code-imagenet-rankings/efficiency_results (1).csv', usecols=columns)

imagenet.loc[:, 'NUMBER OF PARAMS'] = imagenet.loc[:, 'NUMBER OF PARAMS'].apply(lambda s: float(s[:-1]))

imagenet.loc[:, 'TOP 5 ACCURACY'] = imagenet.loc[:, 'TOP 5 ACCURACY'].apply(lambda s: float(s[:-1])/100).astype(float)

imagenet.loc[:, 'TOP 1 ACCURACY'] = imagenet.loc[:, 'TOP 1 ACCURACY'].apply(lambda s: float('.'.join(s[:-1].split(',')))/100).astype(float)



imagenet
imagenet_metrics = ['NUMBER OF PARAMS', 'TOP 1 ACCURACY','TOP 5 ACCURACY']

imagenet_metrics_greater_is_better = [False, True, True]

                    

imagenet_efficiency_scores = data_envelopment_analysis(

    validation_metrics=imagenet.loc[:, imagenet_metrics],

    greater_is_better=imagenet_metrics_greater_is_better,

)



table = hv.Dataset(imagenet.loc[:, imagenet_metrics])

matrix = hv.operation.gridmatrix(table)



top = hv.Dataset(imagenet.loc[:, imagenet_metrics]

                   .assign(efficiency_scores=imagenet_efficiency_scores)

                   .where(lambda df: df.efficiency_scores >= df.efficiency_scores.quantile(0.75))

                   .dropna()

                   .drop(columns=['efficiency_scores']))

best = hv.operation.gridmatrix(top)

(matrix * best).opts(title='Top 25% in Red', width=800, height=600)
(imagenet

 .assign(efficiency_scores=imagenet_efficiency_scores)

 .nlargest(3, 'efficiency_scores'))
(imagenet

 .assign(efficiency_scores=imagenet_efficiency_scores)

 .nsmallest(3, 'efficiency_scores'))