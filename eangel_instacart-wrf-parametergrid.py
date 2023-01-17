# Import necessary modules



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import sklearn

import datetime

import seaborn as sns



from scipy.stats import randint, geom



from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import ParameterGrid



from imblearn.metrics import geometric_mean_score
store = pd.HDFStore('../input/io.h5', 'r')

store.open()

store.keys()



dsets = ['train',

         'test',

         'kaggle']



X = dict.fromkeys(dsets)

y = dict.fromkeys(dsets)



for ds in dsets:

    X[ds] = store['/X/' + str(ds)]

    y[ds] = store['/y/' + str(ds)]



store.close()

store.is_open
skew_train = ((y['train'].count() - y['train'].sum())/y['train'].sum())



skew_test = ((y['test'].count() - y['test'].sum())/y['test'].sum())



print('Ratio of negative class to positive class: Skew = %.4f' % skew_train)
wrfc = RandomForestClassifier(n_estimators=30,

                              class_weight='balanced',

                                 n_jobs=-1,

                                 oob_score=True,

                                 random_state=20190511)
# current grid

# (5,3) 'min_impurity_decrease' vs 'min_samples_leaf' or 'max_features'

param1 = 'min_impurity_decrease'

param2 = 'min_samples_leaf'

# param2 = 'max_features'



param_grid = {

    param1: np.geomspace(1e-8, 1e-6, 5),

    param2: np.around(np.geomspace(12, 48, 3)).astype(int)

#    param2: np.arange(6, 9)    

}
# # Initial coarse search parameters (superset) from another notebook:

# param_grid = {

# #    "max_depth": [4, 8, 16, 32], # does not seem helpful

#     "max_features": [5, 6, 7, 8, 9],

#     "min_samples_leaf": [4, 8, 16, 32],

#     "criterion": ["gini", "entropy"] # just use gini, should not matter

# }
# # (5,3) 'min_impurity_decrease' vs 'max_features'

# param1 = 'min_impurity_decrease'

# param2 = 'max_features'



# param_grid = {

#     param1: np.geomspace(1e-8, 1e-6, 5),

#     param2: np.arange(6, 9)

# }
# # (3,3) 'min_impurity_decrease' vs 'max_features'

# param1 = 'min_impurity_decrease'

# param2 = 'max_features'



# param_grid = {

#     param1: np.geomspace(1e-8, 1e-6, 3),

#     param2: np.arange(6, 9)

# }
# # testing plots

# param1 = 'min_impurity_decrease'

# param2 = 'max_features'



# param_grid = {

#     param1: [1e-6, 1e-5],

#     param2: [6, 7]

# }
# Double check which param_grid we set:

param_grid
rows = []

cols = []

decisions = []

sizes = []



for params in ParameterGrid(param_grid):

    print(datetime.datetime.now())

    print(params)

    wrfc.set_params(**params)

    wrfc.fit(X['train'], y['train'].values.ravel())

    rows.append(params[param1])

    cols.append(params[param2])

    decisions.append(wrfc.oob_decision_function_[:, 1])

    sizes.append([wrfc.estimators_[i].tree_.node_count

                  for i in range(len(wrfc.estimators_))])

    

print(datetime.datetime.now())
print('The number of NaNs is: %i' % np.isnan(decisions[0]).sum())

print('The percentage of NaNs is: {:.4%}'.format(np.isnan(decisions[0]).sum()/len(decisions[0])))
# Create a dictionary of metrics to compute multiple scores



metrics_dict = {}





metrics_dict['auc_roc'] = {'fcn' : metrics.roc_auc_score,

                        'name': 'AUC-ROC',

                        'thr' : False}



metrics_dict['auc_pr'] = {'fcn' : metrics.average_precision_score,

                        'name': 'AUC-PR',

                        'thr' : False}



metrics_dict['log_loss'] = {'fcn' : metrics.log_loss,

                        'name': 'Log Loss',

                        'thr' : False}



metrics_dict['prec'] = {'fcn' : metrics.precision_score,

                        'name': 'Precision',

                        'thr' : True}



metrics_dict['rec'] = {'fcn' : metrics.recall_score,

                        'name': 'Recall',

                        'thr' : True}



metrics_dict['f1'] = {'fcn' : metrics.f1_score,

                        'name': 'F1 Score',

                        'thr' : True}



metrics_dict['bal_acc'] = {'fcn' : metrics.balanced_accuracy_score,

                        'name': 'Balanced Accuracy',

                        'thr' : True}



metrics_dict['g_mean'] = {'fcn' : geometric_mean_score,

                        'name': 'Geometric Mean',

                        'thr' : True}



metrics_dict['kappa'] = {'fcn' : metrics.cohen_kappa_score,

                        'name': 'Cohen\'s Kappa',

                        'thr' : True}
def compute_score(y_true, y_proba, metric, threshold=0.5):

    """Computes score given metric dict as above

    (i.e. metric.keys() == ('fcn', 'name', 'thr'))"""

    

    y_proba_nonan = y_proba[~np.isnan(y_proba)]

    y_true_nonan = y_true[~np.isnan(y_proba)]

    

    if metric['thr'] == True:

        return metric['fcn'](y_true_nonan, (y_proba_nonan >= threshold))

    elif metric['thr'] == False:

        return metric['fcn'](y_true_nonan, y_proba_nonan)

    else:

        return np.NaN

    # try/except?
from scipy.optimize import brentq
# Find the binary threshold which reproduces skew_train

# for decisions array.

# Empirically, a sample of 100,000 seems sufficient for 

# three figures of precision in threshold



thresholds = []



def threshold_skewer(x, decision, sample=100000):

    decisions_sample = np.random.choice(decisions[decision], sample)

    return sum(decisions_sample < x) / (sum(decisions_sample >= x)) - skew_train



for k in range(len(decisions)):

    thresholds.append(brentq(threshold_skewer, 0, 1, args=k))
# Compute scores and concatenate with corresponding parameter values



names = []

scores = []



for key, val in metrics_dict.items():

    names.append(val['name'])    

    for k in range(len(decisions)):

        scores.append(compute_score(y['train'].values.ravel(),

                                    decisions[k],

                                    val,

                                    thresholds[k]))



scores = np.array(scores).reshape((len(metrics_dict.keys()),

                                   len(decisions))).T



scores_df = pd.concat([pd.DataFrame(data=np.array([rows, cols]).T,

             columns=[param1, param2]), 

    pd.DataFrame(data=scores,

             columns=names)],

            axis=1).melt(id_vars=[param1, param2])



scores_df.rename(columns={'variable': 'score'}, inplace=True)
def draw_heatmap(index, columns, values, **kwargs):

    data = kwargs.pop('data')

    d = data.pivot(index=index,

                   columns=columns,

                   values=values)

    sns.heatmap(d, **kwargs)



fg = sns.FacetGrid(scores_df,

                   col='score',

                   col_wrap=3,

                   height=4)



fg.map_dataframe(draw_heatmap,

                 index=param1, 

                 columns=param2,

                 values='value',

                 square=True,

                 annot=True,

                 fmt='0.3f')

plt.show()
# Construct long DataFrames from (samples of) p-r curve arrays

pr_dfs_list = []

y_true = y['train'].values.ravel()



for counter, params in enumerate(ParameterGrid(param_grid)):

    pr_dict = {}

    y_proba = decisions[counter]

    y_proba_nonan = y_proba[~np.isnan(y_proba)]

    y_true_nonan = y_true[~np.isnan(y_proba)]

    precision, recall, _ = metrics.precision_recall_curve(y_true_nonan, y_proba_nonan)

    pr_dict[param1] = params[param1]

    pr_dict[param2] = params[param2]

    pr_dict['precision'] = precision[::1000]

    pr_dict['recall'] = recall[::1000]

    pr_dfs_list.append(pd.DataFrame(pr_dict))
palette = sns.color_palette("Set2", len(param_grid[param1]))



plt.figure(figsize=(12,12))

sns.lineplot(data=pd.concat(pr_dfs_list),

             x='recall',

             y='precision',

             style=param2,

             hue=param1,

             palette=palette

             )

plt.title('Precision-Recall Curves')

plt.show()
# Construct long DataFrames from (samples of) roc curve arrays

roc_dfs_list = []

y_true = y['train'].values.ravel()



for counter, params in enumerate(ParameterGrid(param_grid)):

    roc_dict = {}

    y_proba = decisions[counter]

    y_proba_nonan = y_proba[~np.isnan(y_proba)]

    y_true_nonan = y_true[~np.isnan(y_proba)]

    fpr, tpr, _ = metrics.roc_curve(y_true_nonan, y_proba_nonan)

    roc_dict[param1] = params[param1]

    roc_dict[param2] = params[param2]

    roc_dict['fpr'] = fpr[::1000]

    roc_dict['tpr'] = tpr[::1000]

    roc_dfs_list.append(pd.DataFrame(roc_dict))
palette = sns.color_palette("Set2", len(param_grid[param1]))



plt.figure(figsize=(12,12))

sns.lineplot(data=pd.concat(roc_dfs_list),

             x='fpr',

             y='tpr',

             style=param2,

             hue=param1,

             palette=palette

             )

plt.title('Receiver Operating Characteristic Curves')

plt.show()