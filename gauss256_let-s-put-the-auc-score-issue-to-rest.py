%matplotlib inline



import warnings



import matplotlib.pyplot as plt

import numpy as np

from scipy.stats import beta

import seaborn as sns

from sklearn.metrics import roc_auc_score



warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
def quantile_normalize(trg, ref):

    """Quantile normalize target ndarray to match distribution of reference ndarray"""

    

    # We sort the target in a way that lets us unsort it later

    idx = np.argsort(trg)

    

    # Calculate the percentile points of the target

    n_trg = len(trg)

    percs = np.arange(n_trg) * 100 / (n_trg - 1)

    

    # Calculate the probabilities at the same percentile points in the reference

    probs = np.percentile(ref, percs)

    

    # Now unsort those probabilities

    idx2 = np.argsort(idx)

    trg2 = probs[idx2]

    

    return trg2
n_patients = 3

experiment = 3

if experiment == 1:  # different class probs, same accuracy

    class_probs = [.1, .2, .3]

    a = [2, 2, 2]

    b = [3, 3, 3]

elif experiment == 2:  # same class probs, different accuracies

    class_probs = [.15, .15, .15]

    a = [2, 2, 2]

    b = [2, 4, 6]

elif experiment == 3:  # different class probs, different accuracies

    class_probs = [.15, .20, .25]

    a = [2, 2, 2]

    b = [3.5, 4.0, 4.5]
# Sanity check: generate draws and plot

fig = plt.figure(figsize=(9, 3))

ax = {}

for p in range(n_patients):

    if p == 0:

        ax[p] = plt.subplot(1, 3, p + 1)

    else:

        ax[p] = plt.subplot(1, 3, p + 1, sharey=ax[0])

    plt.xlim(0, 1)

    plt.title('Patient {} - Class 0'.format(p + 1))

    probs = beta.rvs(a[p], b[p], size=1000)

    sns.distplot(probs, kde=None, fit=beta)
n_points = [1000, 2000, 4000]

y_score = []

y_true = []

for p in range(n_patients):

    n_1 = int(n_points[p] * class_probs[p] + 0.5)

    n_0 = n_points[p] - n_1

    

    # Class 0 predictions

    y_s = list(beta.rvs(a[p], b[p], size=n_0))

    y_t = [0] * n_0

    

    # Class 1 predictions

    y_s += list(beta.rvs(b[p], a[p], size=n_1))

    y_t += [1] * n_1

    

    y_score.append(y_s)

    y_true.append(y_t)
fig = plt.figure(figsize=(9, 3))

ax = {}

for p in range(n_patients):

    plt.subplot(1, 3, p + 1)

    if p == 0:

        ax[p] = plt.subplot(1, 3, p + 1)

    else:

        ax[p] = plt.subplot(1, 3, p + 1, sharey=ax[0])

    plt.xlim(0, 1)

    plt.title('Patient {}'.format(p + 1))

    sns.distplot(y_score[p])
auc = {}

for p in range(n_patients):

    auc[p] = roc_auc_score(y_true[p], y_score[p])

    print('Patient {} AUC: {:.3f}'.format(p + 1, auc[p]))

auc_mean = sum(auc.values()) / len(auc)

print('--------------------')

print('Average AUC:   {:.3f}'.format(auc_mean))
y_score_all = np.concatenate(y_score)

y_true_all = np.concatenate(y_true)

auc_all = roc_auc_score(y_true_all, y_score_all)

print('Combined AUC:  {:.3f}'.format(auc_all))
y_score_norm = []

for p in range(n_patients):

    y_score_norm.append(quantile_normalize(y_score[p], y_score[2]))

y_score_all_norm = np.concatenate(y_score_norm)

auc_all_norm = roc_auc_score(y_true_all, y_score_all_norm)

print('Normed AUC:    {:.3f}'.format(auc_all_norm))