%reset -sf
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))

        

import collections

import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix
PREDS_NPY_PATH1 = "/kaggle/input/week6-open-tpu-aug/xlm-roberta.npy"

PREDS_NPY_PATH2 = "/kaggle/input/week4/transformers-preds.npy"

test_preds = np.load(PREDS_NPY_PATH1)

test_preds2 = np.load(PREDS_NPY_PATH2)



test_preds = (0.5*test_preds + 0.5*test_preds2)

test_preds_original = test_preds.copy()

test_class = np.argmax(test_preds,axis=1)
test_class = np.argmax(test_preds,axis=1)

ax = scatter_matrix(pd.DataFrame(test_preds), alpha=1, figsize=(9, 9), diagonal='kde', c=test_class)

for i in range(5):

    for j in range(5):

        ax[i,j].set_xlim(0.0,1.0)

        if i != j:

            ax[i,j].set_ylim(0.0,1.0)

plt.show()
plt.scatter(test_preds[:,3], test_preds[:,4], alpha=0.1, s=0.1, c=test_class)

plt.gca().plot(np.linspace(*plt.gca().get_xlim()), np.linspace(*plt.gca().get_xlim()), color="red")

plt.show()
target_distirbution = [0.11388, 0.02350, 0.06051, 0.39692, 0.40519]



def scale_by_power(ratio):

    return np.argmax(np.power(test_preds_original, ratio), axis=1)



def scale_by_multiply(ratio):

    return np.argmax(np.multiply(test_preds_original, ratio), axis=1)



modification_function = scale_by_power  # change this if you want



def calc_class_share(test_class):

    c = collections.Counter(test_class)

    return [c[clf]/len(test_preds) for clf in range(5)]



def loss(ratio):

    test_class = modification_function(ratio)

    class_share = calc_class_share(test_class)

    loss = [abs(x-y) for x,y in zip(class_share, target_distirbution)]

    return sum(loss)



from scipy.optimize import minimize

res = minimize(loss, [1,1,1,1,1], method="Nelder-Mead")

res.fun, res.x  # loss
test_class = modification_function(res.x)

calc_class_share(test_class)
calc_class_share(modification_function([1,1,1,1,1]))
# test_class = np.argmax(test_preds,axis=1)

ax = scatter_matrix(pd.DataFrame(test_preds), alpha=1, figsize=(9, 9), diagonal='kde', c=test_class)

for i in range(5):

    for j in range(5):

        ax[i,j].set_xlim(0.0,1.0)

        if i != j:

            ax[i,j].set_ylim(0.0,1.0)

plt.show()
test = pd.read_csv("/kaggle/input/student-shopee-code-league-sentiment-analysis/test.csv")

test['rating'] = test_class + 1

# test.loc[test['rating'] == 5, 'rating'] = 4

test[['review_id', 'rating']].to_csv("submission.csv", index=False)
!head submission.csv
collections.Counter(test['rating'])