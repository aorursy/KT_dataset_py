import numpy as np

import pandas as pd



# visualize

import matplotlib.pyplot as plt

import matplotlib.style as style

import seaborn as sns

from matplotlib import pyplot

from matplotlib.ticker import ScalarFormatter

sns.set_context("talk")

style.use('fivethirtyeight')



from scipy import stats

from tqdm import tqdm_notebook as tqdm

import os
# load submission files



BASE_PATH = "../input/trends-assessment-prediction"

SUBMISSION_FILES_PATH = "../input/trends-ensemble-files"



submission = pd.read_csv(f"{BASE_PATH}/sample_submission.csv")



# high scoring public kernels

ENSEMBLES = [

    {"file0": f"{SUBMISSION_FILES_PATH}/submission4.csv", "weight": 1},

    {"file1": f"{SUBMISSION_FILES_PATH}/submission6.csv", "weight": 1},

    {"file2": f"{SUBMISSION_FILES_PATH}/submission7.csv", "weight": 1},

    {"file3": f"{SUBMISSION_FILES_PATH}/submission8.csv", "weight": 1},

    {"file4": f"{SUBMISSION_FILES_PATH}/submission9.csv", "weight": 1},

]



subs = submission.copy()

for i, ensemble in enumerate(ENSEMBLES):

    print(ensemble)

    tmp = pd.read_csv(ensemble[f'file{i}'])

    tmp.sort_values('Id', inplace=True)

    subs[f"predicted_file{i}"] = tmp["Predicted"]



subs.drop(columns=["Id", "Predicted"], inplace=True)
subs.head(10)
# Compute the correlation matrix

corr = subs.corr()



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(15, 12))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, cmap=cmap, annot=True, fmt="g",

            square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

ax.set_ylim(corr.shape[0], 0)

plt.yticks(rotation=0)
def mean_ensemble(row):

    return np.mean(row.values)
submission["Predicted"] = subs.apply(mean_ensemble, axis=1)

submission.to_csv("submission.csv", index=False)
submission.head()