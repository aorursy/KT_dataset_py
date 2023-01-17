%load_ext autoreload

%autoreload 2



import bayes_classifier as bc

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score





import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

from sklearn.neighbors import KernelDensity

from scipy import stats

import os



warnings.filterwarnings("ignore")
DATA_DIR = "../input/santander-customer-transaction-prediction-dataset"

SUBMISSION_DIR = "."



SAMPLE_RATIO = None

TRAIN_SPLIT = 0.75

TARGET_COL = "target"

ID_COL = "ID_code"
train_val = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))

train, val = train_test_split(train_val, test_size=0.2)



VAR_COLS = list(train.columns)

VAR_COLS.remove(TARGET_COL)

VAR_COLS.remove(ID_COL)
train.head()
train.describe()
print("Original train size: {}".format(len(train_val)))



print("[SAMPLED]\nTrain size: {}\nValidation size: {}\n".format(len(train), len(val)))



train.set_index(ID_COL, inplace=True)

val.set_index(ID_COL, inplace=True)

train_val.set_index(ID_COL, inplace=True)



train, val = train_test_split(train, train_size=TRAIN_SPLIT)



if SAMPLE_RATIO is not None:

    val = val[:int(len(val)*SAMPLE_RATIO)]



print("[NEW]\nTrain size: {}\nValidation size: {}\n".format(len(train), len(val)))
SAMPLE = 9

BINS = 501 # bins for distribution



subplot_x_y = (4, int(SAMPLE/4)+1)

plt.figure(figsize=(15,15))





for i in range(SAMPLE):

    plt.subplot(subplot_x_y[0], subplot_x_y[1], i+1)

    

    sns.distplot(train[train.target == 0]["var_" + str(i)], bins=BINS, label="P(var_{}|0)".format(i))

    sns.distplot(train[train.target == 1]["var_" + str(i)], bins=BINS, label="P(var_{}|1)".format(i))

    sns.distplot(train["var_" + str(i)], bins=200, label="P(var_{})".format(i))

    

    plt.legend()
reverse_bayes = bc.ReverseBayes(TARGET_COL, VAR_COLS)
%%time

savgol_params = {"savgol_num": 101}

# p1, val1 = reverse_bayes.fit(train, 

#                              rolling_window=50,

#                              smoothing_method="savgol",

#                              smoothing_params=savgol_params)
# import random

# import warnings

# warnings.filterwarnings("ignore")



# from scipy.signal import savgol_filter, resample

# from scipy.interpolate import interp1d



# SAMPLE = random.sample(range(0, 200), 15)

# BINS_P = 50



# subplot_x_y = ((int(len(SAMPLE)/4)+1)*2,4)

# plt.figure(figsize=(25, 25))



# for i, sample in enumerate(SAMPLE):

    

#     plt.subplot(subplot_x_y[0], subplot_x_y[1], 2*i+1)

#     x1 = val1[sample]

#     y1 = p1[sample]

#     plt.plot(x1, y1, label="P(var_{}|1)".format(sample))

#     plt.legend()

    

#     plt.subplot(subplot_x_y[0], subplot_x_y[1], 2*i+2)

#     sns.distplot(train[train.target == 0]["var_" + str(sample)], bins=BINS_P, label="P(var_{}|0)".format(sample))

#     sns.distplot(train[train.target == 1]["var_" + str(sample)], bins=BINS_P, label="P(var_{}|1)".format(sample))

#     plt.legend()

    

# plt.show()
# _y = reverse_bayes.transform(val[val.columns[1:]])

# print("test")

# eval_dict = reverse_bayes.evaluate(val["target"], _y)

# eval_dict
# best_threshold, _ = reverse_bayes.find_threshold(val["target"], _y)
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

test.set_index(ID_COL, inplace=True)



if SAMPLE_RATIO is not None:

    test = test[:int(len(test)*SAMPLE_RATIO)]
reverse_bayes.fit(train_val, 

                  rolling_window=50,

                  smoothing_method="savgol",

                  smoothing_params=savgol_params)

_y = reverse_bayes.transform(test)
reverse_bayes.save_submission(_y, )