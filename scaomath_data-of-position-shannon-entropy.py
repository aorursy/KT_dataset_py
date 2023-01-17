!conda install -y -c bioconda forgi

!conda install -y -c bioconda viennarna
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        os.path.join(dirname, filename)



import json

from tqdm import tqdm

import shutil

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc

import matplotlib.pyplot as plt

%matplotlib inline



import RNA
pretrain_dir = None

verbose = 0



one_fold = False # if True, train model at only first fold. use if you try a new idea quickly.

run_test = False # if True, use small data. you can check whether this code run or not

if run_test: verbose = 1

denoise = True # if True, use train data whose signal_to_noise > sn_threshold

sn_threshold = 1

bpps_threshold = 5e-2

n_fold = 5



train = pd.read_json("/kaggle/input/stanford-covid-vaccine/train.json",lines=True)

print(train[train.signal_to_noise>1].shape)

print(train[train.signal_to_noise>1.5].shape)

print(train.columns)



if denoise:

    train = train[train.signal_to_noise > sn_threshold].reset_index(drop = True)
import json

import glob



train = pd.read_json("/kaggle/input/stanford-covid-vaccine/train.json",lines=True)

if denoise:

    train = train[train.signal_to_noise > 1].reset_index(drop = True)

test  = pd.read_json("/kaggle/input/stanford-covid-vaccine/test.json",lines=True)

test_pub = test[test["seq_length"] == 107]

test_pri = test[test["seq_length"] == 130]

sub = pd.read_csv("/kaggle/input/stanford-covid-vaccine/sample_submission.csv")



if run_test: ## to test 

    train = train[:30]

    test_pub = test_pub[:30]

    test_pri = test_pri[:30]



As = []

for id in tqdm(train["id"]):

    a = np.load(f"/kaggle/input/stanford-covid-vaccine/bpps/{id}.npy")

    As.append(a)

As = np.array(As)

As_pub = []

for id in tqdm(test_pub["id"]):

    a = np.load(f"/kaggle/input/stanford-covid-vaccine/bpps/{id}.npy")

    As_pub.append(a)

As_pub = np.array(As_pub)

As_pri = []

for id in tqdm(test_pri["id"]):

    a = np.load(f"/kaggle/input/stanford-covid-vaccine/bpps/{id}.npy")

    As_pri.append(a)

As_pri = np.array(As_pri)
def get_entropy(df, As):

    positional_entropy = np.zeros((df.shape[0], As.shape[1], 1))

#     for i in tqdm(range(len(df))):

    for i in range(2):

        idx = df.index[i]

        fc = RNA.fold_compound(df['sequence'][idx])

        mfe_struct, mfe = fc.mfe()

        fc.exp_params_rescale(mfe)

        pp, fp = fc.pf()

        entropy = fc.positional_entropy()

        positional_entropy[i,:,0] = np.array(entropy)[1:]

    return positional_entropy



X_entropy = get_entropy(train, As)

X_entropy_pub = get_entropy(test_pub, As_pub)

X_entropy_pri = get_entropy(test_pri, As_pri)

np.savez_compressed('./data_entropy', 

                    X_entropy=X_entropy, 

                    X_entropy_pub=X_entropy_pub,

                    X_entropy_pri=X_entropy_pri)