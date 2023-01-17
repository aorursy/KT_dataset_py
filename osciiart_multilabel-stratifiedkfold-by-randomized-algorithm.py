# load libraries

import numpy as np

import pandas as pd

import time

import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold, StratifiedKFold
# set parameters

NUM_FOLD = 5

SEED = 42
# fix random seed

np.random.seed(SEED)
# load data

df_train = pd.read_csv("../input/train_curated.csv")

df_test = pd.read_csv("../input/sample_submission.csv")



# preprocess labels

labels = df_test.columns[1:].tolist()

NUM_CLASS = len(labels)



for label in labels:

    df_train[label] = df_train['labels'].apply(lambda x: label in x)

print(df_train.shape)

df_train.head()
# try KFold

folds = list(KFold(n_splits=NUM_FOLD, shuffle=True, random_state=SEED).split(np.arange(len(df_train))))

df_train['fold'] = 0

for i in range(NUM_FOLD):

    df_train['fold'][folds[i][1]] = i
# Check how well the folds are stratified.

print("fold                                         1    2    3    4    5   total")

print("==========================================================================")

for label in labels:

    label_padded = label + " "*(40-len(label))

    dist = ": "

    for i in range(NUM_FOLD):

        dist += "{:4d} ".format(df_train[label][folds[i][1]].sum())

    dist += "{:4d} ".format(df_train[label].sum())

    print(label_padded + dist)

label_padded = "total" + " "*(40-len("total"))

dist = ": "

for i in range(NUM_FOLD):

    dist += "{:4d} ".format(df_train.iloc[folds[i][1]].shape[0])

dist += "{:4d} ".format(df_train.shape[0])

print(label_padded + dist)
# calculate number of positive label for each sample

df_train['num_labels'] = df_train[labels].values.sum(axis=1)

df_train.loc[:,['labels', 'num_labels']].head(10)
# extract data sample with single label and do StratifiedKFold



df_train_single = df_train[df_train['num_labels']==1].reset_index(drop=True)

single_folds = list(StratifiedKFold(n_splits=NUM_FOLD, shuffle=True, random_state=SEED).split(

    np.arange(len(df_train_single)), df_train_single[labels].values.argmax(axis=1)))

df_train_single['fold'] = 0

for i in range(NUM_FOLD):

    df_train_single['fold'][single_folds[i][1]] = i
# Check how well the folds are stratified.

print("fold                                         1    2    3    4    5   total")

print("==========================================================================")

for label in labels:

    label_padded = label + " "*(40-len(label))

    dist = ": "

    for i in range(NUM_FOLD):

        dist += "{:4d} ".format(df_train_single[label][df_train_single['fold']==i].sum())

    dist += "{:4d} ".format(df_train_single[label].sum())

    print(label_padded + dist)

label_padded = "total" + " "*(40-len("total"))

dist = ": "

for i in range(NUM_FOLD):

    dist += "{:4d} ".format(df_train_single[df_train_single['fold']==i].shape[0])

dist += "{:4d} ".format(df_train_single.shape[0])

print(label_padded + dist)
# extract data sample with multi labels

df_train_multi = df_train[df_train['num_labels']!=1].reset_index(drop=True)



# count each label

label_counts = []

for i in range(NUM_CLASS):    

    label = labels[i] + " "*(40-len(labels[i]))

    label_counts.append(df_train_multi[labels[i]].sum())

    print("{:2d} {} {}".format(i, label, label_counts[i]))
reduced_label = np.zeros(len(df_train_multi), np.uint8)

for i in range(NUM_CLASS):

    target_idx = np.argsort(label_counts)[i]

    reduced_label[df_train_multi[labels[target_idx]]==1] = target_idx
# Do StratifiedKFold using reduced label

multi_folds = list(StratifiedKFold(n_splits=NUM_FOLD, shuffle=True, random_state=SEED).split(

    np.arange(len(df_train_multi)), reduced_label))

for i in range(NUM_FOLD):

    df_train_multi['fold'][multi_folds[i][1]] = i
# Check how well the folds are stratified.

print("fold                                         1    2    3    4    5   total")

print("==========================================================================")

for label in labels:

    label_padded = label + " "*(40-len(label))

    dist = ": "

    for i in range(NUM_FOLD):

        dist += "{:4d} ".format(df_train_multi[label][df_train_multi['fold']==i].sum())

    dist += "{:4d} ".format(df_train_multi[label].sum())

    print(label_padded + dist)

label_padded = "total" + " "*(40-len("total"))

dist = ": "

for i in range(NUM_FOLD):

    dist += "{:4d} ".format(df_train_multi[df_train_multi['fold']==i].shape[0])

dist += "{:4d} ".format(df_train_multi.shape[0])

print(label_padded + dist)
# concatenate single-label data and multi-label data

df_train2 = pd.concat([df_train_single, df_train_multi]).reset_index(drop=True)
# Check how well the folds are stratified.

print("fold                                         1    2    3    4    5   total")

print("==========================================================================")

for label in labels:

    label_padded = label + " "*(40-len(label))

    dist = ": "

    for i in range(NUM_FOLD):

        dist += "{:4d} ".format(df_train2[label][df_train2['fold']==i].sum())

    dist += "{:4d} ".format(df_train2[label].sum())

    print(label_padded + dist)

label_padded = "total" + " "*(40-len("total"))

dist = ": "

for i in range(NUM_FOLD):

    dist += "{:4d} ".format(df_train2[df_train2['fold']==i].shape[0])

dist += "{:4d} ".format(df_train2.shape[0])

print(label_padded + dist)
def calc_score(df):

    score = np.zeros([5,NUM_CLASS+1])

    for i in range(5):

        score[i] = df.loc[df.fold==i, labels+['num_labels']].values.sum(axis=0)

    score = score.std(axis=0).mean()

    return score

score = calc_score(df_train)

print("KFold score: {:.6f}".format(calc_score(df_train)))

print("StratifiedKFold score: {:.6f}".format(calc_score(df_train2)))
def do_optimize(df, size, steps):

    """

    df: dataframe to optimize folds

    size: number of data to change fold

    steps: number of for loop

    """

    starttime = time.time()

    score = calc_score(df)

    for i in range(steps):

        # select index to change fold

        change_idx = np.random.choice(np.arange(df.shape[0]), size, replace=False)

        # change fold randomly

        change_fold = np.random.randint(0, NUM_FOLD, size)

        df_new = df.copy()

        df_new['fold'][change_idx] = change_fold



        score_new = calc_score(df_new)

        if score_new < score: # if score getting small, folds will be update

            score = score_new

            df = df_new

        if i%100==0:

            print("step: {:4d}, change size: {:2d}, score: {:.6f}, sec: {:.1f}".format(

                i, size, score, time.time()-starttime))

    return df
# Let's do optimization with randomized algorithm.

df_train3 = df_train2.copy()

df_train3 = do_optimize(df_train3, size=64, steps=1000)

df_train3 = do_optimize(df_train3, size=32, steps=1000)

df_train3 = do_optimize(df_train3, size=16, steps=1000)

df_train3 = do_optimize(df_train3, size=8, steps=1000)

df_train3 = do_optimize(df_train3, size=4, steps=1000)

df_train3 = do_optimize(df_train3, size=2, steps=1000)

df_train3 = do_optimize(df_train3, size=1, steps=10000)



print("StratifiedKFold with randomized algorithm score: {:.6f}".format(calc_score(df_train3)))
# Check how well the folds are stratified.

print("fold                                         1    2    3    4    5   total")

print("==========================================================================")

for label in labels:

    label_padded = label + " "*(40-len(label))

    dist = ": "

    for i in range(NUM_FOLD):

        dist += "{:4d} ".format(df_train3[label][df_train3['fold']==i].sum())

    dist += "{:4d} ".format(df_train3[label].sum())

    print(label_padded + dist)

label_padded = "total" + " "*(40-len("total"))

dist = ": "

for i in range(5):

    dist += "{:4d} ".format(df_train3[df_train3['fold']==i].shape[0])

dist += "{:4d} ".format(df_train3.shape[0])

print(label_padded + dist)
# save

df_train3.to_csv("train_stratified.csv", index=None)