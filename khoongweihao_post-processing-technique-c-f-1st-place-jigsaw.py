# General imports

import numpy as np

import pandas as pd

import os

from matplotlib import pyplot as plt
WEIGHT = 1 # best to keep between 1 and 2 from the orignal authors
submission = pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')

test = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')

sub_best = pd.read_csv('../input/eda-modelling-of-the-external-data-inc-ensemble/external_meta_ensembled.csv')
files_sub = [

    '../input/minmax-ensemble-0-9526-lb/submission.csv',

    '../input/new-basline-np-log2-ensemble-top-10/submission.csv',

    '../input/stacking-ensemble-on-my-submissions/submission_mean.csv',

    '../input/analysis-of-melanoma-metadata-and-effnet-ensemble/ensembled.csv',

    '../input/eda-modelling-of-the-external-data-inc-ensemble/external_meta_ensembled.csv',

    '../input/submission-exploration/submission.csv',

    '../input/rc-fork-siim-isic-melanoma-384x384/sub_EfficientNetB2_384.csv',

    '../input/train-cv/submission.csv',

    '../input/triple-stratified-kfold-with-tfrecords/submission.csv',

    '../input/rank-then-blend/blend_sub.csv',

    '../input/siim-isic-melanoma-classification-ensemble/submission.csv'

]

files_sub = sorted(files_sub)

print(len(files_sub))

files_sub
for file in files_sub:

    test[file.replace(".csv", "")] = pd.read_csv(file).sort_values('image_name')["target"]

test['id'] = test.index
test.head()
test.columns
# Derive the given sub increases or decreases in score

test["diff_good1"] =  test['../input/rank-then-blend/blend_sub'] - test['../input/triple-stratified-kfold-with-tfrecords/submission']

test["diff_good1"] =  test['../input/train-cv/submission'] - test['../input/siim-isic-melanoma-classification-ensemble/submission']

test["diff_good2"] = test['../input/rc-fork-siim-isic-melanoma-384x384/sub_EfficientNetB2_384'] - test['../input/submission-exploration/submission']

test["diff_good3"] = test['../input/analysis-of-melanoma-metadata-and-effnet-ensemble/ensembled'] - test['../input/new-basline-np-log2-ensemble-top-10/submission']



test["diff_bad1"] = test['../input/stacking-ensemble-on-my-submissions/submission_mean'] - test['../input/minmax-ensemble-0-9526-lb/submission']
test["sub_best"] = test['../input/eda-modelling-of-the-external-data-inc-ensemble/external_meta_ensembled']

col_comment = ["id", "image_name", "patient_id", "sub_best"]

col_diff = [column for column in test.columns if "diff" in column]

test_diff = test[col_comment + col_diff].reset_index(drop=True)



test_diff["diff_avg"] = test_diff[col_diff].mean(axis=1) # the mean trend
# Apply the post-processing technique in one line (as explained in the pseudo-code of my post.

test_diff["sub_new"] = test_diff.apply(lambda x: (1+WEIGHT*x["diff_avg"])*x["sub_best"] if x["diff_avg"]<0 else (1-WEIGHT*x["diff_avg"])*x["sub_best"] + WEIGHT*x["diff_avg"] , axis=1)
submission["target"] = sub_best["target"]

submission.head()
test_diff.head()
submission.loc[test["id"], "target"] = test_diff["sub_new"].values
submission.to_csv("submission.csv", index=False)

submission.head()
plt.hist(submission.target,bins=100)

plt.show()