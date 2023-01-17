from scipy.stats import spearmanr, rankdata

import pandas as pd

import numpy as np

import pickle
data_folder = "../input/ieee-fraud-detection/"

sub = pd.read_csv(f'{data_folder}sample_submission.csv')


with open("../input/ieee-sol1/test_uid_nocoverage.p", "rb") as f:

    test_uid_nocoverage = pickle.load(f)
sub1 = rankdata(pd.read_csv("../input/ieee-submissions-and-uids/final_model_blend.csv").isFraud)

sub1 /= len(sub1)

sub2 = rankdata(pd.read_csv("../input/ieee-sol1/blend_of_blends_1.csv").isFraud)

sub2 /= len(sub2)
spearmanr(sub1, sub2)
#sub3 = np.mean([rankdata(sub1.isFraud) / len(sub1), rankdata(sub2.isFraud) / len(sub2)], axis=0)

#sub3 = 0.6*rankdata(sub1.isFraud) / len(sub1) + 0.4*rankdata(sub2.isFraud) / len(sub2)
sub3 = sub1

sub3[test_uid_nocoverage] = 0.5 * sub1[test_uid_nocoverage] + 0.5 * sub2[test_uid_nocoverage]
sub['isFraud'] = sub3

sub.to_csv('submission.csv', index=False)



sub.head()