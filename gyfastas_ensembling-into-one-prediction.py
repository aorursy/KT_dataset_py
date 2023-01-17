## Environment ready

import os, sys

import pandas as pd

import numpy as np
sub0 = pd.read_csv('../input/lyftsubmission/Res50Base_model_0_submission.csv')

sub1 = pd.read_csv('../input/lyftsubmission/Res50Base_model_1_submission.csv')

sub2 = pd.read_csv('../input/lyftsubmission/Res50Base_model_2_submission.csv')

sub3 = pd.read_csv('../input/lyftsubmission/Res18Base_model0_submission.csv')

cols = list(sub0.columns)

confs = cols[2:5]

conf0 = cols[5:105]

conf1 = cols[105:205]

conf2 = cols[205:305]
## Averaging prediction as ensemble

pred0 = sub0[conf0].values

pred1 = sub1[conf0].values

pred2 = sub2[conf0].values

pred3 = sub3[conf0].values



pred_ensemble = (pred0 + pred1 + pred2 + pred3) / 4
sub0[conf0] = pred_ensemble
sub0.to_csv('submission.csv', index=False, float_format='%.6g') 