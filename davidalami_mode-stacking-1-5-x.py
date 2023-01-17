import os

import numpy as np 

import pandas as pd 

from scipy import stats

import warnings

warnings.filterwarnings("ignore")

import glob



all_files = glob.glob("../input/cellstack/*.csv")

all_files
outs = [pd.read_csv((f), index_col=0)['sirna'].values for f in all_files]

collected = np.array(outs)



# getting the mode

m = stats.mode(collected)[0][0]



submission = pd.read_csv('../input/recursion-cellular-image-classification/sample_submission.csv')

submission['sirna'] = m

submission.to_csv('ModeStacker.csv', index=False)