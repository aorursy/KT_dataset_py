import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # showing and rendering figures
# io related
from skimage.io import imread
import os
from glob import glob
# not needed in Kaggle, but required in Jupyter
%matplotlib inline 
base_dir = '../input/'
train_df = pd.read_csv(os.path.join(base_dir, 'train.csv'))
test_df = pd.read_csv(os.path.join(base_dir, 'test.csv'))
train_df.sample(3)
train_df['landmark_id'].value_counts().hist()
submit_df = pd.read_csv(os.path.join(base_dir, 'sample_submission.csv'))
# take the most frequent label
def_guess = train_df['landmark_id'].value_counts()/train_df['landmark_id'].value_counts().sum()
def_guess.index[0], def_guess.values[0]
submit_df['landmarks'] = '%d %2.2f' % (def_guess.index[0], def_guess.values[0])
submit_df.to_csv('submission.csv', index=False)
submit_df.sample(2)
np.random.seed(2018)
r_idx = lambda : np.random.choice(def_guess.index, p = def_guess.values)
%%time
r_score = lambda idx: '%d %2.4f' % (def_guess.index[idx], def_guess.values[idx])
submit_df['landmarks'] = submit_df.id.map(lambda _: r_score(r_idx()))
submit_df.to_csv('rand_submission.csv', index=False)
submit_df.sample(2)
submit_df.sample(2)
