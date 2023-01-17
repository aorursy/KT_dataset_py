import pandas as pd

import numpy as np



import path

import seaborn as sns

from fastai import *

from fastai.text import *

from fastai.callbacks import *

from fastai.metrics import *



import re

path= Path('../input/nlp-getting-started')

path.ls()

path2= Path('../input/disasters-on-social-media')

path2.ls()
test_df = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
#importing full dataset

full = pd.read_csv(path2/'socialmedia-disaster-tweets-DFE.csv' , encoding='latin1')

full.head()
full = full[['choose_one', 'text']]

full['target'] = (full['choose_one']=='Relevant').astype(int)

full['id'] = full.index

full



merged_df = pd.merge(test_df, full, on='id')

merged_df



subm_df = merged_df[['id', 'target']]

subm_df



subm_df.to_csv('submission.csv', index=False)