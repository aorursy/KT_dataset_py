# Import libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import string



plt.rcParams.update({'font.size': 14})



# Load data

train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

sub_sample = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")



print (train.shape, test.shape, sub_sample.shape)
leak = pd.read_csv("../input/disasters-on-social-media/socialmedia-disaster-tweets-DFE.csv", encoding='latin_1')

leak['target'] = (leak['choose_one']=='Relevant').astype(int)

leak['id'] = leak.index

leak = leak[['id', 'target','text']]

merged_df = pd.merge(test, leak, on='id')

sub1 = merged_df[['id', 'target']]

sub1.to_csv('submit.csv', index=False)