# Standard imports

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
train_variants = pd.read_csv('../input/training_variants')

print('Number of training variants: %d' % (len(train_variants)))

train_variants.head()
test_variants = pd.read_csv('../input/test_variants')

print('Number of test variants: %d' % (len(test_variants)))

test_variants.head()
def read_textfile(filename):

    return pd.read_csv(filename, sep='\|\|', header=None, names=['ID', 'Text'], skiprows=1, engine='python')
train_text = read_textfile('../input/training_text')

print('Number of train samples: %d' % (len(train_text)))

train_text.head()
test_text = read_textfile('../input/test_text')

print('Number of test samples: %d' % (len(test_text)))

test_text.head()
train_df = pd.concat([train_text, train_variants.drop('ID', axis=1)], axis=1)

train_df.head()
test_df = pd.concat([test_text, test_variants.drop('ID', axis=1)], axis=1)

test_df.head()
print('[Train] Unique Genes Count: %d' % len(train_df.Gene.unique()))

print('[Train] Unique Variations Count: %d' % len(train_df.Variation.unique()))

print('[Test] Unique Genes Count: %d' % len(test_df.Gene.unique()))

print('[Test] Unique Variations Count: %d' % len(test_df.Variation.unique()))
train_df.Gene = pd.Categorical(train_df.Gene)
train_df['Gene_ID'] = train_df.Gene.cat.codes

train_df.head()