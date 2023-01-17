import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os 
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
comment_only = train[['comment_text', 'target']]
sex_comments = comment_only[comment_only['comment_text'].str.contains('sex')]

print("Mean toxicity of word 'Sex': {}".format(np.round(sex_comments['target'].mean(),2)))
plt.hist(sex_comments['target'])

plt.show()
damn_comments = comment_only[comment_only['comment_text'].str.contains('damn')]

print("Mean toxicity of word 'damn': {}".format(np.round(damn_comments['target'].mean(),2)))
plt.hist(damn_comments['target'])

plt.show()
god_comments = comment_only[comment_only['comment_text'].str.contains('God')]

print("Mean toxicity of word 'God': {}".format(np.round(god_comments['target'].mean(),2)))
plt.hist(god_comments['target'])

plt.show()
porn_comments = comment_only[comment_only['comment_text'].str.contains('porn')]

print("Mean toxicity of word 'porn': {}".format(np.round(porn_comments['target'].mean(),2)))
plt.hist(porn_comments['target'])

plt.show()
asshole_comments = comment_only[comment_only['comment_text'].str.contains('asshole')]

print("Mean toxicity of word 'asshole': {}".format(np.round(asshole_comments['target'].mean(),2)))