import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

% matplotlib inline
trainData = pd.read_csv('../input/train.csv')

testData = pd.read_csv('../input/test.csv')
plt.ylabel('Combination of Hair, Soul and Bone')

sns.boxplot(x=trainData.type, y=trainData.has_soul*trainData.hair_length*trainData.bone_length)
plt.close()

plt.ylabel('Combination of Hair, Soul and Flesh')

sns.boxplot(x=trainData.type, y=trainData.has_soul*trainData.hair_length*trainData.rotting_flesh)