import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O



# Visualisation

import matplotlib.pyplot as plt

import seaborn as sns
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
#Preview the training dats

train_df.head()
train_df.info()
#Return summary of training data set

train_df.describe()
#Return summary of test data set

test_df.describe()
g = sns.FacetGrid(train_df, row="Sex", col="Survived", margin_titles=True)

bins = np.linspace(0, 60, 13)

g.map(plt.hist, "Age", color="steelblue", bins=bins, lw=0)