# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Configs jupyter to print all results, not just last one
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Any results you write to the current directory are saved as output.
dat = pd.read_csv("../input/Iris.csv")
dat.head()
dat.describe()
dat.groupby('Species').size()
dat.groupby('Species').mean()
sns.pairplot(dat.drop(labels=['Id'], axis=1), hue='Species')
features = dat.iloc[:, 0:5]
labels = dat.iloc[:, 5]
# Create dataframes for training and test data
# The function train_test_split creates a random subset of data for training and testing.
# Takes feature columns and label columns as inputs.
# The test_size is a percentage of how much data to hold out for testing
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.30, random_state=0)
print("Dataframe Shapes---------------------")
print("Training features, labels:", X_train.shape, y_train.shape)
print("Test features, labels:", X_test.shape, y_test.shape)
X_train.head()
y_train.head()
