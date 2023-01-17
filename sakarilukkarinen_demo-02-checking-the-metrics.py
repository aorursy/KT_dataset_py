# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Read the solution list

df = pd.read_csv('../input/retinopathy_solution.csv')

df.head()
# Import the metrics and random integer generator

from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report

from numpy.random import randint, randn
# Read the true labels

y_true = df['level'].values



# Generate randomness

y_random = np.round(y_true + randn(len(y_true)))

# Limit the random label values between 0 and 4

y_random = np.maximum(y_random, 0)

y_random = np.minimum(y_random, 4)



# Calculate and print confusion matrix

cm = confusion_matrix(y_true, y_random)

print('Confusion matrix:')

print(cm)

print('')



# Calculate and print Cohen's kappa score

k = cohen_kappa_score(y_true, y_random, weights = 'quadratic')

print("Quadratic weighted Cohen's kappa score = {:.4f}".format(k))

print('')



# Calculate and print classification report

cp = classification_report(y_true, y_random)

print('Classification report:')

print(cp)