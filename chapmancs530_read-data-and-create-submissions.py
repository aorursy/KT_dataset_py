# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Read in training data

train = pd.read_csv('/kaggle/input/chapman-cs530-redwinequality/train.csv')

train.head()
# Read in testing data

test = pd.read_csv('/kaggle/input/chapman-cs530-redwinequality/test.csv')

test.head()
# Read in sample submissions

sample_submission = pd.read_csv('/kaggle/input/chapman-cs530-redwinequality/sample_submission.csv')

sample_submission.head()
# Create a dummy submission that has entries as many as the test set.

y_pred = np.random.rand(test.shape[0]) * 10 # Create random numbers from 0-10 as dummy solution

sample_submission.loc[:, 'Predicted'] = y_pred # Change the Predicted column to your prediction

sample_submission.head()
sample_submission.to_csv('your_submission.csv', header=True, index=False) # Save the header but not the index