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



X_train = pd.read_csv('../input/train.csv', index_col=0).drop('loan_condition', axis=1)

X_test = pd.read_csv('../input/test.csv', index_col=0)
# columns of numeric(int or float)

numerics = []

for col in X_train.columns:

    if X_train[col].dtype == 'float64' or X_train[col].dtype == 'int64':

        numerics.append(col)
import matplotlib.pyplot as plt



for col in X_train[numerics].columns:

    plt.figure(figsize=(4,4))



    plt.hist(X_train[col], bins=50, density=True, label='X_train', alpha=0.6)

    plt.hist(X_test[col], bins=50, density=True, label='X_test', alpha=0.6)

    plt.title('Numerical features frequency: ' + col)

    plt.xlabel('features: ' + col)

    plt.ylabel('freq')

    plt.legend()

    plt.show()