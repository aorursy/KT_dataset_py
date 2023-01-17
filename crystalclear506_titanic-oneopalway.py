# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
training_set = pd.read_csv("../input/train.csv", delimiter=',')

# Eliminate Null Data
training_set = training_set.loc[training_set['Age'] == training_set['Age']]

# Numerizing Feature "Gender
training_set.loc[training_set['Sex'] == 'female', 'Sex'] = 0
training_set.loc[training_set['Sex'] == 'male', 'Sex'] = 1
data_y = np.array(training_set['Survived'])[np.newaxis].T

data_X = np.array(np.array(training_set['Age']))[np.newaxis].T
data_X = np.append(data_X, np.array(training_set['Sex'])[np.newaxis].T, axis=1)
data_X = np.append(data_X, np.array(training_set['Pclass'])[np.newaxis].T, axis=1)
data_X = np.append(data_X, np.array(training_set['Fare'])[np.newaxis].T, axis=1)

%matplotlib inline
import matplotlib.pyplot as plt

# Separating Data for Plotting
survived_indexes = []
not_survived_indexes = []
for i in range(data_y.shape[0]):
    if data_y[i,0] == 1:
        survived_indexes.append(i)
    else:
        not_survived_indexes.append(i)
        

plt.scatter(data_X[survived_indexes, 0], data_X[survived_indexes, 3], color="blue")
plt.scatter(data_X[not_survived_indexes, 0], data_X[not_survived_indexes, 3], color="red")
