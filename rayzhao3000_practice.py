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
train_csv = open("../input/train.csv", 'r')
test_csv = open("../input/test.csv", 'r')
N = 42000
d = 784
X_train = np.zeros((N, d))
Y_train = np.zeros((N, 1))
train_csv.readline()
i = 0
for line in train_csv:
    i += 0
    line = line.rstrip().split(',')
    y, x = line[0], line[1:]
    X_train[i] = np.array(x)
    Y_train[i] = y
contrast_normalize = lambda x: x/np.linalg.norm(x, ord=2)
list(map(contrast_normalize, X_train))
