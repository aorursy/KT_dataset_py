# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import seaborn as sns 
import matplotlib.pyplot as plt
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
X = train.get_values()
y = X[:, 0]
X = X[:, 1:]
sns.distplot(y, bins=10, kde=False)
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
clf = MLPClassifier(solver = 'lbfgs', alpha = 1e-5, hidden_layer_sizes = (100))
clf.fit(X, y)
X_test = test.get_values()
X_test = scaler.transform(X_test)
y_test = clf.predict(X_test)
output = pd.DataFrame(y_test, columns = ['Label'])
output.reset_index(inplace = True)
output.rename(columns = {'index' : 'ImageId'}, inplace = True)
output['ImageId'] = output['ImageId'] + 1
output.to_csv('output.csv', index = False)
