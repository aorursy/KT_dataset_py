# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/creditcard.csv')

df.info()
y = df.pop('Class')
df.Amount.plot(kind='hist', bins=200)
pd.Series(np.log1p(df.Amount)).plot(kind='hist', bins=200)
df = df.assign(Amount_Log = np.log1p(df.Amount))
df[y==1].Time.plot(kind='hist', bins=100)
df[y==0].Time.plot(kind='hist', bins=100)
data_values = df.values
from sklearn.cross_validation import StratifiedShuffleSplit

from sklearn.ensemble import RandomForestClassifier

sss = StratifiedShuffleSplit(y=y, test_size=0.33, n_iter=5)
score_list = []

for train_index, test_index in sss:

    Xtrain, Xtest = data_values[train_index], data_values[test_index]

    Ytrain, Ytest = y[train_index], y[test_index]

    clf = RandomForestClassifier(n_estimators=200, verbose=1, n_jobs=3)

    clf.fit(Xtrain, Ytrain)

    

    score_list.append(clf.score(Xtest, Ytest))
score_list