# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

print(train.head())
labels = train['label']

train = train.drop('label', axis=1)

print(train.head())
clf = RandomForestClassifier(n_estimators=25)

clf = clf.fit(train, labels)



results = clf.predict(test)



np.savetxt('results_rf.csv',

           np.c_[range(1, len(test) + 1), results],

           delimiter=',',

           header='ImageId,Label',

           comments='',

           fmt='%d')
clf = KNeighborsClassifier(n_neighbors=15)

clf = clf.fit(train, labels)



results = clf.predict(test)



np.savetxt('results_knn.csv',

           np.c_[range(1, len(test) + 1), results],

           delimiter=',',

           header='ImageId,Label',

           comments='',

           fmt='%d')