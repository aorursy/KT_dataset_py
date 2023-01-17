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
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



train_x = train.iloc[:-200, 1:].values

train_y = train.iloc[:-200, 0].values



test_x = train.iloc[-200:, 1:].values

test_y = train.iloc[-200:, 0].values
#random forest

from sklearn import ensemble

rf = ensemble.RandomForestClassifier(n_estimators=100)

rf.fit(train_x, train_y)

rf.score(test_x, test_y)
rf = ensemble.RandomForestClassifier(n_estimators=100)

rf.fit(train_x, train_y)
rf.score(test_x, test_y)