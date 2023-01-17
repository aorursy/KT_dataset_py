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
import pandas as pd

from sklearn.linear_model import LogisticRegression





train = pd.read_csv("../input/train.csv").drop('id', axis=1)



y_train = train['target']

X_train = train.drop('target', axis=1)



test = pd.read_csv('../input/test.csv')

X_test = test.drop('id', axis = 1)
from scipy.stats import mstats



def winsor_func(s):

    return mstats.winsorize(s, limits=[0.00, 0.01])

def using_mstats(df):

    return df.apply(winsor_func, axis=0)



X_train_winsor=X_train.apply(winsor_func)

X_test_winsor=X_test.apply(winsor_func)
clf = LogisticRegression(class_weight='balanced', penalty='l1', C=0.1, solver='liblinear').fit(X_train_winsor, y_train)
y_out=clf.predict_proba(X_test)[:,1]

submission = pd.read_csv('../input/sample_submission.csv')

submission['target'] = y_out

submission.to_csv('submission_winsor.csv', index=False)