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
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV 

from sklearn.metrics import confusion_matrix, classification_report
train = pd.read_csv('../input/train.csv', index_col = 'id')

test = pd.read_csv('../input/test.csv', index_col = 'id')

labels = train.columns.drop(['id', 'target'])

target = train['target']
from sklearn.linear_model import LogisticRegression



m = LogisticRegression(

    penalty='l1',

    C=0.1

)

m.fit(train[labels], target)

m.predict_proba(test[labels])[:,1]