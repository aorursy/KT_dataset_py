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



print ( " surivjh")
import os

import pandas as pd

import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.externals import joblib

from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold

train = pd.read_csv(os.path.join('../input/', 'train.csv'))

test = pd.read_csv(os.path.join('../input/', 'test.csv'))

del train['PassengerId']

del train['Name']

del train['Ticket']

for i in [train, test]:

    i['Sex1'] = i['Sex'].apply(lambda x: 1 if x=='male' else 0)

del train['Sex']    

#print (train['Sex'])

for i in [train, test]:

    i['Cabin1'] = i['Cabin'].apply(lambda x: 1 if x!='NaN' else 0)

print (train['Cabin1'])

    