# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

actual = [1,1,0,1,0,0,1,0,0,0]

predicted = [1,0,0,1,0,0,1,1,1,0]

results = confusion_matrix(actual,predicted)

print(' confusion matrix : ')

print(results)

print('accuracy score :',accuracy_score(actual,predicted))

print('report :')

print(classification_report(actual,predicted))

from numpy import array

from sklearn.model_selection import KFold

data= array([0.1,0.2,0.3,0.4,0.5,0.6])

Kfold = KFold(3,True,1)

for train,test in Kfold.split(data):

    print(' train : %s, test : %s' % (data[train],data[test]))