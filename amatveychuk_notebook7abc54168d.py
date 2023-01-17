# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import *



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



data = pd.read_csv('../input/train.csv', index_col='PassengerId')

data.head()

Y = data['Survived']

X = data.filter(items=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin'])

X['Sex'] = X['Sex'].transform(lambda s: 0 if s == 'male' else 1)

X['Cabin'] = X['Cabin'].transform(lambda s: 0 if s == 'NaN' else 1)

X.head()

# Any results you write to the current directory are saved as output.