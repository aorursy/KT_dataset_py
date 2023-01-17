# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/creditcardfraud/creditcard.csv')

data.head()
data.info()
#specify x and y

y=data.Class.values

x_data=data.drop(['Class'],axis=1)

#split data into train and test set

x_train, x_test, y_train, y_test=train_test_split(x_data, y, test_size=0.25, random_state=42)
from sklearn.linear_model import LogisticRegression

lr= LogisticRegression(solver='liblinear')

lr.fit(x_train,y_train)

print("train accuracy {}",format(lr.score(x_train,y_train))) 

print("test accuracy {}",format(lr.score(x_test,y_test)))