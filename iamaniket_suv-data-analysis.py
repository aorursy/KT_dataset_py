# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dataset=pd.read_csv("../input/suv_data.csv")

dataset.head()
X=dataset.iloc[:,[2,3]].values

Y=dataset.iloc[:,4].values
X
Y
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25, random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train=sc.fit_transform(X_train)

X_test=sc.fit_transform(X_test)

from sklearn.linear_model import LogisticRegression

classifier=LogisticRegression(random_state=0)

classifier.fit(X_train, Y_train)
Y_prediction=classifier.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(Y_test, Y_prediction)*100