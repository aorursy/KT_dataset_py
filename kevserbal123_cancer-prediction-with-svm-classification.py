# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Breast_cancer_data.csv')
data.info()
data.head()
y = data.diagnosis.values

x_data = data.drop(['diagnosis'], axis = 1)
#Normalization of data

x = ( x_data - np.min(x_data) ) / (np.max(x_data) - np.min(x_data))

x.head()
from  sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 1)

from sklearn.svm import SVC

svm = SVC (random_state = 1, gamma='auto')

svm.fit(x_train, y_train)
print("score of svm :", svm.score(x_test, y_test))