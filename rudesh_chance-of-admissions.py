# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
ad = pd.read_csv('../input/Admission_Predict.csv')
ad.head()
sns.heatmap(ad.isnull())
ad.drop('Serial No.', axis=1, inplace=True)
ad.head()

ad.columns
X = ad.drop('Chance of Admit ', axis=1)
y = ad['Chance of Admit ']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)
lm.coef_
predictions = lm.predict(X_test)
plt.scatter(y_test, predictions)