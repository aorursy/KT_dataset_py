



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
data = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
data.info()
data.head()
data.tail()
data.shape
data.corr()
import matplotlib.pyplot as plt

import plotly as py

import seaborn as sns

plt.figure(figsize=(15,15))

sns.heatmap(data.corr(),annot=True,fmt='.1f')

plt.show()
sns.pairplot(data)

plt.show()
y = data.target.values

x = data.drop(["target"],axis =1)
x_normalized = (x-np.min(x))/(np.max(x)-np.min(x))
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_normalized,y,test_size=0.17,random_state=42)
from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier()

dt.fit(x_train,y_train)
dt.score(x_test,y_test)