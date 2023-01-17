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
import matplotlib.pyplot as plt

plt.style.use('ggplot')

df = pd.read_csv('../input/Profitability.csv')

df.head()
df.shape
#X = df.drop('Profitable',axis=1).values

#y = df['Profitable'].values



X = np.array(df.ix[:, 2:7]) 	  # end index is exclusive

y = np.array(df['Profitable'])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
#import KNeighborsClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score



clf = KNeighborsClassifier(n_neighbors=3)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)



print("Accuracy score: ")

print(accuracy_score(y_test, y_pred))