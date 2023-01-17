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
import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
df_train = pd.read_csv('../input/train.csv');

df_test = pd.read_csv('../input/test.csv');
from sklearn.preprocessing import StandardScaler
stand = StandardScaler()
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')


X = df_train.values[:,1:]

y = df_train.values[:,0]

Xt = df_test.values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42, stratify=y);
X_train_std = stand.fit_transform(np.float64(X_train));

X_test_std = stand.transform(np.float64(X_train));

Xt_std = stand.fit_transform(np.float64(Xt))
%%time

knn.fit(X_train_std,y_train)
y_pred = knn.predict(Xt_std)
Idimg = np.array(range(1, Xt.shape[0]+1))
Idimg.shape, y_pred.shape
pd.DataFrame({'Imagenid':Idimg,'Label':y_pred})