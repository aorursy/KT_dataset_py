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
df = pd.read_csv(r"/kaggle/input/winequality-red.csv")
X = df.iloc[:,:-1].values

y = df.iloc[:,11].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.33,random_state = 42)
from sklearn.decomposition import PCA

pca = PCA(n_components = 10)

X_train = pca.fit_transform(X_train)

X_test = pca.transform(X_test)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(X_train,y_train)

yh = model.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,yh)
model.score(X_test,y_test)