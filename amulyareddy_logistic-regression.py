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
iris_data = pd.read_csv('../input/iris-dataset-logistic-regression/iris.csv')

X = iris_data[['x0','x1','x2','x3','x4']]

y = iris_data[['type']]
print(iris_data.info())
import warnings

warnings.filterwarnings('ignore')



for i in range(len(y)):

    if y.iloc[i].item()=='Iris-setosa':

        y.iloc[i]=1

    else:

        y.iloc[i]=2
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state = 0)

model.fit(X_train, y_train)
pred = model.predict(X_test)
pred
import matplotlib.pyplot as plt

%matplotlib inline



print(len(X_test),len(pred))
plt.plot(range(len(X_test)),y_test,'o',c='b')
plt.plot(range(len(X_test)),pred,'o',c='r')