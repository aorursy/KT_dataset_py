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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv('../input/bodyfat.csv')

data
x = data.iloc[:, :1].values.reshape(-1, 1)

y = data.iloc[:, 1].values.reshape(-1, 1)

sns.scatterplot(x=data['age'], y=data['fat'], data=data)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train, y_train)

print(regressor.score(x_test, y_test))
y_pred = regressor.predict(x_test)

print(regressor.predict([[47]]))
# from sklearn.metrics import accuracy_score

# print("Accuracy ::  %.2f " % (100*accuracy_score(y_test, y_pred)))