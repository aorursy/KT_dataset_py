# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error 

#importing data

test_data= pd.read_csv("../input/random-linear-regression/test.csv")



train_data= pd.read_csv("../input/random-linear-regression/train.csv")

train_data
train_data.dropna(inplace=True)

test_data.dropna(inplace=True)
X= train_data.iloc[:,:-1].values

Y= train_data.iloc[:,1].values

X_test = np.array(test_data.iloc[:, :-1].values)

y_test = np.array(test_data.iloc[:, 1].values)

model= LinearRegression()

model.fit(X,Y)
y_predict= model.predict(X_test)



plt.scatter(X,Y)

plt.plot(X,model.predict(X),color='blue')
plt.scatter(X_test,y_test)

plt.plot(X,model.predict(X),color="red")