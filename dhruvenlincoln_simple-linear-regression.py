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

import matplotlib.pyplot as plt

train_set = pd.read_csv('/kaggle/input/random-linear-regression/train.csv')



train_set = train_set.dropna()



test_set = pd.read_csv('/kaggle/input/random-linear-regression/train.csv')



test_set = test_set.dropna()
train_set.info() # All the null values are removed !!!
X_test = np.array(test_set.iloc[:,:-1].values)

y_test = np.array(test_set.iloc[:,:1].values)
X_train = np.array(train_set.iloc[:,:-1].values)

y_train = np.array(train_set.iloc[:,:1].values)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)
acc = regressor.score(X_test, y_test)

print('Accuracy = '+ str(acc))
plt.scatter(X_test, y_test, color = 'cyan', marker = 'o', alpha = 0.5,label = 'Test data')

plt.plot(X_train, regressor.predict(X_train), color = 'red', label='Linear Regression')

plt.title('predicted vs given values')

plt.xlabel('x')

plt.ylabel('y')

plt.legend(loc = 'upper left')

plt.show()