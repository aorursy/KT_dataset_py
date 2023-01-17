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

import seaborn as sns

import matplotlib.pylab as plt

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



from sklearn.datasets import load_iris
iris_data = load_iris()

data = iris_data.data

target = iris_data.target
data = np.column_stack([data, target])
iris = pd.DataFrame(data, columns=['1', '2', '3', '4', 'lables'])

iris['lables'] = iris['lables'].astype('int')

iris.head()
iris.info()
iris.describe()
iris.corr()

feat = ['1', '2', '3', '4']
# iris = iris.loc[:, ['1', '3', '4', 'lables']]

# iris.head()
X_train, X_test, y_train, y_test =  train_test_split(iris[feat], iris['lables'], test_size=0.1, random_state=55)
# 标准化

stdsc = StandardScaler()

X_train = stdsc.fit_transform(X_train)

X_test = stdsc.fit_transform(X_test)

# 转化为df

X_train = pd.DataFrame(X_train, columns=[feat])

X_test = pd.DataFrame(X_test, columns=[feat])
clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

com = confusion_matrix(y_test, y_pred)

com
y_test = np.array(y_test)

y_test
count = len(y_test)

error = 0

for i in range(count):

    if y_pred[i] != y_test[i]:

        error += 1

print('right rate: %s' % ((count-error)/count))