# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.neural_network import MLPClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/glass/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/glass/glass.csv', encoding ='latin1')

data.info()
data.head(20000)
y = data['Type'].values

y = y.reshape(-1,1)

x_data = data.drop(['Type'],axis = 1)

print(x_data)
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values

x.head(20000)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.5,random_state=100)



y_train = y_train.reshape(-1,1)

y_test = y_test.reshape(-1,1)



print("x_train: ",x_train.shape)

print("x_test: ",x_test.shape)

print("y_train: ",y_train.shape)

print("y_test: ",y_test.shape)



my_x = x_train

my_y = y_train



my_xx = x_test

my_yy = y_test
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(7, 2), random_state=1)
clf.fit(my_x, my_y)

y_pred = clf.predict(my_xx)

print("Accuracy:",metrics.accuracy_score(my_yy, y_pred))