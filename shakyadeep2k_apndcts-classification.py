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
import pandas as pd

apndcts = pd.read_csv("../input/apndcts/apndcts.csv")

apndcts
apndcts.info()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier



y = apndcts.pop('class')

x = apndcts

y



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,shuffle=True)

model1 = LogisticRegression()

model1.fit(x_train,y_train)

acu1 = model1.score(x_test,y_test)

print(acu1)



model2 = SVC()

model2.fit(x_train,y_train)

acu2 = model2.score(x_test,y_test)

print(acu2)



model3 = DecisionTreeClassifier()

model3.fit(x_train,y_train)

acu3 = model3.score(x_test,y_test)

print(acu3)



model4 = MLPClassifier()

model4.fit(x_train,y_train)

acu4 = model4.score(x_test,y_test)

print(acu4)

import matplotlib.pyplot as plt

import numpy as np



x,y = [1,2,4,3,5],[1,3,3,2,5]
plt.scatter(x,y)

plt.show()
x_mean = np.mean(x)

x_mean