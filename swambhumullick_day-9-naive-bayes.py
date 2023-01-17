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

data = pd.read_csv("/kaggle/input/weather-dataset/weatherHistory.csv")

data
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix
data.drop(['Formatted Date','Precip Type','Daily Summary'],axis=1,inplace=True)

data
X = data.drop('Summary',axis=1)

Y = data['Summary']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
model = GaussianNB()

model.fit(x_train,y_train)
y_pred = model.predict(x_test)

y_pred
Accuracy = accuracy_score(y_test,y_pred)*100

print(" accuracy is :  ",Accuracy)
