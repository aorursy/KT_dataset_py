# This Python 3 environment comes with many helpful analytics libraries installed





import numpy as np # linear algebra

import pandas as pd # data processing



# data visualization

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/titanic_train.csv')

test = pd.read_csv('/kaggle/input/titanic_test.csv')



train.head()
___
___
___
___
___
___
___
train.info()
___
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test =  ___
from sklearn.linear_model import LogisticRegression



___
from sklearn.metrics import classification_report

from sklearn import metrics



print(classification_report(y_test,y_predict))

print("Accuracy:", metrics.accuracy_score(y_test, y_predict))