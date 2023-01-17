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

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('../input/suv-data/suv_data.csv')

df.head()
df.isnull()

df.isnull().sum()

sns.heatmap(df.isnull(), xticklabels=False,cbar=False)
dfNew = df.drop(['User ID','Gender'], axis=1, inplace=False)

dfNew.head()
x = dfNew.drop(['Purchased'], axis=1, inplace=False)

y = df['Purchased']
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.1,random_state=1)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(x_train,y_train)

predict = model.predict(x_test)

model.score(x_test,y_test)
from sklearn.metrics import classification_report

classification_report(y_test,predict)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,predict)
from sklearn.metrics import accuracy_score

accuracy_score(y_test,predict)