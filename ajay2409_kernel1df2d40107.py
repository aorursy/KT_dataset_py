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

dataset = pd.read_csv('/kaggle/input/iris/Iris.csv')

dataset
from sklearn.preprocessing import LabelEncoder

label_encoder= LabelEncoder()

object_cols = [cols for cols in dataset.columns

              if dataset[cols].dtype=='object']

for i in object_cols:

    dataset[i] = label_encoder.fit_transform(dataset[i])

    

X_col= dataset.columns[:5]

y_cols =dataset.columns[5:]

X= dataset[X_col]

Y = dataset[y_cols]

X
from sklearn.model_selection import train_test_split



train_x,valid_x,train_y,valid_y = train_test_split(X,Y,train_size=0.80,random_state=0)
# using multiple linear Regression

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error

a = LinearRegression()

a.fit(train_x,train_y)

predictions =a.predict(valid_x)

mae = mean_absolute_error(valid_y,predictions)

print(mae)

# using decsionTreee

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error

b = DecisionTreeRegressor(random_state=0)

b.fit(train_x,train_y)

pred = b.predict(valid_x)

mae = mean_absolute_error(valid_y,pred)

print(mae)

# using RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error

b = RandomForestClassifier(n_estimators =50,random_state=0)

b.fit(train_x,train_y)

pred = b.predict(valid_x)

mae = mean_absolute_error(valid_y,pred)

print(mae)

# using SVM

from sklearn.svm import SVC

from sklearn.metrics import mean_absolute_error

b = SVC(random_state=0)

b.fit(train_x,train_y)

pred = b.predict(valid_x)

mae = mean_absolute_error(valid_y,pred)

print(mae)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

plt.figure(figsize=(18,5))

sns.lineplot(data=dataset)