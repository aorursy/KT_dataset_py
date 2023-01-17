# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn import linear_model

from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/forest-fires-in-brazil/amazon.csv', encoding = "ISO-8859-1")

df.head(3)
plt.rcParams['figure.figsize'] = [14, 10]

dataset = df.groupby('year', as_index=False)['number'].sum()

dataset
X = list(dataset.index.values)

y = dataset['number']
plt.plot(dataset['year'], dataset['number'])
dataset.shape
X_train, X_test, y_train, y_test = train_test_split(dataset, y, test_size=0.33)

reg = linear_model.Ridge(alpha=25)

reg.fit(X_train, y_train) 

prediction = reg.predict(X_train)
new_prediction = reg.predict(X_test)

X_test
plt.xlabel('Year')

plt.ylabel('Forest burned')

plt.plot(X_test['year'], new_prediction)
