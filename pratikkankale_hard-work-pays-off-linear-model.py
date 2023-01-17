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
X_train=pd.read_excel('/kaggle/input/hard-work-pays-off-for-beginers/X_train.xlsx', sheet_name='in')

Y_train=pd.read_excel('/kaggle/input/hard-work-pays-off-for-beginers/Y_train.xlsx', sheet_name='in')
import matplotlib.pyplot as plt

import seaborn as sns
#sns.scatterplot(X_train,Y_train)

plt.scatter(X_train, Y_train)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.33, random_state=42)
from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(X_train, y_train)
yhat=model.predict(X_test)
from sklearn.metrics import r2_score, mean_squared_error
print(r2_score(y_test, yhat))

print(mean_squared_error(y_test, yhat))
ax = sns.distplot(y_test, hist=False, color='y', label='original')

sns.distplot(yhat, hist=False, color='r', ax=ax, label='predicted')