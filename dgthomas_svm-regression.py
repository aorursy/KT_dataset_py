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
import matplotlib.pyplot as plt

import seaborn as sns
df=pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
df.head(20)
k=5 #variables within heatmap

corrmat = df.corr()

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df[cols].values.T)

sns.set(font_scale=1.5)

heat_map=sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
from sklearn.svm import LinearSVR

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
X=df['GrLivArea']

X=X.values.reshape(-1,1)

X

y=df['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

svm_reg = LinearSVR(epsilon=1.5)

svm_reg.fit(X_train, y_train)
svm_reg = LinearSVR(epsilon=1.5,max_iter=20000)

svm_reg.fit(X_train, y_train)
y_pred = svm_reg.predict(X_test)

mean_squared_error(y_test, y_pred)

from math import sqrt

sqrt(mean_squared_error(y_test, y_pred))
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

test_X = test['GrLivArea']

test_X=test_X.values.reshape(-1,1)

predicted_price=svm_reg.predict(test_X)

my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_price})

my_submission.to_csv('submission.csv', index=False)
