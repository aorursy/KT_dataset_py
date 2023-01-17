# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))





train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")



train_df.head()

# Any results you write to the current directory are saved as output.
test_df.head()
sns.distplot(train_df['y'].dropna())
train_df.dropna(how="any",inplace=True)

test_df.dropna(how="any",inplace=True)
from sklearn.cross_validation import train_test_split 



X_train,X_test,y_train,y_test = train_test_split(train_df[['x']],train_df['y'],test_size=0.3)



from sklearn.linear_model import LinearRegression



lmodel = LinearRegression()

lmodel.fit(X_train,y_train)
y_predicts = lmodel.predict(X_test)

lmodel.coef_
plt.scatter(y_test,y_predicts)
from sklearn import metrics



print('MAE:', metrics.mean_absolute_error(y_test, y_predicts))

print('MSE:', metrics.mean_squared_error(y_test, y_predicts))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_predicts)))