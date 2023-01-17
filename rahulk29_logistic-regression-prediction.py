# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# for Box-Cox Transformation
from scipy import stats
candy = pd.read_csv("../input/candy-data.csv")
candy.head(10)
candy
candy.describe()
# for min_max scaling
from mlxtend.preprocessing import minmax_scaling
original_data = candy["winpercent"]

# mix-max scale the data between 0 and 1
scaled_data = minmax_scaling(original_data, columns = [0])

# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(original_data, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Scaled data")
scaled_data
X = candy[["bar","fruity","winpercent"]]
X["winpercent"] = scaled_data
X.head()
Y = candy[["chocolate"]]
Y.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.33)
y_test
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
y_predict
y_test
from sklearn.metrics import accuracy_score
y_test
accuracy_score = accuracy_score(y_predict,y_test)
accuracy_score*100
import matplotlib.pyplot as plt
plt.scatter(X_train["bar"].values,y_train.values)
plt.title("Chocolate or Candy (1 or 0)")
plt.xlabel("bar")
plt.ylabel("Chocolate/Candy")