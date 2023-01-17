# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/iris/Iris.csv')
df.head()
x = np.array(df.loc[:,'SepalLengthCm']).reshape(-1,1)
y = np.array(df.loc[:,'PetalLengthCm']).reshape(-1,1)

plt.figure(figsize=[8,8])
plt.scatter(x,y)
plt.xlabel('SepalLengthCm')
plt.ylabel('PetalLengthCm')
plt.show()
# Lineer regeressıon

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
predict_space = np.linspace(min(x), max(x)).reshape(-1,1)
lr.fit(x,y)

predict = lr.predict(predict_space)

plt.figure(figsize=[8,8])
plt.plot(predict_space, predict, color='red')
plt.scatter(x,y)
plt.xlabel('SepalLengthCm')
plt.ylabel('PetalLengthCm')
plt.show()
print('K^2 score: ', lr.score(x,y))
# Polynomial Regression

from sklearn.preprocessing import PolynomialFeatures

poly_lr = PolynomialFeatures(degree = 42)
x_poly = poly_lr.fit_transform(x)

lr2 = LinearRegression()
lr2.fit(x_poly,y)
predict2 = lr2.predict(x_poly)

print('K^2 score: ',lr2.score(x_poly,y))

plt.figure(figsize=[8,8])
plt.plot(x,predict2, color='red')
plt.scatter(x,y)
plt.xlabel('SepalLengthCm')
plt.ylabel('PetalLengthCm')
plt.show()
# Decision Tree Regression

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(x,y)

predict_space_tree = np.arange(min(x), max(x), 0.01).reshape(-1,1)
predict_tree = tree_reg.predict(predict_space_tree)

print(tree_reg.score(x,y))

plt.figure(figsize=[8,8])
plt.scatter(x,y)
plt.plot(predict_space_tree, predict_tree, color='red')
plt.xlabel('SepalLengthCm')
plt.ylabel('PetalLengthCm')
plt.show()
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(x,y)

random_forest_predict_space = np.arange(min(x), max(x), 0.01).reshape(-1,1)
forest_predict = forest_reg.predict(random_forest_predict_space)

print(forest_reg.score(x,y))

plt.figure(figsize=[8,8])
plt.scatter(x,y)
plt.plot(random_forest_predict_space, forest_predict, color='red')
plt.xlabel('SepalLengthCm')
plt.ylabel('PetalLengthCm')
plt.show()