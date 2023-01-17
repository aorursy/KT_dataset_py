# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/housetrain.csv')

#df = df.sort_values(by=['SalePrice'])
#df
# convert catecorical data to numeric data
print(len(df.columns))
df = pd.get_dummies(df)
print(len(df.columns))
#fill NaN
df = df.fillna(df.min())
#normalize
price_max =  df['SalePrice'].max()
price_min =  df['SalePrice'].min()
for column in df.columns:
    if column not in ['SalePrice', 'Id']:
        df[column] = df[column] / df[column].max()

#separate feature and label
train_set = df
train_x = train_set.drop(['SalePrice', 'Id'],1)
train_y = train_set['SalePrice']
print(df.columns)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
pca = PCA(.99)

x = train_x.values
print(len(x[0]))
x = StandardScaler().fit_transform(x)
principalComponents = pca.fit_transform(x)
train_x = principalComponents
print(len(train_x[0]))
from sklearn import datasets, linear_model
regr = linear_model.LinearRegression()
regr.fit(train_x, train_y)
error = []
predict_result = regr.predict(train_x)
for i in range(len(predict_result)):
    error.append(abs(predict_result[i] - train_y[i]))
print("mean:",sum(error) / float(len(error)))
xy=[]
test_y=[]
pred_y=[]
for i in range(len(predict_result)):
    xy.append((predict_result[i],train_y[i]))
xy.sort(key=lambda x: x[1])

plt.figure(figsize=(30,10))
for xy_element in xy:
    pred_y.append(xy_element[0])
    test_y.append(xy_element[1])
plt.plot(pred_y, color='red')
plt.plot(test_y, color='blue')
plt.show()
from sklearn.tree import DecisionTreeRegressor
regr = DecisionTreeRegressor(max_depth=10)
regr.fit(train_x, train_y)
error = []
predict_result = regr.predict(train_x)
for i in range(len(predict_result)):
    error.append(abs(predict_result[i] - train_y[i]))
    #error.append(((float(predict_result[i]) - (train_y[i]))**2)**0.5)
print("mean:",sum(error) / float(len(error)))
xy=[]
test_y=[]
pred_y=[]
for i in range(len(predict_result)):
    xy.append((predict_result[i],train_y[i]))
xy.sort(key=lambda x: x[1])

plt.figure(figsize=(30,10))
for xy_element in xy:
    pred_y.append(xy_element[0])
    test_y.append(xy_element[1])
plt.plot(pred_y, color='red')
plt.plot(test_y, color='blue')
plt.show()
from sklearn.neural_network import MLPRegressor
regr = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(50, 5), random_state=5)
regr.fit(train_x, train_y)
error = []
predict_result = regr.predict(train_x)
for i in range(len(predict_result)):
    error.append(abs(predict_result[i] - train_y[i]))
print("mean:",sum(error) / float(len(error)))
xy=[]
test_y=[]
pred_y=[]
for i in range(len(predict_result)):
    xy.append((predict_result[i],train_y[i]))
xy.sort(key=lambda x: x[1])

plt.figure(figsize=(30,10))
for xy_element in xy:
    pred_y.append(xy_element[0])
    test_y.append(xy_element[1])
plt.plot(pred_y, color='red')
plt.plot(test_y, color='blue')
plt.show()