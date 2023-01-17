# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns; sns.set(style="ticks", color_codes=True)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/seed-from-uci/Seed_Data.csv')

data.sample(5)
data.describe()
plt.figure(figsize=[8,8])

sns.heatmap(data.corr(), annot=True, cmap="YlGn")

plt.title('Correlations of the Features')

plt.show()
sns.countplot(data['target'], palette='husl')

plt.show()
i = sns.pairplot(data, vars = ['A', 'P', 'C', 'LK', 'WK', 'A_Coef', 'LKG'] ,hue='target', palette='husl')

plt.show()
a = sns.FacetGrid(data, col='target')

a.map(sns.boxplot, 'A', color='yellow', order=['0', '1', '2'])



p = sns.FacetGrid(data, col='target')

p.map(sns.boxplot, 'P', color='orange', order=['0', '1', '2'])



c = sns.FacetGrid(data, col='target')

c.map(sns.boxplot, 'C', color='red', order=['0', '1', '2'])



lk = sns.FacetGrid(data, col='target')

lk.map(sns.boxplot, 'LK', color='purple', order=['0', '1', '2'])



wk = sns.FacetGrid(data, col='target')

wk.map(sns.boxplot, 'WK', color='blue', order=['0', '1', '2'])



acoef = sns.FacetGrid(data, col='target')

acoef.map(sns.boxplot, 'A_Coef', color='cyan', order=['0', '1', '2'])



lkg = sns.FacetGrid(data, col='target')

lkg.map(sns.boxplot, 'LKG', color='green', order=['0', '1', '2'])
wheat_data_simple = data.iloc[:,0:7]

wheat_data_simple.head(3)
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(wheat_data_simple, data.target,random_state = 2,test_size=0.2)
from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

from sklearn import linear_model



regr = linear_model.LinearRegression()

regr.fit(train_x,train_y)

y_pred = regr.predict(test_x)

print("accuracy: "+ str(regr.score(test_x,test_y)*100) + "%")

print("Mean absolute error: {}".format(mean_absolute_error(test_y,y_pred)))

print("Mean squared error: {}".format(mean_squared_error(test_y,y_pred)))

R2 = r2_score(test_y,y_pred)

print('R Squared: {}'.format(R2))

n=test_x.shape[0]

p=test_x.shape[1] - 1



adj_rsquared = 1 - (1 - R2) * ((n - 1)/(n-p-1))

print('Adjusted R Squared: {}'.format(adj_rsquared))
df = pd.DataFrame({'Actual': test_y.values.flatten(), 'Predicted': y_pred.flatten()})

df
df1 = df.head(25)

df1.plot(kind='bar',figsize=(16,10))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()