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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(context="notebook", palette="gist_rainbow", style = 'darkgrid' ,font_scale = 1.5, color_codes=True)

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

#plt.style.use('ggplot')

#ggplot is R based visualisation package that provides better graphics with higher level of abstraction

import os
dd = pd.read_csv("../input/diamonds.csv")
dd.info()
dd.head()
dd = dd.drop(["Unnamed: 0"],axis=1)

dd.head()
plt.figure(figsize=(20,20))  # on this line I just set the size of figure to 12 by 10.

p=sns.heatmap(dd.corr(), annot=True,cmap='RdYlGn',square=True)  # seaborn has very simple solution for heatmap
d1 = dd.plot(x='carat',y='price',kind='scatter')
dd.describe()
# The minimum values for x,y and z here are 0 but it is not possible because according to the data description they are the length, width and depth
print("Number of rows with x == 0: {} ".format((dd.x==0).sum()))

print("Number of rows with y == 0: {} ".format((dd.y==0).sum()))

print("Number of rows with z == 0: {} ".format((dd.z==0).sum()))

print("Number of rows with depth == 0: {} ".format((dd.depth==0).sum()))
dd[['x','y','z']] = dd[['x','y','z']].replace(0,np.NaN)
dd.isnull().sum()
dd.dropna(inplace=True)
dd.shape
diamond_data.isnull().sum()
d1 = dd.hist(figsize = (20,20),bins=150)
d1 = sns.factorplot(x='cut', data=dd, kind='count',aspect=2.5 )
d1 = sns.factorplot(x='cut', y='price', data=dd, kind='box' ,aspect=2.5 )
d1 = dd.hist(figsize = (20,20), by=dd.cut,grid=True)
d1 = sns.factorplot(x='color', data=dd , kind='count',aspect=2.5 )
d1 = sns.factorplot(x='color', y='price', data=dd, kind='box' ,aspect=2.5 )
d1 = sns.factorplot(x='clarity', data=diamond_data , kind='count',aspect=2.5 )
d1 = sns.factorplot(x='clarity', y='price', data=diamond_data, kind='box' ,aspect=2.5)
one_hot_encoders_dd =  pd.get_dummies(dd)

one_hot_encoders_dd.head()
# a structured approach

cols = one_hot_encoders_dd.columns

dd_clean_data = pd.DataFrame(one_hot_encoders_dd,columns= cols)

dd_clean_data.head()
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

numericals =  pd.DataFrame(sc_X.fit_transform(dd_clean_data[['carat','depth','x','y','z','table']]),columns=['carat','depth','x','y','z','table'],index=dd_clean_data.index)
numericals.head()
dd_clean_data_standard = dd_clean_data.copy(deep=True)

dd_clean_data_standard[['carat','depth','x','y','z','table']] = numericals[['carat','depth','x','y','z','table']]
dd_clean_data_standard.head()
plt.figure(figsize=(20,20))  # on this line I just set the size of figure to 12 by 10.

p=sns.heatmap(dd_clean_data.corr(), annot=True,cmap='RdYlGn')  # seaborn has very simple solution for heatmap
plt.figure(figsize=(20,20))  # on this line I just set the size of figure to 12 by 10.

p=sns.heatmap(dd_clean_data_standard.corr(), annot=True,cmap='RdYlGn')  # seaborn has very simple solution for heatmap
x = dd_clean_data_standard.drop(["price"],axis=1)

y = dd_clean_data_standard.price
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(x, y,random_state = 2,test_size=0.3)
import matplotlib.pyplot as plt

import numpy as np

from sklearn import datasets, linear_model

from sklearn.metrics import mean_squared_error, r2_score





regr = linear_model.LinearRegression(normalize=True)

regr.fit(train_x,train_y)

y_pred = regr.predict(test_x)
dd_clean_data_standard.describe()
dd_clean_data_standard.info()
from sklearn.metrics import mean_absolute_error

print("accuracy: "+ str(regr.score(test_x,test_y)*100) + "%")

print("Mean absolute error: {}".format(mean_absolute_error(test_y,y_pred)))

print("Mean squared error: {}".format(mean_squared_error(test_y,y_pred)))

print('Coefficients: \n', regr.coef_)
las_reg = linear_model.Lasso()

las_reg.fit(train_x,train_y)

y_pred = las_reg.predict(test_x)

print("accuracy: "+ str(las_reg.score(test_x,test_y)*100) + "%")

print("Mean absolute error: {}".format(mean_absolute_error(test_y,y_pred)))

print("Mean squared error: {}".format(mean_squared_error(test_y,y_pred)))

print('Coefficients: \n', las_reg.coef_)
rig_reg = linear_model.Ridge()

rig_reg.fit(train_x,train_y)

y_pred = rig_reg.predict(test_x)

print("accuracy: "+ str(rig_reg.score(test_x,test_y)*100) + "%")

print("Mean absolute error: {}".format(mean_absolute_error(test_y,y_pred)))

print("Mean squared error: {}".format(mean_squared_error(test_y,y_pred)))

print('Coefficients: \n', rig_reg.coef_)