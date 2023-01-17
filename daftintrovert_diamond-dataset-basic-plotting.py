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
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
file = '../input/diamonds.csv'

df = pd.read_csv(file)
df.head()
diadata = df.drop(['Unnamed: 0'],axis = 1)
diadata.info()
diadata.describe()
diadata.corr()
plt.figure(figsize = (7,7))

sns.heatmap(diadata.corr(),annot = True,square = True)
sns.pairplot(diadata,height = 2)
sns.pairplot(diadata,hue = 'color',height = 2)
sns.pairplot(diadata,hue = 'cut',height = 2)
diadata.hist(figsize = (20,20),bins=150)
sns.catplot(x='cut', data=diadata , kind='count',aspect=2.5 )
sns.jointplot(x= 'carat',y = 'price',data = diadata,kind = 'kde')
sns.catplot(x='cut', y='price', data=diadata, kind='box', aspect = 1.5)
diadata.hist(figsize = (20,20), by=diadata.cut,grid=True)
sns.catplot(x='color', data=diadata , kind='count',aspect=2.5 )
sns.catplot(x='clarity', y='price', data=diadata, kind='box' ,aspect=2.5)
one_hot_encoders_diadata =  pd.get_dummies(diadata)

one_hot_encoders_diadata.head()
# a structured approach

cols = one_hot_encoders_diadata.columns

diamond_clean_data = pd.DataFrame(one_hot_encoders_diadata,columns= cols)

diamond_clean_data.head()
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

numericals =  pd.DataFrame(sc_X.fit_transform(diamond_clean_data[['carat','depth','x','y','z','table']]),columns=['carat','depth','x','y','z','table'],index=diamond_clean_data.index)
numericals.head()
diamond_clean_data_standard = diamond_clean_data.copy(deep=True)

diamond_clean_data_standard[['carat','depth','x','y','z','table']] = numericals[['carat','depth','x','y','z','table']]
diamond_clean_data_standard.head()
plt.figure(figsize=(20,20))  # on this line I just set the size of figure to 12 by 10.

p=sns.heatmap(diamond_clean_data.corr(), annot=True,cmap='RdYlGn')  # seaborn has very simple solution for heatmap
x = diamond_clean_data_standard.drop(["price"],axis=1)

y = diamond_clean_data_standard.price
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(x, y,random_state = 2,test_size=0.3)
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
las_reg = linear_model.Lasso()

las_reg.fit(train_x,train_y)

y_pred = las_reg.predict(test_x)

print("accuracy: "+ str(las_reg.score(test_x,test_y)*100) + "%")

print("Mean absolute error: {}".format(mean_absolute_error(test_y,y_pred)))

print("Mean squared error: {}".format(mean_squared_error(test_y,y_pred)))

R2 = r2_score(test_y,y_pred)

print('R Squared: {}'.format(R2))

n=test_x.shape[0]

p=test_x.shape[1] - 1



adj_rsquared = 1 - (1 - R2) * ((n - 1)/(n-p-1))

print('Adjusted R Squared: {}'.format(adj_rsquared))
rig_reg = linear_model.Ridge()

rig_reg.fit(train_x,train_y)

y_pred = rig_reg.predict(test_x)

print("accuracy: "+ str(rig_reg.score(test_x,test_y)*100) + "%")

print("Mean absolute error: {}".format(mean_absolute_error(test_y,y_pred)))

print("Mean squared error: {}".format(mean_squared_error(test_y,y_pred)))

R2 = r2_score(test_y,y_pred)

print('R Squared: {}'.format(R2))

n=test_x.shape[0]

p=test_x.shape[1] - 1



adj_rsquared = 1 - (1 - R2) * ((n - 1)/(n-p-1))

print('Adjusted R Squared: {}'.format(adj_rsquared))
l = list(range(0,len(diamond_clean_data_standard.columns)))
import statsmodels.formula.api as smf

X = np.append(arr = np.ones((diamond_clean_data_standard.shape[0], 1)).astype(int), values = diamond_clean_data_standard.drop(['price'],axis=1).values, axis = 1)

X_opt = X[:, l]

regressor_ols = smf.OLS(endog = diamond_clean_data_standard.price, exog = X_opt).fit()

regressor_ols.summary()

l.pop(5)

X = np.append(arr = np.ones((diamond_clean_data_standard.shape[0], 1)).astype(int), values = diamond_clean_data_standard.drop(['price'],axis=1).values, axis = 1)

X_opt = X[:, l]

regressor_ols = smf.OLS(endog = diamond_clean_data_standard.price, exog = X_opt).fit()

regressor_ols.summary()
l.pop(5)

X = np.append(arr = np.ones((diamond_clean_data_standard.shape[0], 1)).astype(int), values = diamond_clean_data_standard.drop(['price'],axis=1).values, axis = 1)

X_opt = X[:, l]

regressor_ols = smf.OLS(endog = diamond_clean_data_standard.price, exog = X_opt).fit()

regressor_ols.summary()