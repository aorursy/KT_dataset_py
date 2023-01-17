import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore",category=DeprecationWarning)
data = pd.read_csv("../input/auto-mpg.csv")

data.head()
# Making the copy of the dataframe

df = data.copy
data.drop(['car name'],axis=1,inplace=True)

data.head()
data.describe()
data.isnull().sum()
data['horsepower'].unique()
data = data[data.horsepower != '?']
# Checking for null values after dropping the rows

'?' in data
data.shape
data.corr()['mpg'].sort_values()
plt.figure(figsize=(10,10))

sns.heatmap(data.corr(),annot=True,linewidth=0.5,center=0,cmap='rainbow')

plt.show()
sns.countplot(data.cylinders,data=data,palette = "rainbow")

plt.show()
sns.countplot(data['model year'],palette = "rainbow")

plt.show()
sns.countplot(data.origin,palette = "rainbow")

plt.show()
data['horsepower'] = pd.to_numeric(data['horsepower'])

sns.distplot(data['horsepower'])

plt.show()
sns.distplot(data.displacement,rug=False)

plt.show()
## multivariate analysis

sns.boxplot(y='mpg',x='cylinders',data=data,palette = "rainbow")

plt.show()
sns.boxplot(y='mpg',x='model year',data=data,palette = "rainbow")

plt.show()
plot = sns.lmplot('horsepower','mpg',data=data,hue='origin',palette = "rainbow")

plt.show()
plot = sns.lmplot('acceleration','mpg',data=data,hue='origin',palette = "rainbow")

plot.set(ylim = (0,50))

plt.show()
plot = sns.lmplot('weight','mpg',data=data,hue='origin',palette = "rainbow")

plt.show()
plot = sns.lmplot('displacement','mpg',data=data,hue='origin',palette = "rainbow")

plot.set(ylim = (0,50))

plt.show()
X = data.iloc[:,1:].values

Y = data.iloc[:,0].values
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=0)



regressor = LinearRegression()

regressor.fit(X_train,Y_train)
Y_pred = regressor.predict(X_test)

print(regressor.score(X_test,Y_test))
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=2)

X_poly = poly_reg.fit_transform(X)



X_train,X_test,Y_train,Y_test = train_test_split(X_poly,Y,test_size=0.30)



lin_reg = LinearRegression()

lin_reg  = lin_reg.fit(X_train,Y_train)



print(lin_reg.score(X_test,Y_test))