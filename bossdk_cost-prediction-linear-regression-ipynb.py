# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LinearRegression 

from sklearn.metrics import r2_score,mean_squared_error

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import PolynomialFeatures

from sklearn.ensemble import RandomForestRegressor

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df= pd.read_csv('/kaggle/input/insurance/insurance.csv')
df.head(10)
df.describe()
df.info()
print('Number of rows and columns in the data set: ',df.shape)
# Check for null count column wise

df.isnull().sum(axis=0)
f= plt.figure(figsize=(12,4))

ax=f.add_subplot(121)

sns.distplot(df['charges'],bins=50,color='r',ax=ax)

ax.set_title('Distribution of insurance charges')



ax=f.add_subplot(122)

sns.distplot(np.log10(df['charges']),bins=40,color='b',ax=ax)

ax.set_title('Distribution of insurance charges in $log$ sacle')

ax.set_xscale('log')

plt.show()

plt.figure(figsize=(18,4))

plt.subplot(131)

sns.barplot(x='sex', y='charges', data=df)

plt.subplot(132)

sns.barplot(x='smoker', y='charges', data=df)

plt.subplot(133)

sns.barplot(x='region', y='charges', data=df)

plt.show()
sns.pairplot(df,kind="reg")


#Plot a heatmap and look at the corelation

sns.heatmap(df.corr(), cmap='coolwarm',annot=True)
# Let us map the variables with 2 levels to 0 and 1

df['sex']=df['sex'].map({'male':1, 'female':0})

df['smoker']=df['smoker'].map({'yes':1,'no':0})
# Assigning dummy variables to remaining categorical variable- region

df = pd.get_dummies(df, columns=['region'], drop_first=True)

df.head()
from sklearn.model_selection import train_test_split

X = df.drop('charges',axis=1) # Independet variable

y = df['charges'] # dependent variable



X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)
lr = LinearRegression()

lr.fit(X_train,y_train)
y_train_pred = lr.predict(X_train)

y_test_pred = lr.predict(X_test)

print(lr.score(X_test,y_test))
X = df.drop(['charges','region_northwest','region_southeast','region_southwest'], axis = 1)

Y = df.charges







quad = PolynomialFeatures (degree = 2)

x_quad = quad.fit_transform(X)



X_train,X_test,Y_train,Y_test = train_test_split(x_quad,Y, random_state = 0)



plr = LinearRegression().fit(X_train,Y_train)



Y_train_pred = plr.predict(X_train)

Y_test_pred = plr.predict(X_test)



print(plr.score(X_test,Y_test))
forest = RandomForestRegressor(n_estimators = 100,

                              criterion = 'mse',

                              random_state = 1,

                              n_jobs = -1)

forest.fit(X_train,y_train)

forest_train_pred = forest.predict(X_train)

forest_test_pred = forest.predict(X_test)



print('MSE train data: %.3f, MSE test data: %.3f' % (

mean_squared_error(y_train,forest_train_pred),

mean_squared_error(y_test,forest_test_pred)))

print('R2 train data: %.3f, R2 test data: %.3f' % (

r2_score(y_train,forest_train_pred),

r2_score(y_test,forest_test_pred)))
plt.figure(figsize=(10,6))



plt.scatter(forest_train_pred,forest_train_pred - y_train,

          c = 'black', marker = 'o', s = 35, alpha = 0.5,

          label = 'Train data')

plt.scatter(forest_test_pred,forest_test_pred - y_test,

          c = 'c', marker = 'o', s = 35, alpha = 0.7,

          label = 'Test data')

plt.xlabel('Predicted values')

plt.ylabel('Tailings')

plt.legend(loc = 'upper left')

plt.hlines(y = 0, xmin = 0, xmax = 60000, lw = 2, color = 'red')

plt.show()