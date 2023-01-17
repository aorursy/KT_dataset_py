import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.linear_model import LinearRegression

from sklearn.cross_validation import train_test_split

from sklearn import metrics
unemp = pd.read_csv('../input/new-york-city-census-data/nyc_census_tracts.csv')

unemp.head()
unemp.columns
pd.isnull(unemp).sum()
len(unemp)
unemp = unemp.dropna()
len(unemp)
temp = unemp.corr()

plt.subplots(figsize=(20,10))

sns.heatmap(temp, cmap='RdYlGn', annot=True)

plt.show()
unemp.columns
X = unemp[['TotalPop', 'Men', 'Women', 'White', 'Black', 'Asian', 'Citizen','Poverty','ChildPoverty',

           'Professional','Carpool','Transit',

          'WorkAtHome',

          ]]

y = unemp['Unemployment']
Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,random_state = 1, test_size=0.19)
lin = LinearRegression()
lin.fit(Xtrain, ytrain)
y_pred = lin.predict(Xtest)
np.sqrt(metrics.mean_squared_error(ytest,y_pred))
df = pd.DataFrame({})

df['Unemployment'] = ytest

df['Predicted Unemp'] = y_pred

df['ERROR'] = df['Unemployment'] - df['Predicted Unemp']

df.head(15)
df['ERROR'].describe()