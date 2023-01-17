import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

import os
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv('../input/fish-market/Fish.csv')

df.head()
cols = df.columns.tolist()

cols
cols = ['Species', 'Width', 'Length1', 'Length2', 'Length3', 'Height', 'Weight']
df = df[cols]

print(df['Species'].unique())

df.dtypes
odf = df.select_dtypes(include=['object']).copy()



odf = pd.get_dummies(df, columns=["Species"]).astype('float64')

odf = odf[ ['Width',

 'Length1',

 'Length2',

 'Length3',

 'Height',

 'Species_Whitefish',

 'Species_Bream',

 'Species_Parkki',

 'Species_Perch',

 'Species_Pike',

 'Species_Roach',

 'Species_Smelt'

 ]]





ax = sns.boxplot(x="Species", y="Weight", data=df, orient='v')
g = sns.pairplot(df, kind='scatter', hue = 'Species');
sns.heatmap(df.corr().round(3), annot=True, cmap='Greys');
X = odf.values

y = df['Weight'].values
X = X[:, :11]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

l_score = regressor.score(X_test,y_test)

print("ACCURACY WITH LINEAR REGRESSION --->",l_score*100)
import statsmodels.api as sm

X = np.append(arr = np.ones((159,1)).astype(int) , values = X,axis =1)

X_opt = X[:,[0,1,2,3,4,5,6,7,8,9,10,11]]





rego = sm.OLS(endog = y ,exog = X_opt).fit()

rego.summary()
X_opt = X[:,[0,2,3,4,5,6,7,8,9,10,11]]

rego = sm.OLS(endog = y,exog = X_opt).fit()

rego.summary()
X_opt = X[:,[0,2,3,4,6,7,8,9,10,11]]

rego = sm.OLS(endog = y,exog = X_opt).fit()

rego.summary()
X_opt = X[:,[0,2,3,6,7,8,9,10,11]]

rego = sm.OLS(endog = y,exog = X_opt).fit()

rego.summary()
X_trainBE, X_testBE, y_trainBE, y_testBE = train_test_split(X_opt, y, test_size = 0.3, random_state = 0)



from sklearn.linear_model import LinearRegression

regressor_backward = LinearRegression()

regressor_backward.fit(X_trainBE, y_trainBE)



# Predicting the Test set results

y_predBE = regressor_backward.predict(X_testBE)

mscore = regressor_backward.score(X_testBE,y_testBE)

print("ACCURACY WITH MULTIPLE LINEAR REGRESSION---->",mscore)