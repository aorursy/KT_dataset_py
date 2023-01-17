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
df = pd.read_csv('../input/train.csv')

data = df.copy()

df.head()
df.columns
home_features = ['OverallQual','OverallCond','YearBuilt','TotalBsmtSF','1stFlrSF', 'GrLivArea', 

'FullBath', 'TotRmsAbvGrd','GarageCars', 'GarageArea','Fireplaces', 'LotArea','YearRemodAdd','SalePrice']
df_req = df[home_features]

df_req.head()
df_req.dtypes
correlations = df_req.corr()['SalePrice']

X = df_req.drop(columns='SalePrice')

y = df['SalePrice']

X.head()
y.hist();
import numpy as np

y_copy = y

y = np.log1p(y)

y.hist();
# let's train our first model....Linear Regression 

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression



X_train,X_valid,y_train,y_valid = train_test_split(X,y,test_size=0.275,random_state=42)

lin_reg = LinearRegression()

lin_reg.fit(X_train,y_train)



print('R-squared score (training): {:.3f}'.format(lin_reg.score(X_train, y_train)))

print('R-squared score (validation): {:.3f}'.format(lin_reg.score(X_valid, y_valid)))

LINREG = lin_reg.score(X_valid, y_valid)

LINREG_TRAIN = lin_reg.score(X_train, y_train)

# Polynomial regression



from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree = 2)

X_Poly = poly.fit_transform(X)

X_poly_train,X_poly_valid,y_poly_train,y_poly_valid = train_test_split(X_Poly,y,test_size=0.275,random_state=42)



poly_lin_reg = LinearRegression()

poly_lin_reg.fit(X_poly_train, y_poly_train)

print('R-squared score (training): {:.3f}'.format(poly_lin_reg.score(X_poly_train, y_poly_train)))

print('R-squared score (validation): {:.3f}'.format(poly_lin_reg.score(X_poly_valid, y_poly_valid)))



POLLINREG = poly_lin_reg.score(X_poly_valid, y_poly_valid)

POLLINREG_TRAIN = poly_lin_reg.score(X_poly_train, y_poly_train)
# let's try for degree 3 :



from sklearn.preprocessing import PolynomialFeatures

poly3 = PolynomialFeatures(degree = 3)

X_Poly3 = poly3.fit_transform(X)

X_poly3_train,X_poly3_valid,y_poly3_train,y_poly3_valid = train_test_split(X_Poly3,y,test_size=0.275,random_state=42)



poly3_lin_reg = LinearRegression()

poly3_lin_reg.fit(X_poly3_train, y_poly3_train)

print('R-squared score (training): {:.3f}'.format(poly3_lin_reg.score(X_poly3_train, y_poly3_train)))

print('R-squared score (validation): {:.3f}'.format(poly3_lin_reg.score(X_poly3_valid, y_poly3_valid)))



POLL3INREG = poly3_lin_reg.score(X_poly3_valid, y_poly3_valid)

POLL3INREG_TRAIN = poly3_lin_reg.score(X_poly3_train, y_poly3_train)
from sklearn.linear_model import Lasso



poly_lasso = Lasso(alpha=10)

poly_lasso.fit(X_poly_train, y_poly_train)



print('R-squared score (training): {:.3f}'.format(poly_lasso.score(X_poly_train, y_poly_train)))

print('R-squared score (validation): {:.3f}'.format(poly_lasso.score(X_poly_valid, y_poly_valid)))



LASREG = poly_lasso.score(X_poly_valid, y_poly_valid)

LASREG_TRAIN = poly_lasso.score(X_poly_train, y_poly_train)
from sklearn.linear_model import Ridge



poly_ridge = Ridge(alpha=300)

poly_ridge.fit(X_poly_train, y_poly_train)



print('R-squared score (training): {:.3f}'.format(poly_ridge.score(X_poly_train, y_poly_train)))

print('R-squared score (validation): {:.3f}'.format(poly_ridge.score(X_poly_valid, y_poly_valid)))



RIDREG = poly_ridge.score(X_poly_valid, y_poly_valid)

RIDREG_TRAIN = poly_ridge.score(X_poly_train, y_poly_train)
Models = pd.DataFrame({'Models' : ['Linear Regression','Linear Regression With polynomial(degree=2)','Linear Regression With polynomial(degree=3)'

                                   ,'Lasso','Ridge'],

                       'Training Score' : [LINREG_TRAIN,POLLINREG_TRAIN,POLL3INREG_TRAIN,LASREG_TRAIN,RIDREG_TRAIN],

                       'Validation Score' : [LINREG,POLLINREG,POLL3INREG,LASREG,RIDREG]

                      })

Models
df_test = pd.read_csv('../input/test.csv')

ID = df_test['Id']

df_test.head()
test_home_features =['OverallQual','OverallCond','YearBuilt','TotalBsmtSF','1stFlrSF', 'GrLivArea', 

'FullBath', 'TotRmsAbvGrd','GarageCars', 'GarageArea','Fireplaces', 'LotArea','YearRemodAdd']

X_test = df_test[test_home_features]

X_test.head()
X_test = X_test.fillna(method='ffill')
test_lasso_reg = Lasso(alpha=10).fit(X_Poly,y)

X_test_Poly = poly.fit_transform(X_test)

predictions  = test_lasso_reg.predict(X_test_Poly)

predictions[:10]
predictions = np.expm1(predictions) 

predictions[:10]
submissions = pd.DataFrame({'Id': ID,'SalePrice':predictions})

submissions.head()
submissions.to_csv('my_submission.csv', index=False)