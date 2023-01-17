# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Scikit-Learn kullanarak Ridge Regresyon uygulaması:



# Önce rasggele veriler oluşturalım:

import numpy as np

import numpy.random as rnd

import matplotlib.pyplot as plt



np.random.seed(42)



m = 100

X = 6 * np.random.rand(m, 1) - 3

y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)



# Ridge Regresyon (kapalı-form çözüm kullanarak)

from sklearn.linear_model import Ridge

ridge_reg = Ridge(alpha=1, solver="cholesky")

ridge_reg.fit(X, y)

ridge_reg.predict([[1.5]])
# Ridge Regresyon (Stochastic Gradient Descent kullanarak):



from sklearn.linear_model import SGDRegressor

sgd_reg_ridge = SGDRegressor(penalty="l2")

sgd_reg_ridge.fit(X, y.ravel())

sgd_reg_ridge.predict([[1.5]])
# Lasso sınıfını kullanan küçük bir Scikit-Learn örneği:



from sklearn.linear_model import Lasso

lasso_reg = Lasso(alpha=0.1)

lasso_reg.fit(X,y)

lasso_reg.predict([[1.5]])
# Lasso Regresyon (Stochastic Gradient Descent kullanarak):



from sklearn.linear_model import SGDRegressor

sgd_reg_lasso = SGDRegressor(penalty="l1")

sgd_reg_lasso.fit(X, y.ravel())

sgd_reg_lasso.predict([[1.5]])
# Scikit-Learn’ün Elastik Net sınıfını kullanan kısa bir örnek (l1_ratio, r karışım oranına karşılık gelir):



from sklearn.linear_model import ElasticNet

elastic_net=ElasticNet(l1_ratio=0.5,alpha=0.1)

elastic_net.fit(X,y)

elastic_net.predict([[1.5]])