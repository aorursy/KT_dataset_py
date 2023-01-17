# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

price = [160,180,200,220,240,260,280]

sale = [126,103,82,75,82,40,20]

priceDF = pd.DataFrame(price, columns=list('x'))

saleDF = pd.DataFrame(sale, columns=list('y'))

houseDf = pd.concat((priceDF, saleDF),axis=1)

print(houseDf)

print(priceDF)
import statsmodels.api as sm

import statsmodels.formula.api as smf

smfModel = smf.ols('y~x',data=houseDf).fit()

print(smfModel.summary())
from sklearn.datasets import fetch_california_housing

from sklearn.datasets import load_boston

import pandas as pd 

boston = load_boston()

california = fetch_california_housing()

dataset = pd.DataFrame(boston.data, columns=boston.feature_names)

dataset['target'] = boston.target

print(dataset.head()) 
import pandas as pd

price = [160,180,200,220,240,260,280]

sale = [126,103,82,75,82,40,20]

cars = [0,9,19,5,25,1,20]

priceDF = pd.DataFrame(price, columns=list('x'))

saleDF = pd.DataFrame(sale, columns=list('y'))

carsDf = pd.DataFrame(cars, columns=list('z'))

houseDf = pd.concat([priceDF,saleDF,carsDf],axis=1)
import statsmodels.api as sm

import statsmodels.formula.api as smf



X = houseDf.drop(['y'], axis=1)

y = houseDf.y

Xc = sm.add_constant(X)

linear_regression = sm.OLS(y,Xc)

fitted_model = linear_regression.fit()

fitted_model.summary()
from sklearn.datasets import fetch_california_housing

from sklearn.datasets import load_boston

boston = load_boston()

california = fetch_california_housing()

dataset = pd.DataFrame(boston.data, columns=boston.feature_names)

dataset['target'] = boston.target