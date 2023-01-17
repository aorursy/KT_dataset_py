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
data=pd.read_csv('../input/climate-change/climate_change.csv')
data.head()
data.info()
data.isnull().sum()
data.columns
data_train=data[data['Year']<=2006]
data_train
data_test=data[data['Year']>2006]
data_test
data_train.info()
data_test.info()
import seaborn as sns

import matplotlib.pyplot as plt
corr = data.corr()

fig, ax = plt.subplots(figsize = (15,10))

g= sns.heatmap(corr,ax=ax, annot= True)

ax.set_title('Correlation between variables')


X_train=data.drop(['Month'],axis=1)
X_train=X_train.drop(['Temp'],axis=1)
X_train=X_train.drop(['Year'],axis=1)

X_train
Y_train=data['Temp']
Y_train




from sklearn import datasets,linear_model

from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train,Y_train)
print(model.coef_)
X_test=data_test.drop(['Temp'],axis=1)



Y_test=data_test['Temp']
X_test=X_test.drop(['Year'],axis=1)

X_test=X_test.drop(['Month'],axis=1)



Y_predt=model.predict(X_test)
Y_predt




print(mean_squared_error(Y_test,Y_predt))
print(r2_score(Y_test,Y_predt))
from statsmodels.api import OLS
OLS(Y_train,X_train).fit().summary()
model.score(X_train, Y_train)