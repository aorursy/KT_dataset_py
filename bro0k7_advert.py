# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
advert = pd.read_csv('/kaggle/input/advertising.csv/Advertising.csv')

advert.drop('Unnamed: 0',axis=1,inplace=True)

advert
sns.pairplot(advert)
advert['TV'] = np.log(advert['TV'])
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression



x = advert[['TV','radio','newspaper']]

y = advert['sales']



x_train,x_test,y_train,y_test = train_test_split(x, y, train_size=0.7, random_state=100)



lr = LinearRegression()

lr.fit(x_train,y_train)



print(lr.intercept_)

print(lr.coef_)



coeff = pd.DataFrame(lr.coef_, x_test.columns, columns = ['Coefficient'])



y_pred = lr.predict(x_test)
import statsmodels.api as sm

x_train_sm = sm.add_constant(x_train)

lm = sm.OLS(y_train, x_train_sm).fit()

print(lm.summary())
x_test_sm = sm.add_constant(x_test)

y_pred_lm = lm.predict(x_test_sm)
c = [i for i in range(1, 61, 1)]

plt.plot(c, y_test, color='blue')

plt.plot(c, y_pred_lm[:60], color='red')
c = [i for i in range(1, 61, 1)]

plt.figure()

a = np.linspace(0,60,1000)

plt.plot(c, y_test-y_pred, color='blue')

plt.plot(a,a*0,color='red')
lr.score(x_train, y_train)