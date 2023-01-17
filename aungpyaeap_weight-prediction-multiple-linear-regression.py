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



dataset = pd.read_csv('../input/Fish.csv', delimiter=',')

nRow, nCol = dataset.shape

print(f'There are {nRow} rows and {nCol} columns')
dataset.head(5)
dataset.hist(bins=5)
# correlation

import seaborn as sns

corr = dataset.corr()

ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

);

print(corr)
feature_cols = ['Species','Length1','Length2','Length3','Height','Width']

x = dataset[feature_cols]

y = dataset.Weight
# label encoding

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_encoder = LabelEncoder()

x.iloc[:,0] = label_encoder.fit_transform(x.iloc[:,0]) #LabelEncoder is used to encode the country value

hot_encoder = OneHotEncoder(categorical_features = [0])

x = hot_encoder.fit_transform(x).toarray()
# split dataset

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=1)
# fit linear regression

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train, y_train)



print("Coefficients: ",regressor.intercept_, regressor.coef_)
# predict

predict_val = regressor.predict(x_test)

print(predict_val)
# metrics

from sklearn import metrics

import numpy as np

print(metrics.mean_absolute_error(y_test, predict_val))

print(metrics.mean_squared_error(y_test, predict_val))

print(np.sqrt(metrics.mean_squared_error(y_test,predict_val)))



print("y error (difference between observed and predicted values) = ", y_test, predict_val)



from sklearn.metrics import r2_score

print("R square ", r2_score(y_test,predict_val))
from matplotlib import pyplot as plt

plt.scatter(y_test, predict_val, color='red')

plt.xlabel('Real weight', color='red')

plt.ylabel('Predicted weight', color='blue')

plt.plot(y_test, y_test + 1, '-o' , linestyle='solid',label='y=2x+1', color='blue')

plt.legend(loc='upper left')

plt.grid()

plt.show()
# optimal model using backward elimination

import numpy as np

import statsmodels.formula.api as sm

x = np.append(arr = np.ones((159, 1)).astype(int), values = x, axis = 1)

x_opt = x[:, [0,1,2,3,4,5,6,7,8,9,10,11,12]]

regressor_ols = sm.OLS(endog=y, exog=x_opt).fit()

regressor_ols.summary()