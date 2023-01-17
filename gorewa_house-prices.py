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
# load training data file

train_data = pd.read_csv('../input/train.csv')

train_data = train_data.drop(train_data.index[0])

test_data = pd.read_csv('../input/test.csv')

# explore data

import seaborn as sns

# print missing values in X

#print(len(X)-X.count())

df = train_data[['LotArea','YearBuilt','FullBath','GrLivArea', 'BedroomAbvGr', 'TotRmsAbvGrd']]

sns.pairplot(df)

      

# coeffeciants of linear model

from sklearn import linear_model

from matplotlib  import pyplot as plt

feature_columns = ['YearBuilt', 'GrLivArea','FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

#Assign  to X subset of data with feature columns

X = train_data[feature_columns]

y = train_data['SalePrice']

X_test = test_data[feature_columns]

ridge = linear_model.RidgeCV()

ridge.fit(X,y)

#visualize coeffcients

coefs = ridge.coef_

plt.figure(figsize=(6,4))

plt.barh(np.arange(coefs.size), coefs)

plt.yticks(np.arange(coefs.size),feature_columns)

plt.title("Coefficients")

plt.tight_layout()
#scaling the coefficients

X_std = X.std()

plt.figure(figsize = (6,4))

plt.barh(np.arange(coefs.size), coefs*X_std)

plt.yticks(np.arange(coefs.size),feature_columns)

plt.title("Scaled coefficients")

plt.tight_layout()
# import sklearn linear regression

from sklearn.linear_model import LinearRegression

linreg = LinearRegression()

linreg.fit(X,y)

y_predict= linreg.predict(X_test)





# calculate error

from sklearn import metrics

print(np.sqrt(metrics.mean_squared_error(y,y_predict)))



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id,

                        'SalePrice': y_predict})



output.to_csv('submission.csv', index=False)
