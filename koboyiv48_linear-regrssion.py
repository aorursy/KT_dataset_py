import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import glob

import seaborn as sns

import matplotlib.pyplot as plt   #Data visualisation libraries 

%matplotlib inline
# import datasets

files = glob.glob("../input/*.csv")

list = []

for f in files:

    df = pd.read_csv(f,index_col=None)

    list.append(df)

df = pd.concat(list)
# drops columns

df = df.drop(['from_address','to_address','status','date'],axis=1)
df.head()
print(df.dtypes)
# import sklearn

from sklearn.model_selection import train_test_split
X = df[['open','high','low','volumefrom']]

y = df['close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# import Lib

from sklearn.linear_model import LinearRegression

from sklearn import metrics

from sklearn.metrics import r2_score
# build the model using sklearn

lm = LinearRegression()

lm.fit(X_train,y_train)

r2 = lm.score(X_train,y_train)

predictions = lm.predict(X_test)

print("R-squared :",r2)

print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

print(r2_score(y_test, predictions))

    
coef = pd.DataFrame(lm.coef_, X.columns, columns = ['Coefficients'])

coef
corr = X.corr()

corr
columns = np.full((corr.shape[0],), True, dtype=bool)

for i in range(corr.shape[0]):

    for j in range(i+1, corr.shape[0]):

        if corr.iloc[i,j] >= 0.7:

            if columns[j]:

                columns[j] = False

selected_columns = X.columns[columns]

data = X[selected_columns]
data.corr()
X = df[['open','volumefrom']]

y = df['close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# build the model using sklearn

lm = LinearRegression()

lm.fit(X_train,y_train)

r2 = lm.score(X_train,y_train)

predictions = lm.predict(X_test)

print("R-squared :",r2)

print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

print(r2_score(y_test, predictions))
lm.coef_

sns.regplot(y_test, predictions)