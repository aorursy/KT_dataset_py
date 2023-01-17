import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import  matplotlib.pyplot as plt

df = pd.read_csv('../input/train.csv')
df.head()
df.info()
X = df[['budget','popularity']]

X.head()
y = df.revenue

y.head()
from sklearn import linear_model

model = linear_model.LinearRegression()
%matplotlib inline

plt.scatter(df.budget,df.revenue)

plt.scatter(df.popularity,df.revenue)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(df[['budget']],df[['revenue']])

model.fit(X_train,y_train)

model.score(X_test,y_test)
test = pd.read_csv('../input/test.csv')

test.head()
features = ['budget']

target = 'revenue'
predictions = model.predict(test[features])

predictions
submission = pd.DataFrame()

submission['id'] = test['id']

submission['revenue'] = predictions

submission.to_csv('submission.csv', index=False)
df = pd.read_csv('submission.csv')

df.head()