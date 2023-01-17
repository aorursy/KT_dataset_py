import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn import linear_model
train_df = pd.read_csv('train.csv')

test_df = pd.read_csv('test.csv')

train_df = train_df.dropna()

test_df = test_df.dropna()
x_df = train_df.as_matrix(['x'])

y_df = train_df.as_matrix(['y'])
x_test = test_df.as_matrix(['x'])

y_test = test_df.as_matrix(['y'])
lm = linear_model.LinearRegression()
lm.fit(x_df,y_df)
lm.score(x_df,y_df)
y_predct = lm.predict(x_test)
plt.scatter(y_predct,y_test)

plt.show()