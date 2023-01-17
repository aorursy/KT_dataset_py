import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression



droptrain_df = pd.read_csv('../input/train.csv')

droptest_df = pd.read_csv('../input/test.csv')



train_df = droptrain_df.dropna()

test_df = droptest_df.dropna()



X_train = np.array(train_df.iloc[:, :-1].values)

y_train = np.array(train_df.iloc[:, 1].values)



X_test = np.array(test_df.iloc[:, :-1].values)

y_test = np.array(test_df.iloc[:, 1].values)



model = LinearRegression()

model.fit(X_train, y_train)



y_pred = model.predict(X_test)

accuracy = model.score(X_test, y_test)



plt.scatter(X_train, y_train, color='blue')

plt.scatter(X_test, y_test, color='red')

plt.plot(X_train, model.predict(X_train), color='green')

plt.show()

print(accuracy)