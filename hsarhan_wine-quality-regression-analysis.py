import numpy as np

import pandas as pd
df = pd.read_csv('../input/winequality-red.csv')
df.head()
df.info()
df.describe()
import matplotlib.pyplot as plt

%matplotlib inline

df.hist(bins = 50, figsize = (20,15))

plt.show()
corr_matrix = df.corr()

corr_matrix['quality'].sort_values(ascending = False)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df_scaled = scaler.fit_transform(df)

df_scaled = pd.DataFrame(df_scaled, columns = df.columns)
from sklearn.model_selection import train_test_split

X = df_scaled.drop(['quality'], axis = 1)

y = df_scaled['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
from sklearn.linear_model import LinearRegression



lm = LinearRegression()

lm.fit(X_train, y_train)
predictions = lm.predict(X_test)
from sklearn.metrics import mean_squared_error

lm_mse = mean_squared_error(y_test,predictions)

lm_rmse = np.sqrt(lm_mse)

lm_rmse
from sklearn.tree import DecisionTreeRegressor



tree = DecisionTreeRegressor()

tree.fit(X_train, y_train)

predictions = tree.predict(X_test)

tree_mse = mean_squared_error(y_test, predictions)

tree_rmse = np.sqrt(tree_mse)

tree_rmse