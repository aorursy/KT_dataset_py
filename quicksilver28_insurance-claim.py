import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df = pd.read_csv('../input/auto-insurance-in-sweden-small-dataset/insurance.csv', skiprows=5, names=['Claims','TotalPayment'])
df.info()
df.head()
from sklearn.model_selection import train_test_split
X = df.values[:,0].astype('int64')
#X = df.values[:,0]
y = df.values[:,1]
X_train, X_test, y_train, y_test = train_test_split(X.reshape(-1,1), y, test_size=0.2, random_state=1)
#print(X, y)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
#print(predictions)
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test, predictions))
from sklearn.tree import DecisionTreeRegressor
treemodel = DecisionTreeRegressor()
treemodel.fit(X_train, y_train)
treepredictions = treemodel.predict(X_test)
#print(treepredictions)
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test, treepredictions))