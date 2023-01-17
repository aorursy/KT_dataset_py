import numpy as np

import matplotlib.pyplot as plt



import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression



from scipy import stats
column_names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]

data = pd.read_csv("../input/boston-housing.csv", header=None, delimiter=r"\s+", names=column_names)

print("\n\nData loaded\n\n")
data.head()
data.describe()
data = pd.DataFrame(np.c_[data['RM'],data['AGE'],data['MEDV']], columns = ['RM', 'AGE', 'MEDV'])



# Check null values

print("\n\nCheck null values\n",data.isnull().sum())
# Discovering outliers by Z-Score

ZScore = np.abs(stats.zscore(data))

print("\n\nChecking where outliers are less than the ZScore")

print("ZScore > 1\n",np.where(ZScore > 1)[0],"\n",np.where(ZScore > 1)[1],"\n")

print("ZScore > 2\n",np.where(ZScore > 2)[0],"\n",np.where(ZScore > 2)[1],"\n")

print("ZScore > 3\n",np.where(ZScore > 3)[0],"\n",np.where(ZScore > 3)[1],"\n")
data_o = data[(ZScore<3).all(axis=1)]

print ("Shape before removing outliers : ",np.shape(data),"\nShape after removing outliers : ",np.shape(data_o))
X = pd.DataFrame(np.c_[data_o['RM'],data_o['AGE']], columns = ['RM', 'AGE'])

Y = pd.DataFrame(np.c_[data_o['MEDV']], columns = ['MEDV'])

print("\n\nX =\n",X.head(5))

print("\n\nY =\n",Y.head(5))
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.25)

print("X_train.shape : ", X_train.shape, "\tX_test.shape", X_test.shape)

print("Y_train.shape : ", Y_train.shape, "\tY_train.shape", Y_train.shape)
lin_model = LinearRegression()

lin_model = lin_model.fit(X_train, Y_train)
predictions = lin_model.predict(X_test)



# Scatter Plot

plt.scatter(Y_test, predictions)

plt.xlabel("True Values",color='red')

plt.ylabel("Predictions",color='blue')

plt.title("Predicted vs Actual value")

plt.grid(True)

plt.show()

print(lin_model.score(X_test,Y_test))