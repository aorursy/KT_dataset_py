# Importing the libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
# Importing the dataset

dataset = pd.read_csv('../input/startups-dataset/50_Startups.csv')

X = dataset.iloc[:, :-1].values

y = dataset.iloc[:, 4].values


# Encoding categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder = LabelEncoder()

X[:, 3] = labelencoder.fit_transform(X[:, 3])



onehotencoder = OneHotEncoder(categorical_features = [3])

X = onehotencoder.fit_transform(X).toarray()
# Avoiding the Dummy Variable Trap

X = X[:, 1:]
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)


# Predicting the Test set results

y_pred = regressor.predict(X_test)

y_pred_1 = regressor.predict(X_train)
#calculationg mape



def mape_test (y_true,y_p):

    y_true,y_p = np.array(y_test),np.array(y_pred)

    return np.mean(np.abs((y_true-y_p)/y_true))


#Plot Administration vs Profit

x1 = dataset.iloc[:, 1].values

y1 = dataset.iloc[:, -1].values

plt.scatter(x1,y1,color='Red',s=50)

plt.xlabel('Administration')

plt.ylabel('Profit')

plt.title('Administration vs Profit')

plt.show()
#Plot State vs Profit

x1 = dataset.iloc[:, -2].values

y1 = dataset.iloc[:, -1].values

plt.scatter(x1,y1,color='Blue',s=50)

plt.xlabel('State')

plt.ylabel('Profit')

plt.title('State vs Profit')

plt.show()


#Plot Marketing Spend vs Profit

x1 = dataset.iloc[:, 2].values

y1 = dataset.iloc[:, -1].values

plt.scatter(x1,y1,color='Black',s=50)

plt.xlabel('Marketing Spend')

plt.ylabel('Profit')

plt.title('Marketing Spend vs Profit')

plt.show()
