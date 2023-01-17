# Importing Libraries

import numpy as np                  # working with array 

import pandas as pd                 # import data set

import matplotlib.pyplot as plt     # for visualization

          
# Import Training Data Set

Training_Dataset = pd.read_csv("../input/random-linear-regression/train.csv")

Training_Dataset = Training_Dataset.dropna()

X_train = np.array(Training_Dataset.iloc[:, :-1].values) # Independent Variable

y_train = np.array(Training_Dataset.iloc[:, 1].values)   # Dependent Variable
# Import Testing Data Set

Testing_Dataset = pd.read_csv("../input/random-linear-regression/test.csv")

Testing_Dataset = Testing_Dataset.dropna()

X_test = np.array(Testing_Dataset.iloc[:, :-1].values)   # Independent Variable

y_test = np.array(Testing_Dataset.iloc[:, 1].values)     # Dependent Variable
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)
accuracy = regressor.score(X_test, y_test)

print('Accuracy = '+ str(accuracy))
plt.style.use('seaborn')

plt.scatter(X_test, y_test, color = 'red', marker = 'o', s = 35, alpha = 0.5,

          label = 'Test data')

plt.plot(X_train, regressor.predict(X_train), color = 'blue', label='Model Plot')

plt.title('Predicted Values vs Inputs')

plt.xlabel('Inputs')

plt.ylabel('Predicted Values')

plt.legend(loc = 'upper left')

plt.show()
dataset = pd.read_csv('../input/insurance/insurance.csv')

print(dataset)
X = dataset.iloc[:, :-1] # Independent Variable

y = dataset.iloc[:, -1]  # Dependent Variable
# We have to apply encoding in the dataset as there are words present.

# for 'sex' and 'smoker' column we will apply Label Encoding as there are only 2 catagories

# for 'region' we will apply OneHot Encoding as there are more than 2 catagories



# Label Encoding:

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

X.iloc[:, 1] = le.fit_transform(X.iloc[:, 1])

X.iloc[:, 4] = le.fit_transform(X.iloc[:, 4])



# OneHot Encoding:

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [5])], remainder='passthrough')

X = np.array(ct.fit_transform(X))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Training the Model

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)
accuracy = regressor.score(X_test, y_test)

print('Accuracy = '+ str(accuracy))
dataset = pd.read_csv('../input/polynomialregressioncsv/polynomial-regression.csv')

X = dataset.iloc[:, :-1] # Independent Variable

y = dataset.iloc[:, -1] # Dependent Variable
# Trianing the Model

from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree = 5)

X_poly = poly_reg.fit_transform(X)

lin_reg_2 = LinearRegression()

lin_reg_2.fit(X_poly, y)
plt.style.use('seaborn')

plt.scatter(X, y, color = 'red', marker = 'o', s = 35, alpha = 0.5,

          label = 'Test data')

plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue', label='Model Plot')

plt.title('Predicted Values vs Inputs')

plt.xlabel('Inputs')

plt.ylabel('Predicted Values')

plt.legend(loc = 'upper left')

plt.show()
Training_Dataset = pd.read_csv("../input/random-linear-regression/train.csv")

Training_Dataset = Training_Dataset.dropna()

X_train = np.array(Training_Dataset.iloc[:, :-1].values) # Independent Variable

y_train = np.array(Training_Dataset.iloc[:, 1].values) # Dependent Variable

y_train = y_train.reshape(len(y_train),1)
Testing_Dataset = pd.read_csv("../input/random-linear-regression/test.csv")

Testing_Dataset = Testing_Dataset.dropna()

X_test = np.array(Testing_Dataset.iloc[:, :-1].values) # Independent Variable

y_test = np.array(Testing_Dataset.iloc[:, 1].values) # Dependent Variable

y_test = y_test.reshape(len(y_test),1)
# Scalling X and y

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

sc_y = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.fit_transform(X_test)

y_train = sc_y.fit_transform(y_train)

y_test = sc_y.fit_transform(y_test)
# Training the Model

from sklearn.svm import SVR

regressor = SVR(kernel = 'rbf')

regressor.fit(X_train, y_train)
accuracy = regressor.score(X_test, y_test)

print('Accuracy = '+ str(accuracy))
plt.scatter(sc_X.inverse_transform(X_test), sc_y.inverse_transform(y_test), color = 'red', 

           marker = 'o', s = 35, alpha = 0.5, label = 'Test data')

plt.plot(sc_X.inverse_transform(X_test), sc_y.inverse_transform(regressor.predict(X_test)), 

           color = 'blue', label='Model Plot')

plt.title('Predicted Values vs Inputs')

plt.xlabel('Inputs')

plt.ylabel('Predicted Values')

plt.legend(loc = 'upper left')

plt.show()
Training_Dataset = pd.read_csv("../input/random-linear-regression/train.csv")

Training_Dataset = Training_Dataset.dropna()

X_train = np.array(Training_Dataset.iloc[:, :-1].values) # Independent Variable

y_train = np.array(Training_Dataset.iloc[:, 1].values) # Dependent Variable
Testing_Dataset = pd.read_csv("../input/random-linear-regression/test.csv")

Testing_Dataset = Testing_Dataset.dropna()

X_test = np.array(Testing_Dataset.iloc[:, :-1].values) # Independent Variable

y_test = np.array(Testing_Dataset.iloc[:, 1].values)   # Dependent Variable
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state = 0)

regressor.fit(X_train, y_train)
accuracy = regressor.score(X_test, y_test)

print('Accuracy = '+ str(accuracy))
X_grid = np.arange(min(X_test), max(X_test), 0.01)

X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X_test, y_test, color = 'red', marker = 'o', s = 35, alpha = 0.5,

          label = 'Test data')

plt.plot(X_grid, regressor.predict(X_grid), color = 'blue', label='Model Plot')

plt.title('Predicted Values vs Inputs')

plt.xlabel('Inputs')

plt.ylabel('Predicted Values')

plt.legend(loc = 'upper left')

plt.show()
Training_Dataset = pd.read_csv("../input/random-linear-regression/train.csv")

Training_Dataset = Training_Dataset.dropna()

X_train = np.array(Training_Dataset.iloc[:, :-1].values) # Independent Variable

y_train = np.array(Training_Dataset.iloc[:, 1].values) # Dependent Variable
Testing_Dataset = pd.read_csv("../input/random-linear-regression/test.csv")

Testing_Dataset = Testing_Dataset.dropna()

X_test = np.array(Testing_Dataset.iloc[:, :-1].values) # Independent Variable

y_test = np.array(Testing_Dataset.iloc[:, 1].values) # Dependent Variable
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)

regressor.fit(X_train, y_train)
accuracy = regressor.score(X_test, y_test)

print('Accuracy = '+ str(accuracy))
X_grid = np.arange(min(X_test), max(X_test), 0.01)

X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X_test, y_test, c = 'red', marker = 'o', s = 35, alpha = 0.5,

          label = 'Test data')

plt.plot(X_grid, regressor.predict(X_grid), color = 'blue', label='Model Plot')

plt.title('Predicted Values vs Inputs')

plt.xlabel('Position level')

plt.ylabel('Predicted Values')

plt.legend(loc = 'upper left')

plt.show()