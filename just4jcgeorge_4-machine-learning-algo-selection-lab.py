#Machine Learning Algorithm Selection
#Which version of Python are we running?

import sys



print(sys.version)
# Show us where our files are located and called what name

!ls "../input/"
# Data preparation

import pandas as pd

# Load data

melbourne_data = pd.read_csv('../input/melb_ML_data.csv')

melbourne_data
# Filter rows with missing values

melbourne_data = melbourne_data.dropna(axis=0)
# Choose target and features

y = melbourne_data.Price

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 

                        'YearBuilt', 'Lattitude', 'Longtitude']

X = melbourne_data[melbourne_features]
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error



# split data into training and validation data, for both features and target

# The split is based on a random number generator. Supplying a numeric value to

# the random_state argument guarantees we get the same split every time we

# run this script.

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state = 0)
#Python code for Linear Regression

from sklearn.linear_model import LinearRegression



model = LinearRegression()

model.fit(X_train, y_train)

predicted = model.predict(X_test)



print('Predicted ',predicted)

print('Mean Absolute Error ',mean_absolute_error(y_test, predicted))
#Python code for Logistic Regression

from sklearn.linear_model import LogisticRegression



model = LogisticRegression()

model.fit(X_train, y_train)

predicted = model.predict(X_test)



print('Predicted ',predicted)

print('Mean Absolute Error ',mean_absolute_error(y_test, predicted))
#Python code for K-Nearest Neighbors

NN_X = [[0], [1], [2], [3]]

NN_y = [0, 0, 1, 1]



from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=3)

model.fit(NN_X, NN_y)

predicted = model.predict([[1.1]])



print('Predicted ',predicted)

print('Predict Probad ',model.predict_proba([[0.9]]))
#Python code for Decision Tree

from sklearn.tree import DecisionTreeClassifier



model = DecisionTreeClassifier()

model.fit(X_train, y_train)

predicted = model.predict(X_test)



print('Predicted ',predicted)

print('Mean Absolute Error ',mean_absolute_error(y_test, predicted)) 
#Python code for Random Forest

from sklearn.ensemble import RandomForestRegressor



model = RandomForestRegressor(random_state=1)

model.fit(X_train, y_train)

predicted = model.predict(X_test)



print('Predicted ',predicted)

print('Mean Absolute Error ',mean_absolute_error(y_test, predicted)) 
#Python code for Gradient Boosting Machine

from sklearn.ensemble import GradientBoostingClassifier



model = GradientBoostingClassifier()

model.fit(X_train, y_train)

predicted = model.predict(X_test)



print('Predicted ',predicted)

print('Mean Absolute Error ',mean_absolute_error(y_test, predicted)) 
# End of Machine Learning Algorithm Selection