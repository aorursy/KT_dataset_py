import matplotlib.pylab as plt

import numpy as np

%matplotlib inline

from sklearn.linear_model import LinearRegression

from sklearn import datasets
diabetes = datasets.load_diabetes() # load data
diabetes.data.shape # feature matrix shape
diabetes.target.shape # target vector shape
diabetes.feature_names # column names
# Sperate train and test data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.2, random_state=0)
# Sperate train and test data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.2, random_state=0)

# There are three steps to model something with sklearn

# 1. Set up the model

model = LinearRegression()

# 2. Use fit

model.fit(X_train, y_train)

# 3. Check the score

model.score(X_test, y_test)
model.coef_ # Get the coefficients, beta
model.intercept_ # Get the intercept, c
model.predict(X_test) # Predict unkown data
# plot prediction and actual data

y_pred = model.predict(X_test) 

plt.plot(y_test, y_pred, '.')



# plot a line, a perfit predict would all fall on this line

x = np.linspace(0, 330, 100)

y = x

plt.plot(x, y)

plt.show()