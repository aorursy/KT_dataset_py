import pandas as pd

import numpy as np



# Importing the dataset

dataset = pd.read_csv('../input/insurance.csv')

X = dataset.iloc[:, :-1].values

y = dataset.iloc[:, 6].values

dataset.head()



# Plot Dependent Variable (And notice how it's right skewed)

import matplotlib.pyplot as plt

plt.hist(y, bins=50)

plt.title('Dependent Variable histogram')

plt.show()



y = np.log(y) # Fix the skewness
# Feature Encoding

from sklearn.preprocessing import LabelEncoder, OneHotEncoder



# Sex

label_encoder_X_sex = LabelEncoder()

X[:, 1] = label_encoder_X_sex.fit_transform(X[:, 1])



# Smoker

label_encoder_X_smoker = LabelEncoder()

X[:, 4] = label_encoder_X_smoker.fit_transform(X[:, 4])



# Region

label_encoder_X_region = LabelEncoder()

X[:, 5] = label_encoder_X_region.fit_transform(X[:, 5])



# Hot Encode Region

one_hot_encoder = OneHotEncoder(categorical_features = [5]) # Creates columns (vectors) for categorical values

X = one_hot_encoder.fit_transform(X).toarray()



# Remove one dummy feature (dummy trap)

X = X[:, 1:]
# Split The dataset

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)
# Create Polynomial Features

from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree = 3)

X_poly = poly_features.fit_transform(X_train, y_train)
# Train

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_poly, y_train)
# R^2 Score on Test Set

poly = poly_features.fit_transform(X_test)

regressor.score(poly, y_test)
# Predict (Test)

y_pred = regressor.predict(poly)

plt.scatter(y_test, y_pred)

plt.xlabel('Actual Y')

plt.ylabel('Predicted Y')

plt.show()
# Ende 