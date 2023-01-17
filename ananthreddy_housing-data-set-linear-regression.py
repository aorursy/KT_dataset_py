# Import the following packages pandas, numpy and sklearn.
# A Simple Linear Regression is used to predict the housing rates.
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the dataset '.../Housing.csv' by adding path file.
df = pd.read_csv("../input/Housing.csv")

# Replace the string 'yes' and 'no' with 1 and 0 for computatinal convenience.
df = df.replace(to_replace='yes',value=1,regex=True)
df = df.replace(to_replace='no',value=0,regex=True)

# Load the data to be predicted/output 'price' in X
X = df[['price']]
# Load the input parameters in y
y = df[['lotsize','bedrooms','stories','bathrms','bathrms','driveway','recroom','fullbase','gashw','airco','garagepl','prefarea']]

# Split the training and testing data in required ratio at test_size.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Perform Linear Regressiona dn fit the training examples in the model and predict the testing data.
model = linear_model.LinearRegression()
model.fit(y_train,X_train)
predicted_data = model.predict(y_test)

# Print the predicted data.
print (predicted_data)