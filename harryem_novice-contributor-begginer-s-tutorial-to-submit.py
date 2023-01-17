import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
# Read data in from text file



# You will need to run this command on your computer

# data = pd.read_csv('house-prices-advanced-regression-techniques/train.csv')



# This one works on the live Kaggle notebook

data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")



# We can now look at the data in the notebook

display(data.head)
y = data['SalePrice']



X = data['LotArea']
plt.scatter(y, X)

plt.title("Sale Price vs Lot Area")

plt.xlabel("Lot Area")

plt.ylabel("Sale Price")

plt.show()
# Find the best straight line through those points

regressor = LinearRegression()  

regressor.fit(X.values.reshape(-1, 1), y)



# And visualize that line on our training data

training_predictions = regressor.predict(X.values.reshape(-1, 1))



plt.scatter(y, X)

plt.plot(training_predictions, X, color='red')

plt.title("Sale Price vs Lot Area")

plt.xlabel("Lot Area")

plt.ylabel("Sale Price")

plt.show()
# Load test data



# Run this one on your own computer

# data_test = pd.read_csv('house-prices-advanced-regression-techniques/test.csv')



# This one works on the live Kaggle Notebook

data_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')



# Extract Square Footage Feature

X_test = data_test['LotArea']



# Predict!

predictions = regressor.predict(X_test.values.reshape(-1, 1))
print(predictions[0:5])
# Create pandas dataframe in correct format

predictions_df = pd.DataFrame({'Id': data_test['Id'], 'SalePrice': predictions})



# Output your predictions as a csv

predictions_df.to_csv('predictions.csv',index=False)
# Double brackets are needed to extract multiple features

X = data[['LotArea','OverallQual']]



# Now reshape is not required becayse X is like a matrix

regressor.fit(X, y)



# Same for predict

X_test = data_test[['LotArea','OverallQual']]

predictions = regressor.predict(X_test)