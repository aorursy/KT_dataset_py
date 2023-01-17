import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
# Read the data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

test.tail(5)
# Convert String to Num

labelEncoder = LabelEncoder()

train.Neighborhood = labelEncoder.fit_transform(train.Neighborhood)

test.Neighborhood = labelEncoder.fit_transform(test.Neighborhood)

train.tail(5)


# pull data into target (y) and predictors (X)

train_y = train.SalePrice

predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd', '1stFlrSF']



# Create training predictors NAmesdata

train_X = train[predictor_cols]

test_X = test[predictor_cols]



my_model = RandomForestRegressor()

my_model.fit(train_X, train_y)



features = train_X.columns

importances = my_model.feature_importances_

indices = np.argsort(importances)



plt.figure(figsize=(6,6))

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), features[indices])

plt.show()
# Use the model to make predictions

predicted_prices = my_model.predict(test_X)



# We will look at the predicted prices to ensure we have something sensible.

print(predicted_prices)
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)