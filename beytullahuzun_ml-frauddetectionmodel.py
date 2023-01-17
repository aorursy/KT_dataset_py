

import pandas as pd

fraud_data_path = '../input/credit_fraud_sytn.csv'

fraud_data = pd.read_csv(fraud_data_path)

print("Setup Complete")
fraud_data.columns
fraud_data.head()
y = fraud_data.isFradulent

feature_names = ["Average Amount/transaction/day", "Transaction_amount", "isForeignTransaction",

                      "isHighRiskCountry"]

X = fraud_data[feature_names]

X.describe()
X.head()
from sklearn.tree import DecisionTreeRegressor

#specify the model. 

#For model reproducibility, set a numeric value for random_state when specifying the model

fraud_model = DecisionTreeRegressor(random_state=1)

# Fit the model

fraud_model.fit(X, y)
print("Making predictions for the following 5 transactions:")

print(X.head())

print("The predictions are")

print(fraud_model.predict(X.head()))
from sklearn.metrics import mean_absolute_error



predicted_fraud = fraud_model.predict(X)

mean_absolute_error(y, predicted_fraud)
from sklearn.model_selection import train_test_split



# split data into training and validation data, for both features and target



train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

# Define model

fraud_model = DecisionTreeRegressor()

# Fit model

fraud_model.fit(train_X, train_y)



# get predicted prices on validation data

val_predictions = fraud_model.predict(val_X)

print(mean_absolute_error(val_y, val_predictions))