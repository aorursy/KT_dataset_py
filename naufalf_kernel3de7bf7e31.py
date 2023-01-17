# Import Libraries

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
# Load Training Data

train_path = 'train.csv'
train_data = pd.read_csv(train_path)
# Features
features = ['num_recharge_trx', 'num_topup_trx', 
            'num_transfer_trx', 'isActive',
            'min_recharge_trx', 'min_topup_trx', 'min_transfer_trx',
            'total_transaction', 'isUpgradedUser', 'num_transaction',
            'premium', 'blocked', 'pinEnabled', 'userLevel', 'isVerifiedPhone', 'isVerifiedEmail']

# Cleaning null data
clean_data = train_data[features + ['isChurned']].fillna(train_data[features + ['isChurned']].mean())

X = clean_data[features]
y = clean_data.isChurned

# Split data for training and validation
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
# Iterate through several max_leaf_nodes value for least mean absolute error
n = 64
for i in range(10):
    n = n * 2
    model = DecisionTreeRegressor(random_state=1, max_leaf_nodes=n, criterion='mse')
    model.fit(train_X, train_y)
    prediction = model.predict(val_X)
    prediction = map(lambda x: 1 if x >= 0.5 else 0, prediction)
    print(n, '\t\t', mean_absolute_error(val_y, list(prediction)))

# Final training model
model = DecisionTreeRegressor(random_state=1, max_leaf_nodes=512)
model.fit(train_X, train_y)
# Load test data
test_path = 'test.csv'
test_data = pd.read_csv(test_path)
test_X = test_data[features].fillna(test_data[features].mean())
# Predict using previous model
test_prediction = list(map(lambda x: 1 if x >= 0.5 else 0, model.predict(test_X)))
# Save to csv
pd.DataFrame({'isChurned' : pd.Series(test_prediction, index=test_data['idx'])}).to_csv('submission.csv')
