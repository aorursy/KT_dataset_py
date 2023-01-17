# Kaggle Competition.
# Import libraries.
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
# Model features.
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
# Get train and test data.
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")

# Get train variables.
X = train_data[features]
y = train_data.SalePrice

# Get independent variables.
X1 = test_data[features]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes, random_state = 0)
    model.fit(train_X, train_y)
    preds_val= model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae
candidate_max_leaf_nodes = [5, 50, 500, 5000]
scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}
best_tree_size = min(scores, key = scores.get)
best_tree_size
tree_model = DecisionTreeRegressor(random_state = 0, max_leaf_nodes = best_tree_size)
tree_model.fit(X, y)
val_preds = tree_model.predict(X1)
output = pd.DataFrame({
    "Id": test_data.Id, 
    "SalePrice": val_preds
})
output
