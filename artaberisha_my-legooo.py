import pandas as pd

colors_data = pd.read_csv("../input/lego-database/colors.csv")
colors_data.head()
colors_data.columns
y = colors_data.id
colors_data = colors_data.dropna(axis=0)
colors_features = ['id']
X = colors_data [colors_features]
X.describe()
X.head()
from sklearn.tree import DecisionTreeRegressor

colors_model = DecisionTreeRegressor(random_state=1)

colors_model.fit(X, y)

print("Making predictions for the following 5 houses:")

print(X.head())

print("The predictions are")

print(colors_model.predict(X.head()))
from sklearn.metrics import mean_absolute_error



predicted_home_prices = colors_model.predict(X)

mean_absolute_error(y, predicted_home_prices)
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

melbourne_model = DecisionTreeRegressor()

# Fit model

colors_model.fit(train_X, train_y)



# get predicted prices on validation data

val_predictions = colors_model.predict(val_X)

print(mean_absolute_error(val_y, val_predictions))
from sklearn.metrics import mean_absolute_error

from sklearn.tree import DecisionTreeRegressor



def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):

    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)

    model.fit(train_X, train_y)

    preds_val = model.predict(val_X)

    mae = mean_absolute_error(val_y, preds_val)

    return(mae)
# compare MAE with differing values of max_leaf_nodes

for max_leaf_nodes in [5, 50, 500, 5000]:

    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)

    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



forest_model = RandomForestRegressor(random_state=1)

forest_model.fit(train_X, train_y)

melb_preds = forest_model.predict(val_X)

print(mean_absolute_error(val_y, melb_preds))