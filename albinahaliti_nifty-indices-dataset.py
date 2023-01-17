import pandas as pd

indices_data = pd.read_csv("../input/nifty-indices-dataset/NIFTY 50.csv")
indices_data.head()
indices_data.columns
y = indices_data.Low
indices_data = indices_data.dropna(axis=0)
indices_features = ['Low']
X = indices_data [indices_features]
X.describe()
X.head()
from sklearn.tree import DecisionTreeRegressor

indices_model = DecisionTreeRegressor(random_state=1)

indices_model.fit(X, y)
print("The Low 5 following numbers :")

print(X.head())

print("Numbers are")

print(indices_model.predict(X.head()))
from sklearn.metrics import mean_absolute_error



predicted_home_prices = indices_model.predict(X)

mean_absolute_error(y, predicted_home_prices)
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

melbourne_model = DecisionTreeRegressor()

# Fit model

indices_model.fit(train_X, train_y)



# get predicted prices on validation data

val_predictions = indices_model.predict(val_X)

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