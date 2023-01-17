import pandas as pd

married_data = pd.read_csv("../input/married-at-first-sight/mafs.csv")
married_data.head()
married_data.columns
y = married_data.Age
married_data = married_data.dropna(axis=0)
married_features = ['Age']
X = married_data [married_features]
X.describe()
X.head()
from sklearn.tree import DecisionTreeRegressor

married_model = DecisionTreeRegressor(random_state=1)

married_model.fit(X, y)
print("Age of 5 following people :")

print(X.head())

print("The age is")

print(married_model.predict(X.head()))
from sklearn.metrics import mean_absolute_error



predicted_home_prices = married_model.predict(X)

mean_absolute_error(y, predicted_home_prices)
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

melbourne_model = DecisionTreeRegressor()

# Fit model

married_model.fit(train_X, train_y)



# get predicted prices on validation data

val_predictions = married_model.predict(val_X)

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